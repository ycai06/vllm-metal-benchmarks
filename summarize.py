#!/usr/bin/env python3
"""Validate a completed local benchmark suite and write results.csv and REPORT.md.

Usage: .venv/bin/python summarize.py results/RUN_DIRECTORY [--overwrite]
Only derived reports are written; raw measurements, logs and summaries are untouched.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3
MIB = 1024 ** 2


def number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def optional_max(values):
    values = [v for v in values if number(v)]
    return max(values) if values else None


def fmt(value, decimals=1):
    return f"{value:,.{decimals}f}" if number(value) else ""


def flag(command, key, default=None):
    try:
        return command[command.index(key) + 1]
    except (ValueError, IndexError):
        return default


def load(path):
    return json.loads(path.read_text())


def cache_slots(log):
    """Read announced capacity, never infer physical allocation from RSS."""
    patterns = [r"max_tokens_cached[=:]\s*([\d,]+)",
                r"(?:GPU|MLX|Metal) KV cache size:\s*([\d,]+) tokens",
                r"KV cache size:\s*([\d,]+) tokens"]
    for pattern in patterns:
        matches = re.findall(pattern, log)
        if matches:
            return int(matches[-1].replace(",", ""))
    return None


def classify(name):
    for prefix in ("weight", "batch", "context", "capacity"):
        if name.startswith(prefix):
            return prefix
    return "other"


def read_trial(path):
    summary = load(path)
    prefix = Path(str(path).removesuffix(".summary.json"))
    raw = Path(str(prefix) + ".jsonl")
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    args = summary["arguments"]
    measured = [r for r in rows if not r["warmup"]]
    failures = []
    for i, row in enumerate(rows):
        usage = row.get("usage") or {}
        if row.get("error"):
            failures.append(f"row {i}: {row['error']}")
        if usage.get("prompt_tokens") != args["input_tokens"]:
            failures.append(f"row {i}: prompt count mismatch")
        if usage.get("completion_tokens") != args["output_tokens"]:
            failures.append(f"row {i}: output count mismatch")
        if "generated_token_ids" in row and len(row["generated_token_ids"]) != usage.get("completion_tokens"):
            failures.append(f"row {i}: streamed token IDs disagree with usage")
        if not number(row.get("ttft_s")) or row["ttft_s"] < 0:
            failures.append(f"row {i}: missing or invalid TTFT")
        if not number(row.get("latency_s")) or (number(row.get("ttft_s")) and row["latency_s"] < row["ttft_s"]):
            failures.append(f"row {i}: invalid latency")
    expected = args["requests"] * args["repeats"]
    if len(measured) != expected:
        failures.append(f"expected {expected} measured rows; found {len(measured)}")
    identities = [(r["repeat"], r["request_id"]) for r in measured]
    if len(set(identities)) != len(identities):
        failures.append("duplicate measured repeat/request IDs")
    if summary["overall"].get("errors") != 0:
        failures.append("summary reports errors")
    token_sum = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in measured)
    if token_sum != summary["overall"].get("output_tokens"):
        failures.append("raw output token total disagrees with summary")
    if len(measured) != summary["overall"].get("successful_requests"):
        failures.append("raw request count disagrees with summary")
    if len(summary.get("repeats", [])) != args["repeats"]:
        failures.append("repeat summary count mismatch")
    wall = sum(r["wall_s"] for r in summary["repeats"])
    if not number(wall) or wall <= 0:
        failures.append("invalid measured wall duration")
    if failures:
        raise ValueError(str(raw) + "\n" + "\n".join(failures[:20]))

    command_path = path.parent / "server-command.json"
    command = load(command_path) if command_path.exists() else []
    text = " ".join(command) + " " + path.parent.name
    bits_match = re.search(r"Qwen3-0\.6B-(4|8)bit|\bq(4|8)\b", text)
    bits = int(next(g for g in bits_match.groups() if g)) if bits_match else None
    budget = flag(command, "--kv-cache-memory-bytes")
    resources = summary.get("resources", {})
    gauges = resources.get("request_gauge_peak_by_metric", {})
    preemptions = resources.get("preemptions_delta_by_metric", {})
    log = (path.parent / "server.log").read_text(errors="replace") if (path.parent / "server.log").exists() else ""
    return {"name": prefix.name, "session": path.parent.name, "path": path,
            "category": classify(prefix.name), "bits": bits,
            "concurrency": args["concurrency"], "input_tokens": args["input_tokens"],
            "output_tokens": args["output_tokens"], "requests": len(measured),
            "warmup_requests": len(rows) - len(measured), "rows": measured,
            "wall_s": wall, "output_token_total": token_sum,
            "throughput": token_sum / wall,
            "repeat_throughputs": [r["aggregate_output_tokens_per_s"] for r in summary["repeats"]],
            "ttft_ms": statistics.median(r["ttft_s"] for r in measured) * 1000,
            "latency_ms": statistics.median(r["latency_s"] for r in measured) * 1000,
            "rss_bytes": resources.get("peak_process_tree_rss_bytes"),
            "cache_peak": optional_max(resources.get("kv_cache_usage_peak_by_metric", {}).values()),
            "running_peak": optional_max(v for k, v in gauges.items() if "num_requests_running" in k),
            "waiting_peak": optional_max(v for k, v in gauges.items() if "num_requests_waiting" in k),
            "preemptions": sum(preemptions.values()) if preemptions else None,
            "kv_budget_bytes": int(budget) if budget else None,
            "cache_slots": cache_slots(log), "command": command,
            "monitor_errors": resources.get("monitor_errors", [])}


def combined(trials):
    rows = [r for t in trials for r in t["rows"]]
    wall = sum(t["wall_s"] for t in trials)
    repeats = [v for t in trials for v in t["repeat_throughputs"] if number(v)]
    return {"requests": len(rows), "throughput": sum(t["output_token_total"] for t in trials) / wall,
            "ttft_ms": statistics.median(r["ttft_s"] for r in rows) * 1000,
            "latency_ms": statistics.median(r["latency_s"] for r in rows) * 1000,
            "repeat_min": min(repeats), "repeat_max": max(repeats),
            "rss_gib": (rss / GIB if (rss := optional_max(t["rss_bytes"] for t in trials)) is not None else None),
            "cache_peak": optional_max(t["cache_peak"] for t in trials),
            "running_peak": optional_max(t["running_peak"] for t in trials),
            "waiting_peak": optional_max(t["waiting_peak"] for t in trials),
            "preemptions": sum(t["preemptions"] for t in trials) if all(t["preemptions"] is not None for t in trials) else None,
            "cache_slots": optional_max(t["cache_slots"] for t in trials)}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def report(result_dir, trials, models):
    def select(category, key, value):
        return [t for t in trials if t["category"] == category and t[key] == value]
    sections = ["# vLLM-Metal experiments on this Mac", "",
                "Measured on Apple M2 Pro (16 GPU cores, 32 GiB unified memory), using vLLM-Metal 0.33.0 and MLX. "
                "The Mac remained on AC power with Low Power Mode enabled. These are exploratory results for Qwen3-0.6B and this workload.", "",
                f"Validated **{sum(t['requests'] for t in trials)} measured requests** and "
                f"{sum(t['warmup_requests'] for t in trials)} excluded warmup requests: no recorded request errors; "
                "every raw request has the exact requested input/output token counts. "
                "Aggregate throughput is total output tokens divided by the sum of measured repeat durations; group medians use individual raw requests.", ""]
    quant = {}
    quant_rows = []
    for bits in (4, 8):
        selected = select("weight", "bits", bits)
        if not selected:
            continue
        s = combined(selected)
        quant[bits] = s
        files = models.get(str(bits), {}).get("files", {})
        weight_bytes = sum(v["bytes"] for k, v in files.items() if k.endswith(".safetensors")) or None
        quant_rows.append([f"{bits}-bit", s["requests"], fmt(weight_bytes / MIB if weight_bytes else None),
                           fmt(s["throughput"]), fmt(s["ttft_ms"]),
                           f"{fmt(s['repeat_min'])}–{fmt(s['repeat_max'])}", fmt(s["rss_gib"], 2)])
    sections += ["## Weight quantization", "", "256 input tokens, 64 output tokens, one concurrent request; independent warmed servers in 4/8/8/4 order. "
                 "Both use affine quantization with group size 64, the same architecture and one explicitly fixed tokenizer.", "",
                 table(["Weights", "Measured requests", "Weight files MiB", "Output tok/s", "Median TTFT ms", "Repeat tok/s range", "Peak process RSS GiB"], quant_rows), ""]
    if 4 in quant and 8 in quant:
        ratio = quant[4]["throughput"] / quant[8]["throughput"]
        sections += [f"The 4-bit run delivered {ratio:.2f}× the 8-bit output throughput in this test. "
                     "The weight-file sizes show storage savings directly; process RSS includes the server, runtime, shared pages and fixed KV pool. "
                     "This experiment does not establish output-quality equivalence.", ""]
        overlap = max(quant[4]["repeat_min"], quant[8]["repeat_min"]) <= min(quant[4]["repeat_max"], quant[8]["repeat_max"])
        if 0.9 <= ratio <= 1.1 and overlap:
            sections += ["The speed gap is small and the per-repeat throughput ranges overlap. "
                         "This pilot does not establish a robust speed advantage for either precision.", ""]

    weight_load_rows = []
    weight_loads = {}
    for bits in (4, 8):
        path = result_dir / f"weights-q{bits}.json"
        if not path.exists():
            continue
        measurement = load(path)
        if measurement.get("bits") != bits:
            raise ValueError(f"Supplemental weight precision mismatch: {path}")
        weight_loads[bits] = measurement
        values = [measurement.get("logical_tensor_bytes"),
                  measurement.get("unique_parameter_buffer_bytes"),
                  measurement.get("active_allocation_delta_bytes"),
                  measurement.get("after_evaluated_load", {}).get("peak_since_preload_reset_bytes")]
        if any(value is not None and (not number(value) or value < 0) for value in values):
            raise ValueError(f"Invalid supplemental weight allocation: {path}")
        weight_load_rows.append([f"{bits}-bit"] + [fmt(value / MIB if value is not None else None, 2) for value in values])
    if weight_load_rows:
        sections += ["## Supplemental isolated weight allocation", "",
                     "Each checkpoint was loaded and evaluated in a fresh MLX process after the serving suite. "
                     "Checkpoint dtypes were preserved, with no cast. This measurement excludes the vLLM server, KV cache and inference activations; "
                     "it does not measure serving throughput or process RSS.", "",
                     table(["Weights", "Logical parameter tensors MiB", "Unique parameter buffers MiB", "Active allocation increase MiB", "Peak active allocation during load MiB"], weight_load_rows), "",
                     "Logical tensor bytes sum each parameter array, including packed weights, scales, biases and unquantized parameters; shared backing storage can be counted more than once. "
                     "Unique buffer bytes use MLX's backing-buffer accounting. Active allocation excludes cached allocator buffers. "
                     "The peak is measured after resetting immediately before load and can include transient loader/model-initialization arrays.", ""]
        if 4 in weight_loads and 8 in weight_loads:
            q4 = weight_loads[4].get("unique_parameter_buffer_bytes")
            q8 = weight_loads[8].get("unique_parameter_buffer_bytes")
            if number(q4) and number(q8) and q8 > 0:
                sections += [f"The evaluated 4-bit parameter buffers used {q4/q8:.2f}× the 8-bit buffer storage "
                             f"({100*(1-q4/q8):.1f}% less) for this checkpoint pair. This direct weight measurement is separate from the serving RSS comparison.", ""]

    batch = {}
    batch_rows = []
    for c in (1, 4, 8):
        selected = select("batch", "concurrency", c)
        if not selected:
            continue
        s = combined(selected)
        batch[c] = s
        batch_rows.append([c, s["requests"], fmt(s["throughput"]), fmt(s["ttft_ms"]), fmt(s["latency_ms"]),
                           f"{fmt(s['repeat_min'])}–{fmt(s['repeat_max'])}", fmt(s["running_peak"], 0),
                           fmt(s["cache_peak"] * 100 if s["cache_peak"] is not None else None)])
    sections += ["## Concurrent requests", "", "4-bit weights; 1,024 input tokens and 64 output tokens. Client concurrency changes while the server sequence cap stays at eight. "
                 "Forward and reversed condition orders are combined.", "",
                 table(["Concurrency", "Requests", "Output tok/s", "Median TTFT ms", "Median total latency ms", "Repeat tok/s range", "Observed running requests", "Sampled peak KV %"], batch_rows), ""]
    if 1 in batch and 8 in batch:
        sections += [f"Eight concurrent requests produced {batch[8]['throughput']/batch[1]['throughput']:.2f}× aggregate throughput versus one. "
                     "The latency columns show the corresponding per-request cost. Running-request gauges support that requests overlapped; "
                     "they do not identify the exact number of rows in every GPU batch.", ""]

    context = {}
    context_rows = []
    config = models.get("4", {}).get("config", {})
    cfg = config.get("text_config", config)
    try:
        kv_bytes_per_token = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * cfg["head_dim"] * 2
    except KeyError:
        kv_bytes_per_token = None
    for length in (256, 1024, 4096):
        selected = select("context", "input_tokens", length)
        if not selected:
            continue
        s = combined(selected)
        context[length] = s
        ideal = (length + 64) * kv_bytes_per_token / MIB if kv_bytes_per_token else None
        context_rows.append([f"{length:,}", s["requests"], fmt(s["ttft_ms"]), fmt(s["throughput"]),
                             fmt(s["cache_peak"] * 100 if s["cache_peak"] is not None else None),
                             fmt(ideal), fmt(s["rss_gib"], 2)])
    sections += ["## Context length and KV use", "", "4-bit weights, one concurrent request, 64 output tokens, and a fixed 1 GiB KV budget. "
                 "The input length changes; forward/reversed orders are combined.", "",
                 table(["Input tokens", "Requests", "Median TTFT ms", "Output tok/s", "Sampled peak KV %", "Ideal final sequence KV MiB", "Peak process RSS GiB"], context_rows), ""]
    if 256 in context and 4096 in context:
        sections += [f"Moving from 256 to 4,096 input tokens changed median TTFT from {context[256]['ttft_ms']:.1f} to {context[4096]['ttft_ms']:.1f} ms. "
                     "Longer inputs add prefill work and retain more key/value data. Pool capacity is fixed in advance, but physical pages are populated lazily; the byte budget is not a resident-memory reservation. Sampled RSS is not reliable occupied-KV accounting.", ""]
    if kv_bytes_per_token:
        sections += [f"With 28 attention layers, eight KV heads, head dimension 128 and FP16 KV, the ideal storage is "
                     f"{kv_bytes_per_token:,} bytes/token ({kv_bytes_per_token/1024:.0f} KiB), before block/layout overhead. "
                     "The table's ideal final KV includes the requested 64 generated tokens; sampled utilization is relative to the configured pool and may miss its exact peak.", ""]

    capacity_rows = []
    capacity = {}
    budgets = sorted({t["kv_budget_bytes"] for t in trials if t["category"] == "capacity" and t["kv_budget_bytes"] is not None})
    for budget in budgets:
        selected = select("capacity", "kv_budget_bytes", budget)
        s = combined(selected)
        capacity[budget] = s
        capacity_rows.append([fmt(budget / GIB, 0), fmt(s["cache_slots"], 0), s["requests"], fmt(s["throughput"]),
                              fmt(s["ttft_ms"]), fmt(s["preemptions"], 0), fmt(s["running_peak"], 0),
                              fmt(s["cache_peak"] * 100 if s["cache_peak"] is not None else None), fmt(s["waiting_peak"], 0)])
    sections += ["## KV capacity pressure", "", "4-bit weights, four concurrent requests, 4,096 input tokens and 64 output tokens; only the KV budget changes. "
                 "The cache token capacity below is read from each startup log when available.", "",
                 table(["KV budget GiB", "Announced token slots", "Requests", "Output tok/s", "Median TTFT ms", "Preemptions", "Observed running requests", "Sampled peak KV %", "Observed waiting requests"], capacity_rows), ""]
    if len(capacity) == 2:
        smaller, larger = (capacity[b] for b in budgets)
        sections += [f"The larger pool delivered {larger['throughput']/smaller['throughput']:.2f}× throughput in this capacity test. "
                     "Four complete 4,160-token sequences require about 1.78 GiB of ideal FP16 KV. "
                     "The 1 GiB condition intentionally creates capacity pressure; the 2 GiB condition can retain more simultaneous context. "
                     f"Observed running requests increased from {fmt(smaller['running_peak'], 0)} to {fmt(larger['running_peak'], 0)}. "
                     "Preemption counters and waiting gauges distinguish cache pressure from weight quantization effects.", ""]

    warnings = [f"{t['session']}/{t['name']}: {'; '.join(t['monitor_errors'])}" for t in trials if t["monitor_errors"]]
    sections += ["## Controls and limits", "",
                 "- Synthetic repeated prose with seeded varied prefixes (seed 713); exact token-ID prompts, greedy generation, `ignore_eos`, and fixed 64-token outputs. Prefix caching is disabled.",
                 "- Server controls: float16 model/KV dtype, maximum sequence length 4,608, sequence cap eight, prefill token budget 1,024, and a 1 GiB KV pool except the explicit capacity comparison. Chunked prefill and runtime defaults are captured in startup logs.",
                 "- Throughput includes prefill and HTTP work. TTFT uses the first generated token-ID event and includes local queueing/network time; stream events can contain multiple tokens.",
                 "- Process-tree RSS can count shared pages and is not GPU allocation or occupied KV bytes. Cache percentages are sampled fractions of each pool, so compare percentages across different pool sizes carefully.",
                 "- Three repetitions per condition and this small model provide a learning baseline. Tail latencies, larger models, different power settings and other Mac chips need separate tests.",
                 "- The weight order is 4/8/8/4, but its spacing is unequal: batching/context trials occur between the first 4-bit and first 8-bit measurements. "
                 "This order helps reveal drift but does not fully control elapsed time, background activity or temperature.",
                 "- Output text and token IDs are retained for inspection, but accuracy, reasoning quality and quantization quality were not evaluated. No general 4-bit speed or quality claim follows.",
                 "- The official installation overrides vLLM core's Torch metadata pin with a newer pinned nightly. "
                 "The resulting single dependency-check incompatibility is recorded; GPU computation, smoke testing and measured requests provide runtime verification. "
                 "See [dependency-check output](../../provenance/pip-check.txt) and [official source notes](../../provenance/official-sources.md).",
                 "- Commands, package/model provenance, per-request JSONL, per-repeat summaries, metrics and server logs accompany these tables. Raw artifacts are untouched by this report.", ""]
    construction_failures = []
    for path in result_dir.glob("*/context-c1-p4096.client.log"):
        text = path.read_text(errors="replace")
        if "Generated prompt was shorter than requested" in text and not Path(str(path).removesuffix(".client.log") + ".summary.json").exists():
            construction_failures.append(path)
    if construction_failures:
        sections += ["A first 4,096-token attempt stopped during prompt construction before sending warmup or measured requests. "
                     "The generator was corrected to extend text before exact-length truncation, and the suite resumed while preserving earlier completed trials. "
                     "Its setup failure is retained in the original client log and is excluded from request counts; the successful resumed trial is included above. "
                     "See the [lab README](../../README.md) for the correction and preservation details.", ""]
    if warnings:
        sections += ["Resource monitoring limitations:", ""] + ["- " + w for w in warnings] + [""]
    before, after = result_dir / "suite-state-before.json", result_dir / "suite-state-after.json"
    if before.exists() and after.exists():
        b, a = load(before), load(after)
        def local_time(value):
            return datetime.fromisoformat(value).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%b %d, %Y %I:%M:%S %p %Z")
        sections += [f"Suite local interval: {local_time(b['utc'])} to {local_time(a['utc'])}. "
                     f"Available memory before/after: {fmt(b.get('memory_available_bytes', 0)/GIB, 2)}/{fmt(a.get('memory_available_bytes', 0)/GIB, 2)} GiB. "
                     f"Swap used before/after: {fmt(b.get('swap_used_bytes', 0)/GIB, 2)}/{fmt(a.get('swap_used_bytes', 0)/GIB, 2)} GiB.", ""]
    sections += [f"Report generated {datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()}. Missing metrics appear as blank cells, not zeros.", ""]
    return "\n".join(sections)


def csv_text(trials):
    fields = ["session", "trial", "category", "weight_bits", "concurrency", "input_tokens", "output_tokens",
              "measured_requests", "warmup_requests", "output_tokens_total", "measured_wall_s", "aggregate_output_tok_s",
              "median_ttft_ms", "median_latency_ms", "repeat_tok_s_min", "repeat_tok_s_max", "peak_process_tree_rss_bytes",
              "kv_budget_bytes", "announced_cache_token_slots", "sampled_cache_usage_fraction_peak", "running_requests_peak",
              "waiting_requests_peak", "preemptions_delta", "raw_summary"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for t in trials:
        writer.writerow(dict(zip(fields, [t["session"], t["name"], t["category"], t["bits"], t["concurrency"],
            t["input_tokens"], t["output_tokens"], t["requests"], t["warmup_requests"], t["output_token_total"], t["wall_s"],
            t["throughput"], t["ttft_ms"], t["latency_ms"], min(t["repeat_throughputs"]), max(t["repeat_throughputs"]),
            t["rss_bytes"], t["kv_budget_bytes"], t["cache_slots"], t["cache_peak"], t["running_peak"], t["waiting_peak"],
            t["preemptions"], str(t["path"].relative_to(t["path"].parent.parent))])))
    return stream.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--overwrite", action="store_true", help="Replace only derived REPORT.md and results.csv")
    args = p.parse_args()
    directory = args.results.resolve()
    if not (directory / "suite-state-after.json").exists():
        p.error("Suite is not marked complete: suite-state-after.json is missing")
    outputs = [directory / "REPORT.md", directory / "results.csv"]
    if not args.overwrite and any(x.exists() for x in outputs):
        p.error("Derived report already exists; use --overwrite to regenerate it")
    paths = sorted(directory.glob("*/*.summary.json"))
    if not paths:
        p.error("No trial summaries found")
    trials = [read_trial(path) for path in paths]
    # Same request index must use the same prompt for equal-length conditions.
    hashes = {}
    for t in trials:
        for r in t["rows"]:
            key = (t["input_tokens"], r["request_id"])
            digest = r.get("input_token_ids_sha256")
            if key in hashes and hashes[key] != digest:
                raise ValueError(f"Prompt hash differs across equal-length trials: {key}")
            hashes[key] = digest
    manifest = ROOT / "provenance/models.json"
    models = load(manifest) if manifest.exists() else {}
    # Validate and render both outputs completely before writing either.
    report_text = report(directory, trials, models)
    data_text = csv_text(trials)
    outputs[0].write_text(report_text)
    outputs[1].write_text(data_text)
    print(f"Validated {len(trials)} trials, {sum(t['requests'] for t in trials)} measured requests")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
