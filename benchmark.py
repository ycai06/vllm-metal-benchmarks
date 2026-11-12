#!/usr/bin/env python3
"""Benchmark a local vLLM server with exact-length token prompts and streamed output.

Run only one benchmark client at a time. Disable server prefix caching, use identical
server settings and a local tokenizer for comparisons, and keep the Mac on AC power.
Client TTFT includes HTTP/queue time; streamed chunks are not necessarily single tokens.
Process RSS is not Metal allocation or occupied KV-cache memory. Metrics sampling adds
a small, consistent load and is enabled for every run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8011")
    p.add_argument("--model", required=True, help="Server's served model name")
    p.add_argument("--tokenizer", help="Local tokenizer directory; otherwise use /tokenize")
    p.add_argument("--output", type=Path, required=True, help="Prefix for .jsonl/.summary.json/.metrics.txt")
    p.add_argument("--concurrency", type=int, choices=(1, 4, 8), default=1)
    p.add_argument("--input-tokens", type=int, default=256)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--requests", type=int, default=24, help="Measured requests per repeat")
    p.add_argument("--warmups", type=int, default=1, help="Unmeasured requests per worker")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--server-pid", type=int, help="Optional parent server PID for RSS sampling")
    p.add_argument("--timeout", type=float, default=600, help="Per-request timeout in seconds")
    p.add_argument("--sample-interval", type=float, default=0.5)
    a = p.parse_args()
    if min(a.input_tokens, a.output_tokens, a.requests, a.repeats) < 1:
        p.error("Token counts, requests, and repeats must be positive")
    if a.requests < a.concurrency or a.warmups < 0 or a.sample_interval <= 0:
        p.error("Requests must be >= concurrency; warmups >= 0; sample interval > 0")
    a.base_url = a.base_url.rstrip("/").removesuffix("/v1")
    return a


def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * pct / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def distribution(values):
    return {"mean": statistics.mean(values) if values else None,
            "p50": percentile(values, 50), "p95": percentile(values, 95)}


async def get_text(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            text = await r.text()
            return text if r.status == 200 else f"# HTTP {r.status}: {text[:500]}"
    except Exception as e:
        return f"# Metrics unavailable: {type(e).__name__}: {e}"


def selected_metrics(metrics):
    """Preserve labels: cache usage is a fraction even where named *_perc."""
    found = {}
    for line in metrics.splitlines():
        if line.startswith("#") or not re.search(
                r"(?:gpu_cache_usage_perc|kv_cache_usage_perc|num_requests_running|num_requests_waiting|num_preemptions_total)", line):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            try:
                found[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return found


async def build_prompts(session, args, count):
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                                  trust_remote_code=False)

    async def encode(text):
        if tokenizer is not None:
            return tokenizer.encode(text, add_special_tokens=False)
        async with session.post(args.base_url + "/tokenize",
                                json={"model": args.model, "prompt": text,
                                      "add_special_tokens": False}) as r:
            r.raise_for_status()
            return (await r.json())["tokens"]

    # Tokenization happens before any timed workload. Vary a leading prefix to
    # prevent accidental shared-prefix benefits if server configuration is wrong.
    base = ("The local library has books about rivers, plants, machines, and music. "
            "Readers compare observations and write clear notes. ")
    rng = random.Random(args.seed)
    base_tokens = await encode(base)
    repetitions = math.ceil(args.input_tokens / max(1, len(base_tokens))) + 2
    prompts = []
    for i in range(count):
        prefix = f"Record {rng.randrange(10**8, 10**9)} sample {i}. "
        text = prefix + base * repetitions
        tokens = await encode(text)
        # Tokens can merge at repeated-text boundaries, so an estimate from an
        # isolated sentence is insufficient for long prompts.
        while len(tokens) < args.input_tokens:
            text += base * repetitions
            tokens = await encode(text)
        prompts.append(tokens[:args.input_tokens])
    return prompts


async def request(session, args, tokens, request_id, repeat, warmup):
    payload = {"model": args.model, "prompt": tokens,
               "max_tokens": args.output_tokens, "temperature": 0,
               "seed": args.seed, "ignore_eos": True,
               "stream": True, "return_token_ids": True,
               "stream_options": {"include_usage": True}}
    started = time.perf_counter()
    first = last_content = None
    usage = None
    chunks = 0
    generated_ids = []
    first_event_tokens = None
    output = []
    reason = None
    error = None
    try:
        async with session.post(args.base_url + "/v1/completions", json=payload,
                                timeout=aiohttp.ClientTimeout(total=args.timeout)) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:2000]}")
            async for raw in response.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    token_ids = choice.get("token_ids") or []
                    if token_ids:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                            first_event_tokens = len(token_ids)
                        last_content = now
                        generated_ids.extend(token_ids)
                    piece = choice.get("text") or ""
                    if piece:
                        chunks += 1
                        output.append(piece)
                    if choice.get("finish_reason"):
                        reason = choice["finish_reason"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    if error is None:
        if usage is None:
            error = "No terminal usage record; completion length cannot be verified"
        elif usage.get("prompt_tokens") != args.input_tokens:
            error = f"Prompt length mismatch: {usage.get('prompt_tokens')} != {args.input_tokens}"
        elif usage.get("completion_tokens") != args.output_tokens:
            error = f"Output length mismatch: {usage.get('completion_tokens')} != {args.output_tokens}"
        elif first is None:
            error = "No generated token IDs observed"
        elif len(generated_ids) != args.output_tokens:
            error = f"Stream token count mismatch: {len(generated_ids)} != {args.output_tokens}"
    return {"request_id": request_id, "repeat": repeat, "warmup": warmup,
            "input_token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "input_tokens_requested": args.input_tokens,
            "output_tokens_requested": args.output_tokens,
            "usage": usage, "finish_reason": reason,
            "latency_s": finished - started,
            "ttft_s": first - started if first is not None else None,
            "decode_span_s": last_content - first if first is not None else None,
            "generated_token_ids": generated_ids, "first_token_event_count": first_event_tokens,
            "nonempty_stream_chunks": chunks, "text": "".join(output), "error": error}


async def run_batch(session, args, prompts, repeat, warmup=False):
    gate = asyncio.Event()
    rows = []

    async def worker(worker_id):
        await gate.wait()
        for i in range(worker_id, len(prompts), args.concurrency):
            rows.append(await request(session, args, prompts[i], i, repeat, warmup))

    tasks = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
    await asyncio.sleep(0)
    started = time.perf_counter()
    gate.set()
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda r: r["request_id"])
    return rows, elapsed


def summarize(rows, wall):
    good = [r for r in rows if r["error"] is None]
    output_tokens = sum(r["usage"]["completion_tokens"] for r in good)
    return {"requests": len(rows), "successful_requests": len(good),
            "errors": len(rows) - len(good), "wall_s": wall,
            "output_tokens": output_tokens,
            "aggregate_output_tokens_per_s": output_tokens / wall if wall else None,
            "ttft_s": distribution([r["ttft_s"] for r in good]),
            "latency_s": distribution([r["latency_s"] for r in good]),
            "decode_span_s": distribution([r["decode_span_s"] for r in good]),
            "note": "Throughput includes input prefill and HTTP overhead. TTFT uses first generated token-ID event. Decode span is first-to-last token-ID event timing; events may group tokens."}


async def main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {"raw": Path(str(args.output) + ".jsonl"),
             "summary": Path(str(args.output) + ".summary.json"),
             "metrics": Path(str(args.output) + ".metrics.txt")}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite results: " + ", ".join(existing))
    psutil = None
    if args.server_pid:
        import psutil as psutil_module
        psutil = psutil_module
    resources = {"peak_process_tree_rss_bytes": None,
                 "kv_cache_usage_peak_by_metric": {},
                 "request_gauge_peak_by_metric": {},
                 "preemptions_delta_by_metric": {}, "samples": 0,
                 "monitor_errors": [],
                 "note": "RSS includes CPU/possibly shared pages and is not GPU allocation or occupied KV memory; KV usage gauge is a fraction of configured logical pool capacity, whose physical pages are populated lazily."}
    stop = asyncio.Event()
    measured = asyncio.Event()
    connector = aiohttp.TCPConnector(limit=max(32, args.concurrency + 4))
    async with aiohttp.ClientSession(connector=connector) as session:
        before = await get_text(session, args.base_url + "/metrics")
        n_warm = args.warmups * args.concurrency
        prompts = await build_prompts(session, args, max(args.requests, n_warm))

        async def monitor():
            while not stop.is_set():
                if measured.is_set():
                    metrics = await get_text(session, args.base_url + "/metrics")
                    resources["samples"] += 1
                    for key, value in selected_metrics(metrics).items():
                        group = ("kv_cache_usage_peak_by_metric" if "cache_usage_perc" in key
                                 else "request_gauge_peak_by_metric")
                        if "num_preemptions_total" not in key:
                            resources[group][key] = max(resources[group].get(key, value), value)
                    if psutil is not None:
                        try:
                            parent = psutil.Process(args.server_pid)
                            procs = [parent] + parent.children(recursive=True)
                            rss = sum(p.memory_info().rss for p in procs if p.is_running())
                            resources["peak_process_tree_rss_bytes"] = max(
                                resources["peak_process_tree_rss_bytes"] or 0, rss)
                        except Exception as exc:
                            msg = f"{type(exc).__name__}: {exc}"
                            if msg not in resources["monitor_errors"]:
                                resources["monitor_errors"].append(msg)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=args.sample_interval)
                except asyncio.TimeoutError:
                    pass

        all_rows, repeats = [], []
        if n_warm:
            warm_rows, _ = await run_batch(session, args, prompts[:n_warm], -1, True)
            all_rows.extend(warm_rows)
            if any(r["error"] for r in warm_rows):
                paths["raw"].write_text("".join(json.dumps(r) + "\n" for r in all_rows))
                raise RuntimeError("Warmup failed; inspect " + str(paths["raw"]))
        measured_before = await get_text(session, args.base_url + "/metrics")
        task = asyncio.create_task(monitor())
        try:
            for repeat in range(args.repeats):
                measured.set()
                rows, wall = await run_batch(session, args, prompts[:args.requests], repeat)
                measured.clear()
                all_rows.extend(rows)
                repeats.append(summarize(rows, wall))
                print(json.dumps({"repeat": repeat, **repeats[-1]}), flush=True)
        finally:
            stop.set()
            await task
        after = await get_text(session, args.base_url + "/metrics")
        prior_metrics = selected_metrics(measured_before)
        for key, value in selected_metrics(after).items():
            if "num_preemptions_total" in key and key in prior_metrics:
                resources["preemptions_delta_by_metric"][key] = value - prior_metrics[key]
    rows = [r for r in all_rows if not r["warmup"]]
    summary = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "workload": "Closed-loop, synchronized workers; fixed total requests per repeat; warmups excluded; deterministic exact-length token IDs; greedy fixed-length output with ignore_eos.",
               "overall": summarize(rows, sum(r["wall_s"] for r in repeats)),
               "repeats": repeats, "resources": resources}
    paths["raw"].write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    paths["metrics"].write_text("# BEFORE WARMUP\n" + before
                               + "\n# AFTER WARMUP / BEFORE MEASURED\n" + measured_before
                               + "\n# AFTER MEASURED\n" + after)
    print(json.dumps({"files": {k: str(v) for k, v in paths.items()},
                      "overall": summary["overall"]}), flush=True)
    if summary["overall"]["errors"]:
        raise RuntimeError("Measured requests failed; inspect saved raw results")


if __name__ == "__main__":
    try:
        asyncio.run(main(arguments()))
    except KeyboardInterrupt:
        sys.exit(130)
