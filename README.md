# vLLM-Metal benchmarks

This isolated lab uses vLLM-Metal **0.33.0**, the official compatible vLLM core, MLX **0.32.4**, native Python **3.12.8**, and a matched Qwen3-0.6B MLX 4-bit/8-bit pair. Hardware: **M2 Pro, 16 GPU cores, 32 GB unified memory, macOS 27.0.1**. The Mac was connected to AC with Low Power Mode enabled. Power settings were preserved.

The existing `~/.venv-vllm-metal`, Qwen3.5-9B model, launch commands, and Hugging Face cache are preserved. This lab keeps its own environment, model copies, caches, downloads, and results. Benchmark servers listen only at **127.0.0.1:8011** and stop after each block. Nothing is configured to start at login.

## Results and records

The completed run is in `results/controlled-run-01/`. Read the [results report](results/controlled-run-01/REPORT.md) or [per-trial CSV](results/controlled-run-01/results.csv), and each block includes raw request JSONL, metrics, exact launch/client commands, startup logs, and before/after machine state. The separate `results/smoke/` is an excluded smoke test. The earlier `results/controlled-run/` records an aborted start caused by the smoke test's closing socket; no measurements were taken there.

The first 4,096-token trial stopped during prompt construction because repeated text tokenized shorter than estimated. It sent no requests. The generator now extends text until it has enough tokens, then truncates to the exact length. The 256/1,024-token inputs were independently verified unchanged. The resumed 4,096-token trial runs in `block-1-q4-resume-1/`; original logs and completed measurements remain intact.

`provenance/` contains model revisions and file hashes, official release metadata and wheel digests, package versions, the fixed package replay list, hardware/GPU verification, dependency check output, source links, and summary preservation checks. The machine-specific original file inventory is excluded from this repository.

## Run again

Clone the repository and reconstruct the recorded environment first:

```bash
git clone https://github.com/ycai06/vllm-metal-benchmarks.git
cd vllm-metal-benchmarks
bash replay_environment.sh

# Use the reconstructed environment. A new timestamped results folder
# is created, and existing results are never overwritten.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python run_suite.py

# After the suite stops its servers, measure each precision in a fresh process.
# Replace TIMESTAMP with the folder printed when the suite finishes.
.venv/bin/python measure_weights.py --bits 4 --output results/TIMESTAMP/weights-q4.json
.venv/bin/python measure_weights.py --bits 8 --output results/TIMESTAMP/weights-q8.json
.venv/bin/python summarize.py results/TIMESTAMP
```

If a full suite is interrupted, resume its incomplete folder without replacing successful trials:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python run_suite.py --resume --output results/TIMESTAMP
```

To reconstruct the measured versions in this lab after removing or relocating its environment:

```bash
bash replay_environment.sh
```

`replay_environment.sh` uses installed `uv` and native Python 3.12. Set `UV_BIN` or `PYTHON_BIN` if their paths differ. It installs the full recorded package list without resolving new dependency versions, downloads pinned model revisions, and verifies wheel SHA256 against official release metadata. `setup.sh` documents the original, version-pinned official wheel installation sequence; its transitive dependencies may resolve differently on a future first installation. Both scripts operate inside this lab only.

For one small verification run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python run_suite.py --smoke
```

To inspect one condition manually after the suite ends, start a server in one Terminal window:

```bash
bash serve_local.sh 4  # Use 8 for the other checkpoint. Control-C stops it.
```

In a second window in this lab directory, run an exact-length trial:

```bash
.venv/bin/python benchmark.py --model bench --tokenizer models/Qwen3-0.6B-4bit \
  --concurrency 4 --input-tokens 1024 --output-tokens 64 \
  --requests 16 --repeats 3 --warmups 2 --seed 713 --output results/manual-c4
```

This manual command saves timings and cache metrics; the automated suite additionally supplies its server PID for RSS sampling. Choose a new output prefix each time.

## What changes in each experiment

| Experiment | Changes | Held fixed |
|---|---|---|
| Weight quantization | 4-bit versus 8-bit, order 4/8/8/4 | Group size 64, architecture, tokenizer, 256 input tokens, 64 generated tokens, concurrency 1, 1 GiB KV pool |
| Concurrent requests | 1, 4, 8 clients; repeat in reverse order | 4-bit weights, server sequence cap 8, 1,024 input / 64 output tokens, 1 GiB KV pool |
| Context length | 256, 1,024, 4,096 input tokens; repeat in reverse order | 4-bit weights, concurrency 1, 64 output tokens, 1 GiB KV pool |
| KV capacity | 1 GiB versus 2 GiB configured KV capacity | 4-bit weights, concurrency 4, 4,096 input / 64 output tokens |

Every trial has three measured repeats after two excluded warmup requests per client. Quantization has 48 measured requests per precision; concurrency has 96 per setting; context has 24 per setting; capacity has 24 per budget. There are **504 measured requests** in the full suite.

All servers use FP16 execution/KV storage, maximum sequence length 4,608, sequence cap 8, prefill chunks up to 1,024 tokens, no prefix caching, no speculative decoding, and default Model Runner V2/MLX sampling. The same deterministic input-token sequences are used across matching conditions (seed 713), greedy generation is fixed to 64 tokens with EOS ignored, and every token count is checked. The two checkpoint tokenizer configuration files differ only in their chat templates. Both servers explicitly use the 4-bit directory's tokenizer, and the measured completions endpoint receives token IDs without applying a chat template.

The 4-bit/8-bit setting comes from the saved MLX checkpoint's quantization configuration. `--dtype float16` controls floating-point execution and the uncompressed KV cache; packed low-bit weights still remain quantized. There is no need to convert or overwrite the existing 9B model to run this pair.

## Read the numbers

- **Output tokens/second** is total generated tokens divided by workload wall time. It includes input processing, queueing, and local HTTP overhead. It is not decode-only speed.
- **First-token latency (TTFT)** runs from client submission until the first streamed generated token-ID event. A streamed event can contain multiple tokens; raw records preserve that count.
- **Request latency** is end-to-end time until the response finishes. More concurrency can improve total throughput while increasing a person's wait.
- **RSS** is the sampled sum of the experiment server process tree's resident memory. Shared pages can be counted more than once. It is not a direct measurement of Metal allocations or occupied KV bytes.
- **Isolated weight allocation** comes from a separate fresh MLX process per precision, after serving tests end. The model's complete parameter arrays are evaluated, their byte sizes inventoried, and MLX allocator readings recorded. It excludes vLLM, KV pools, and inference activations; checkpoint dtypes are preserved. Load time is recorded but is not a controlled cold-start benchmark.
- **KV usage** is the sampled fraction of configured pool capacity in use. Capacity is fixed in advance, but physical pages are populated lazily; the byte budget is not a resident-memory reservation. RSS does not directly measure occupied KV storage. Compare utilization fractions with their corresponding pool size. Raw resource notes that say “preallocated pool” refer to its logical capacity.

For this model, ordinary FP16 KV storage is `2 × 28 layers × 8 KV heads × 128 head dimension × 2 bytes = 114,688 bytes/token` (112 KiB/token), before block/layout overhead. Four requests with 4,096-token prompts and 64-token outputs need about **1.78 GiB** of KV slots. The capacity experiment deliberately puts pressure on the 1 GiB pool and compares it with 2 GiB. Startup logs record actual cache blocks/token capacity; counters record any preemptions. Weight quantization does not change that KV geometry. KV quantization/TurboQuant is a separate experiment and is not enabled here.

These are exploratory measurements of a small model and a synthetic workload on a working Mac. Low Power Mode and background activity affect absolute speed; forward/reverse order and repeated trials help reveal drift but do not eliminate it. Tail percentiles from these small samples are descriptive. Output text is saved, but this suite does not evaluate answer quality or establish equivalence between precisions. Server gauges show concurrent scheduled requests, not direct instrumentation of individual GPU batch rows.

The precision order is 4/8/8/4, but it is not equally spaced in time: other conditions ran after the first 4-bit baseline. Treat a small throughput gap as exploratory, especially when per-repeat ranges overlap.

## Installation detail

The official v0.33.0 installer installs a pinned PyTorch nightly after vLLM. Core's metadata still requires an older Torch, so the recorded dependency check reports exactly one Torch version incompatibility. This is the documented official override, retained here; Metal computation, the inference smoke test, and all measured requests provide runtime verification. See `provenance/pip-check.txt` and `provenance/official-sources.md`.

Core logs may say `device_config=cpu` or `CPU KV cache size`: these are compatibility labels used by the Metal plugin. The worker's `MLX device set to: Device(gpu, 0)` and `provenance/gpu-verification.json` confirm that model computation uses the Apple GPU.

The suite checks available memory before startup and during trials, stops its own work if available memory falls below 4 GiB, and refuses to replace another service on its port. It temporarily prevents idle sleep while its server runs. Stop with Control-C to terminate only this suite's clients/servers.

## Repository export

The original lab remains on the measured Mac. This copy includes scripts, validated request records, reports, and provenance. Absolute local paths in historical records are replaced with `<LAB>`, `<HOME>`, or `<TMP>`; timing, token IDs, prompt hashes, and measurements are unchanged. Historical command JSON is evidence of the original run; use the supplied scripts for runnable commands on a new checkout. Virtual environments, caches, downloads, model weights, and the private preserved-file inventory are excluded. See [publication notes](provenance/publication-notes.md) and [next experiments](docs/next-experiments.md).
