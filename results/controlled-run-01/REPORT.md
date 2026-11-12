# vLLM-Metal experiments on this Mac

Measured on Apple M2 Pro (16 GPU cores, 32 GiB unified memory), using vLLM-Metal 0.33.0 and MLX. The Mac remained on AC power with Low Power Mode enabled. These are exploratory results for Qwen3-0.6B and this workload.

Validated **504 measured requests** and 88 excluded warmup requests: no recorded request errors; every raw request has the exact requested input/output token counts. Aggregate throughput is total output tokens divided by the sum of measured repeat durations; group medians use individual raw requests.

## Weight quantization

256 input tokens, 64 output tokens, one concurrent request; independent warmed servers in 4/8/8/4 order. Both use affine quantization with group size 64, the same architecture and one explicitly fixed tokenizer.

| Weights | Measured requests | Weight files MiB | Output tok/s | Median TTFT ms | Repeat tok/s range | Peak process RSS GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 4-bit | 48 | 319.9 | 100.3 | 113.8 | 94.7–104.9 | 2.42 |
| 8-bit | 48 | 604.1 | 93.1 | 117.4 | 82.5–99.7 | 2.75 |

The 4-bit run delivered 1.08× the 8-bit output throughput in this test. The weight-file sizes show storage savings directly; process RSS includes the server, runtime, shared pages and fixed KV pool. This experiment does not establish output-quality equivalence.

The speed gap is small and the per-repeat throughput ranges overlap. This pilot does not establish a robust speed advantage for either precision.

## Supplemental isolated weight allocation

Each checkpoint was loaded and evaluated in a fresh MLX process after the serving suite. Checkpoint dtypes were preserved, with no cast. This measurement excludes the vLLM server, KV cache and inference activations; it does not measure serving throughput or process RSS.

| Weights | Logical parameter tensors MiB | Unique parameter buffers MiB | Active allocation increase MiB | Peak active allocation during load MiB |
| --- | --- | --- | --- | --- |
| 4-bit | 319.84 | 319.84 | 319.84 | 319.84 |
| 8-bit | 604.02 | 604.03 | 604.03 | 604.03 |

Logical tensor bytes sum each parameter array, including packed weights, scales, biases and unquantized parameters; shared backing storage can be counted more than once. Unique buffer bytes use MLX's backing-buffer accounting. Active allocation excludes cached allocator buffers. The peak is measured after resetting immediately before load and can include transient loader/model-initialization arrays.

The evaluated 4-bit parameter buffers used 0.53× the 8-bit buffer storage (47.0% less) for this checkpoint pair. This direct weight measurement is separate from the serving RSS comparison.

## Concurrent requests

4-bit weights; 1,024 input tokens and 64 output tokens. Client concurrency changes while the server sequence cap stays at eight. Forward and reversed condition orders are combined.

| Concurrency | Requests | Output tok/s | Median TTFT ms | Median total latency ms | Repeat tok/s range | Observed running requests | Sampled peak KV % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 96 | 59.6 | 485.5 | 1,044.0 | 55.7–64.0 | 1 | 11.6 |
| 4 | 96 | 83.9 | 1,461.8 | 3,002.8 | 79.6–88.6 | 4 | 46.6 |
| 8 | 96 | 88.6 | 2,031.6 | 5,617.1 | 83.5–93.5 | 8 | 93.2 |

Eight concurrent requests produced 1.49× aggregate throughput versus one. The latency columns show the corresponding per-request cost. Running-request gauges support that requests overlapped; they do not identify the exact number of rows in every GPU batch.

## Context length and KV use

4-bit weights, one concurrent request, 64 output tokens, and a fixed 1 GiB KV budget. The input length changes; forward/reversed orders are combined.

| Input tokens | Requests | Median TTFT ms | Output tok/s | Sampled peak KV % | Ideal final sequence KV MiB | Peak process RSS GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 24 | 117.4 | 95.2 | 3.4 | 35.0 | 2.43 |
| 1,024 | 24 | 474.8 | 61.8 | 11.6 | 119.0 | 2.43 |
| 4,096 | 24 | 2,756.4 | 19.2 | 44.5 | 455.0 | 2.43 |

Moving from 256 to 4,096 input tokens changed median TTFT from 117.4 to 2756.4 ms. Longer inputs add prefill work and retain more key/value data. Pool capacity is fixed in advance, but physical pages are populated lazily; the byte budget is not a resident-memory reservation. Sampled RSS is not reliable occupied-KV accounting.

With 28 attention layers, eight KV heads, head dimension 128 and FP16 KV, the ideal storage is 114,688 bytes/token (112 KiB), before block/layout overhead. The table's ideal final KV includes the requested 64 generated tokens; sampled utilization is relative to the configured pool and may miss its exact peak.

## KV capacity pressure

4-bit weights, four concurrent requests, 4,096 input tokens and 64 output tokens; only the KV budget changes. The cache token capacity below is read from each startup log when available.

| KV budget GiB | Announced token slots | Requests | Output tok/s | Median TTFT ms | Preemptions | Observed running requests | Sampled peak KV % | Observed waiting requests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 9,360 | 24 | 19.0 | 9,885.8 | 0 | 2 | 89.0 | 3 |
| 2 | 18,720 | 24 | 19.7 | 4,743.4 | 0 | 4 | 89.0 | 3 |

The larger pool delivered 1.04× throughput in this capacity test. Four complete 4,160-token sequences require about 1.78 GiB of ideal FP16 KV. The 1 GiB condition intentionally creates capacity pressure; the 2 GiB condition can retain more simultaneous context. Observed running requests increased from 2 to 4. Preemption counters and waiting gauges distinguish cache pressure from weight quantization effects.

## Controls and limits

- Synthetic repeated prose with seeded varied prefixes (seed 713); exact token-ID prompts, greedy generation, `ignore_eos`, and fixed 64-token outputs. Prefix caching is disabled.
- Server controls: float16 model/KV dtype, maximum sequence length 4,608, sequence cap eight, prefill token budget 1,024, and a 1 GiB KV pool except the explicit capacity comparison. Chunked prefill and runtime defaults are captured in startup logs.
- Throughput includes prefill and HTTP work. TTFT uses the first generated token-ID event and includes local queueing/network time; stream events can contain multiple tokens.
- Process-tree RSS can count shared pages and is not GPU allocation or occupied KV bytes. Cache percentages are sampled fractions of each pool, so compare percentages across different pool sizes carefully.
- Three repetitions per condition and this small model provide a learning baseline. Tail latencies, larger models, different power settings and other Mac chips need separate tests.
- The weight order is 4/8/8/4, but its spacing is unequal: batching/context trials occur between the first 4-bit and first 8-bit measurements. This order helps reveal drift but does not fully control elapsed time, background activity or temperature.
- Output text and token IDs are retained for inspection, but accuracy, reasoning quality and quantization quality were not evaluated. No general 4-bit speed or quality claim follows.
- The official installation overrides vLLM core's Torch metadata pin with a newer pinned nightly. The resulting single dependency-check incompatibility is recorded; GPU computation, smoke testing and measured requests provide runtime verification. See [dependency-check output](../../provenance/pip-check.txt) and [official source notes](../../provenance/official-sources.md).
- Commands, package/model provenance, per-request JSONL, per-repeat summaries, metrics and server logs accompany these tables. Raw artifacts are untouched by this report.

A first 4,096-token attempt stopped during prompt construction before sending warmup or measured requests. The generator was corrected to extend text before exact-length truncation, and the suite resumed while preserving earlier completed trials. Its setup failure is retained in the original client log and is excluded from request counts; the successful resumed trial is included above. See the [lab README](../../README.md) for the correction and preservation details.

Suite local interval: Nov 12, 2026 04:31:35 AM PST to Nov 12, 2026 04:55:28 AM PST. Available memory before/after: 13.08/13.59 GiB. Swap used before/after: 0.44/0.44 GiB.

Report generated 2026-11-12T05:08:42.677274-08:00. Missing metrics appear as blank cells, not zeros.
