# Next experiments (planned, not measured)

The published pilot found 47% smaller 4-bit weight buffers, an inconclusive precision speed gap, 1.49x throughput at concurrency eight with higher request latency, and lower first-token latency with a larger KV budget.

First repeat controlled Mac comparisons with Low Power Mode disabled, then a larger model and separate short-input/long-output and long-input/short-output workloads. Record the new power condition and preserve previous results. Choose model sizes and KV budgets against actual available memory.

A rented NVIDIA GPU would be a separate standard-vLLM/CUDA experiment; vLLM-Metal targets Apple Silicon. Keep input/output lengths, model base revision and sampling controlled, choose supported quantization formats, and record backend differences. Cross-platform differences combine hardware and software effects. Run the load client beside the rented server to exclude internet round trips from first-token timing. Measure throughput per dollar with the actual session bill. No rental, cloud benchmark, or full-power rerun has been performed for this repository.

Sources: [vLLM platform installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) and [vLLM-Metal](https://docs.vllm.ai/projects/vllm-metal/en/latest/).
