#!/bin/bash
# Serve either lab checkpoint for manual experiments. Control-C stops it.
set -euo pipefail
LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
BITS="${1:-4}"
case "$BITS" in 4|8) ;; *) echo "Usage: bash serve_local.sh [4|8]" >&2; exit 2 ;; esac
export HF_HOME="$LAB_DIR/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$LAB_DIR/cache" VLLM_CACHE_ROOT="$LAB_DIR/cache/vllm" TORCH_HOME="$LAB_DIR/cache/torch"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_METAL_BACKEND=mlx VLLM_METAL_KV_COMMIT_PROBE=1
"$LAB_DIR/.venv/bin/python" -c 'import psutil; assert psutil.virtual_memory().available >= 4*1024**3, "Need at least 4 GiB available memory"'
exec /usr/bin/caffeinate -i "$LAB_DIR/.venv/bin/python" -m vllm.entrypoints.cli.main serve \
  "$LAB_DIR/models/Qwen3-0.6B-${BITS}bit" \
  --tokenizer "$LAB_DIR/models/Qwen3-0.6B-4bit" --served-model-name bench \
  --host 127.0.0.1 --port 8011 --dtype float16 \
  --max-model-len 4608 --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.3 --kv-cache-memory-bytes 1073741824 \
  --no-enable-prefix-caching --generation-config vllm
