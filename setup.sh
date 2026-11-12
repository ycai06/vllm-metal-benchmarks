#!/bin/bash
# Isolated, pinned official release install. Never touches ~/.venv-vllm-metal.
set -euo pipefail
LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
UV_BIN="${UV_BIN:-/opt/homebrew/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12}"
export UV_CACHE_DIR="$LAB_DIR/cache/uv"
export HF_HOME="$LAB_DIR/cache/huggingface"
export PIP_CACHE_DIR="$LAB_DIR/cache/pip"
export PYTHONDONTWRITEBYTECODE=1
export UV_SYSTEM_CERTS=true
mkdir -p "$LAB_DIR/downloads" "$LAB_DIR/provenance"
"$PYTHON_BIN" "$LAB_DIR/download_release.py"
if [[ ! -x "$LAB_DIR/.venv/bin/python" ]]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$LAB_DIR/.venv"
fi
"$UV_BIN" pip install --python "$LAB_DIR/.venv/bin/python" "$LAB_DIR/downloads/vllm-0.33.0+cpu-cp38-abi3-macosx_11_0_arm64.whl"
"$UV_BIN" pip install --python "$LAB_DIR/.venv/bin/python" "$LAB_DIR/downloads/vllm_metal-0.33.0-cp312-cp312-macosx_15_0_arm64.whl"
"$UV_BIN" pip install --python "$LAB_DIR/.venv/bin/python" -r "$LAB_DIR/.venv/lib/python3.12/site-packages/vllm_metal/requirements-torch.txt"
"$UV_BIN" pip freeze --python "$LAB_DIR/.venv/bin/python" > "$LAB_DIR/provenance/requirements-freeze.txt"
if ! "$UV_BIN" pip check --python "$LAB_DIR/.venv/bin/python" > "$LAB_DIR/provenance/pip-check.txt" 2>&1; then
  # The official installer overrides core's torch pin with this nightly.
  "$PYTHON_BIN" -c 'from pathlib import Path; import sys; t=Path(sys.argv[1]).read_text(); assert "Found 1 incompatibility" in t and "requires `torch==2.14.0" in t and "`2.16.0.dev20261013` is installed" in t, t; print("Recorded official PyTorch override metadata conflict; runtime verification is required.")' "$LAB_DIR/provenance/pip-check.txt"
fi
