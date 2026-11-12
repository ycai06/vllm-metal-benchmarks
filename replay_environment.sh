#!/bin/bash
# Recreate the exact measured package versions, including the official Torch override.
set -euo pipefail
LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
UV_BIN="${UV_BIN:-/opt/homebrew/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12}"
export UV_CACHE_DIR="$LAB_DIR/cache/uv"
export UV_SYSTEM_CERTS=true
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$LAB_DIR/downloads" "$LAB_DIR/provenance"
"$PYTHON_BIN" "$LAB_DIR/download_release.py"
if [[ ! -x "$LAB_DIR/.venv/bin/python" ]]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$LAB_DIR/.venv"
fi
# The lock contains all installed packages; --no-deps preserves the official
# nightly Torch override despite core's older metadata requirement.
"$UV_BIN" pip install --python "$LAB_DIR/.venv/bin/python" --no-deps -r "$LAB_DIR/provenance/requirements-replay.txt"
"$LAB_DIR/.venv/bin/python" "$LAB_DIR/prepare_models.py"
