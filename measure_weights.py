#!/usr/bin/env python3
"""Measure one local checkpoint's evaluated MLX weights in a fresh process.

Run after stopping all inference servers, once per precision. This supplemental
measurement excludes vLLM server overhead, KV cache, and inference activations;
it is neither process RSS nor a measurement of actual vLLM throughput.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON output file; existing files are preserved")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    model_path = ROOT / "models" / f"Qwen3-0.6B-{args.bits}bit"
    config = json.loads((model_path / "config.json").read_text())
    quantization = config.get("quantization") or config.get("quantization_config")
    if not quantization or quantization.get("bits") != args.bits:
        raise ValueError(f"Checkpoint quantization does not match --bits {args.bits}")
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No local safetensors files in {model_path}")

    # Set local, offline caches before importing either MLX-LM or Transformers.
    # No model casts, model rewrites, or machine-wide settings are performed.
    os.environ.update({
        "HF_HOME": str(ROOT / "cache/huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(ROOT / "cache"),
        "VLLM_CACHE_ROOT": str(ROOT / "cache/vllm"),
        "TORCH_HOME": str(ROOT / "cache/torch"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    sys.dont_write_bytecode = True
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load

    if not mx.is_available(mx.gpu):
        raise RuntimeError("Metal GPU is unavailable; refusing a CPU-only measurement")
    mx.set_default_device(mx.gpu)
    mx.synchronize(mx.gpu)
    baseline = {"active_bytes": mx.get_active_memory(),
                "cached_allocator_bytes": mx.get_cache_memory(),
                "peak_before_reset_bytes": mx.get_peak_memory()}
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, tokenizer = load(str(model_path), lazy=False, trust_remote_code=False)
    parameters = model.parameters()
    mx.eval(parameters)
    mx.synchronize(mx.gpu)
    elapsed = time.perf_counter() - started
    after = {"active_bytes": mx.get_active_memory(),
             "cached_allocator_bytes": mx.get_cache_memory(),
             "peak_since_preload_reset_bytes": mx.get_peak_memory()}

    tensors = tree_flatten(parameters)
    by_dtype = {}
    tensor_inventory = []
    for name, array in tensors:
        dtype = str(array.dtype)
        nbytes = int(array.nbytes)
        elements = int(array.size)
        group = by_dtype.setdefault(dtype, {"tensor_count": 0, "elements": 0, "bytes": 0})
        group["tensor_count"] += 1
        group["elements"] += elements
        group["bytes"] += nbytes
        tensor_inventory.append({"name": name, "dtype": dtype,
                                 "shape": list(array.shape), "elements": elements,
                                 "bytes": nbytes})
    logical_bytes = sum(item["bytes"] for item in tensor_inventory)
    buffer_bytes = mx.get_array_buffer_size(parameters)
    manifest_path = ROOT / "provenance/models.json"
    provenance = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text()).get(str(args.bits), {})
        provenance = {k: manifest.get(k) for k in ("repo", "revision")}
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "Isolated MLX checkpoint weight load; no vLLM server or inference",
        "scope": {
            "included": "Evaluated model parameter arrays, including packed quantized weights, scales, biases and any unquantized parameters",
            "excluded": "vLLM server/KV allocations, inference activations, HTTP serving, and process RSS",
            "precision_control": "Checkpoint dtypes preserved exactly; no cast requested",
            "load_timing": "Local mlx_lm.load(lazy=False), tokenizer construction, explicit parameter evaluation, and GPU synchronization; filesystem cache state is uncontrolled",
            "logical_tensor_bytes": "Sum of array.nbytes across model.parameters(); includes quantization metadata arrays and could count shared storage more than once",
            "unique_parameter_buffer_bytes": "MLX allocator size of unique backing buffers for evaluated model parameters",
            "active_allocation": "MLX allocator active bytes; excludes cached buffers and is not system RSS",
            "peak_allocation": "Peak active allocation since resetting immediately before model load; may include transient loader/model initialization arrays",
        },
        "bits": args.bits,
        "model_path": str(model_path),
        "model_provenance": provenance,
        "quantization": quantization,
        "device": {"default_device": str(mx.default_device()),
                   "info": mx.device_info(mx.gpu)},
        "software": {"python": sys.version, "machine": platform.machine(),
                     "macos": platform.mac_ver()[0],
                     "mlx": version("mlx"), "mlx_lm": version("mlx-lm"),
                     "vllm_metal": version("vllm-metal")},
        "load_and_evaluate_elapsed_s": elapsed,
        "baseline": baseline,
        "after_evaluated_load": after,
        "active_allocation_delta_bytes": after["active_bytes"] - baseline["active_bytes"],
        "tensor_count": len(tensors),
        "logical_tensor_bytes": logical_bytes,
        "unique_parameter_buffer_bytes": buffer_bytes,
        "dtype_distribution": by_dtype,
        "safetensors_file_bytes": sum(path.stat().st_size for path in files),
        "safetensors_files": [{"name": path.name, "bytes": path.stat().st_size}
                              for path in files],
        "tensor_inventory": tensor_inventory,
        "command": [str(Path(sys.executable).absolute()), str(Path(__file__).resolve()),
                    "--bits", str(args.bits), "--output", str(output)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(output), "bits": args.bits,
                      "active_allocation_delta_bytes": result["active_allocation_delta_bytes"],
                      "logical_tensor_bytes": logical_bytes,
                      "unique_parameter_buffer_bytes": buffer_bytes,
                      "peak_since_preload_reset_bytes": after["peak_since_preload_reset_bytes"]}),
          flush=True)
    # The process exits after recording the live-weight measurement, releasing
    # its own model allocations. Launch the other precision as a new process.


if __name__ == "__main__":
    main()
