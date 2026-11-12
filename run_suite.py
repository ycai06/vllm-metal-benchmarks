"""Run controlled local-only experiments; owns/stops only the servers it creates."""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/bin/python"
PORT = 8011

def environment():
    env = os.environ.copy()
    env.update({"HF_HOME": str(ROOT / "cache/huggingface"), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": str(ROOT / "cache"),
                "VLLM_CACHE_ROOT": str(ROOT / "cache/vllm"), "TORCH_HOME": str(ROOT / "cache/torch"),
                "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "1", "VLLM_METAL_BACKEND": "mlx",
                "VLLM_METAL_KV_COMMIT_PROBE": "1", "NO_PROXY": "127.0.0.1,localhost"})
    return env

def state():
    def read_command(command):
        return subprocess.run(command, text=True, capture_output=True).stdout.strip()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {"utc": datetime.now(timezone.utc).isoformat(), "memory_available_bytes": vm.available,
            "memory_total_bytes": vm.total, "swap_used_bytes": swap.used,
            "swap_in_bytes": swap.sin, "swap_out_bytes": swap.sout,
            "power": read_command(["pmset", "-g", "batt"]),
            "thermal": read_command(["pmset", "-g", "therm"]),
            "power_settings": read_command(["pmset", "-g", "custom"])}

def safety_check():
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB available memory; stopping this experiment safely")

@contextlib.contextmanager
def server(bits, kv_mib, session_dir):
    safety_check()
    for attempt in range(15):
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", PORT))  # Fail if another app owns this port.
            break
        except OSError:
            if attempt == 14:
                raise
            time.sleep(1)
    model = ROOT / "models" / f"Qwen3-0.6B-{bits}bit"
    command = ["/usr/bin/caffeinate", "-i", str(PYTHON), "-m", "vllm.entrypoints.cli.main", "serve", str(model),
               "--host", "127.0.0.1", "--port", str(PORT), "--served-model-name", "bench",
               "--tokenizer", str(ROOT / "models/Qwen3-0.6B-4bit"),
               "--dtype", "float16", "--max-model-len", "4608", "--max-num-seqs", "8",
               "--max-num-batched-tokens", "1024", "--gpu-memory-utilization", "0.3",
               "--kv-cache-memory-bytes", str(kv_mib * 1024**2),
               "--no-enable-prefix-caching", "--generation-config", "vllm"]
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "server-command.json").write_text(json.dumps(command, indent=2))
    (session_dir / "state-before.json").write_text(json.dumps(state(), indent=2))
    log = (session_dir / "server.log").open("w")
    proc = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (session_dir / "pid.txt").write_text(str(proc.pid))
    try:
        for _ in range(240):
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited {proc.returncode}; see {session_dir / 'server.log'}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError("Server did not become ready within 240 seconds")
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models") as response:
            (session_dir / "served-models.json").write_bytes(response.read())
        yield proc
    finally:
        # This new process group belongs exclusively to this experiment.
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        log.close()
        (session_dir / "state-after.json").write_text(json.dumps(state(), indent=2))

def benchmark(directory, proc, name, concurrency, input_tokens, requests, repeats=3, output_tokens=64):
    safety_check()
    command = [str(PYTHON), str(ROOT / "benchmark.py"), "--base-url", f"http://127.0.0.1:{PORT}",
               "--model", "bench", "--tokenizer", str(ROOT / "models/Qwen3-0.6B-4bit"),
               "--output", str(directory / name), "--server-pid", str(proc.pid), "--seed", "713",
               "--concurrency", str(concurrency), "--input-tokens", str(input_tokens),
               "--output-tokens", str(output_tokens), "--requests", str(requests),
               "--warmups", "2", "--repeats", str(repeats), "--sample-interval", "0.1"]
    (directory / f"{name}.command.json").write_text(json.dumps(command, indent=2))
    print(f"Running {directory.name}/{name}", flush=True)
    with (directory / f"{name}.client.log").open("w") as log:
        client = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
        try:
            while client.poll() is None:
                safety_check()
                time.sleep(1)
            if client.returncode:
                raise subprocess.CalledProcessError(client.returncode, command)
        finally:
            if client.poll() is None:
                client.terminate()
                try:
                    client.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    client.kill()
                    client.wait()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="New results directory; defaults to timestamp")
    parser.add_argument("--resume", action="store_true", help="Resume successful trials in --output; keep prior files and create new server blocks")
    args = parser.parse_args()
    results = args.output or ROOT / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = results.resolve()
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    if args.resume:
        if not results.is_dir() or (results / "suite-state-after.json").exists():
            parser.error("Resume requires an existing incomplete suite")
        (results / f"resume-state-{time.time_ns()}.json").write_text(json.dumps(state(), indent=2))
    else:
        results.mkdir(parents=True, exist_ok=False)
        (results / "suite-state-before.json").write_text(json.dumps(state(), indent=2))

    def completed(block, trial):
        for directory in results.glob(block + "*"):
            path = directory / (trial + ".summary.json")
            if path.exists():
                summary = json.loads(path.read_text())
                if summary["overall"]["errors"] or not (directory / (trial + ".jsonl")).exists():
                    raise RuntimeError(f"Existing trial needs review: {path}")
                return True
        return False

    def next_directory(block):
        directory = results / block
        attempt = 1
        while directory.exists():
            directory = results / f"{block}-resume-{attempt}"
            attempt += 1
        return directory
    if args.smoke:
        directory = results / "smoke-q4"
        with server(4, 1024, directory) as proc:
            benchmark(directory, proc, "smoke", 1, 256, 1, repeats=1, output_tokens=16)
    else:
        # Quantization order ABBA; independent warmed server per block.
        for index, bits in enumerate((4, 8, 8, 4)):
            block = f"block-{index+1}-q{bits}"
            trials = [("weight-c1-p256", 1, 256, 8)]
            if index == 0:
                trials += [(f"batch-c{c}-p1024", c, 1024, 16) for c in (1, 4, 8)]
                trials += [(f"context-c1-p{length}", 1, length, 4) for length in (256, 1024, 4096)]
            if index == 3:
                trials += [(f"batch-reverse-c{c}-p1024", c, 1024, 16) for c in (8, 4, 1)]
                trials += [(f"context-reverse-c1-p{length}", 1, length, 4) for length in (4096, 1024, 256)]
            trials = [t for t in trials if not completed(block, t[0])]
            if not trials:
                continue
            directory = next_directory(block)
            with server(bits, 1024, directory) as proc:
                for trial in trials:
                    benchmark(directory, proc, *trial)
        for kv_mib in (1024, 2048):
            block = f"kv-q4-{kv_mib}mib"
            if completed(block, "capacity-c4-p4096"):
                continue
            directory = next_directory(block)
            with server(4, kv_mib, directory) as proc:
                benchmark(directory, proc, "capacity-c4-p4096", 4, 4096, 8)
    (results / "suite-state-after.json").write_text(json.dumps(state(), indent=2))
    print(f"Completed: {results}", flush=True)

if __name__ == "__main__":
    main()
