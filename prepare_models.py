"""Download a pinned, matched MLX Qwen3-0.6B pair into this lab only."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "cache" / "huggingface")
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
from huggingface_hub import HfApi, snapshot_download

manifest_path = ROOT / "provenance" / "models.json"
previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
manifest = dict(previous)  # Preserve both revision pins if a resumed download is interrupted.
api = HfApi(token=False)
for bits in (4, 8):
    repo = f"mlx-community/Qwen3-0.6B-{bits}bit"
    revision = previous.get(str(bits), {}).get("revision") or api.model_info(repo).sha
    destination = ROOT / "models" / f"Qwen3-0.6B-{bits}bit"
    snapshot_download(repo, revision=revision, token=False, local_dir=destination,
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "README.md", "*.model"], max_workers=3)
    files = {}
    for path in sorted(destination.iterdir()):
        if path.is_file():
            files[path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
    config = json.loads((destination / "config.json").read_text())
    assert config["quantization"]["bits"] == bits
    assert config["quantization"]["group_size"] == 64
    manifest[str(bits)] = {"repo": repo, "revision": revision, "path": str(destination), "config": config, "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Verified {repo}@{revision}", flush=True)

for name in ("tokenizer.json",):
    assert manifest["4"]["files"][name]["sha256"] == manifest["8"]["files"][name]["sha256"], f"Tokenizer mismatch: {name}"
tokenizer_configs = [json.loads((ROOT / "models" / f"Qwen3-0.6B-{bits}bit" / "tokenizer_config.json").read_text()) for bits in (4, 8)]
assert {k: v for k, v in tokenizer_configs[0].items() if k != "chat_template"} == {k: v for k, v in tokenizer_configs[1].items() if k != "chat_template"}
manifest["tokenizer_control"] = "tokenizer.json hashes match; tokenizer_config.json differs only in chat_template. Both servers explicitly use the 4-bit directory tokenizer; benchmarks send exact token IDs to completions without a chat template."
def without_quantization(config):
    return {k: v for k, v in config.items() if k not in ("quantization", "quantization_config")}
assert without_quantization(manifest["4"]["config"]) == without_quantization(manifest["8"]["config"]), "Architecture configs differ"
manifest_path.write_text(json.dumps(manifest, indent=2))
print("Architecture and tokenizer match; quantization is affine, group size 64.")
