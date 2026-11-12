"""Fetch release wheels and verify the SHA256 digests in official release metadata."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
RELEASES = [
    ("vllm-project/vllm-metal", "v0.33.0", "vllm_metal-0.33.0-cp312-cp312-macosx_15_0_arm64.whl"),
    ("vllm-project/vllm", "v0.33.0rc3", "vllm-0.33.0+cpu-cp38-abi3-macosx_11_0_arm64.whl"),
]

def fetch(url, target):
    # macOS curl uses the system trust store; do not disable TLS verification.
    subprocess.run(["/usr/bin/curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3", "--max-time", "600", url, "--output", str(target)], check=True)

for repo, tag, filename in RELEASES:
    metadata_path = ROOT / "provenance" / f"{repo.split('/')[-1]}-{tag}-release.json"
    fetch(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", metadata_path)
    metadata = json.loads(metadata_path.read_text())
    asset = next(x for x in metadata["assets"] if x["name"] == filename)
    target = ROOT / "downloads" / filename
    if not target.exists():
        print(f"Downloading {filename} ({asset['size']:,} bytes)", flush=True)
        temporary = target.with_suffix(".partial")
        fetch(asset["browser_download_url"], temporary)
        temporary.rename(target)
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    digest = asset.get("digest")
    if not digest or digest != f"sha256:{checksum}":
        raise RuntimeError(f"Missing or mismatched official digest for {filename}: {digest}, local {checksum}")
    print(f"Verified {filename}: sha256:{checksum}", flush=True)
