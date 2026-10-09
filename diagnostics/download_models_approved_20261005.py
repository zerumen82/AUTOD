"""Download ONLY the two model comparisons explicitly approved on 2026-10-05.

This diagnostic CLI is never imported by production. Existing files are checked
and preserved; new weights are published only after official SHA256 validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import requests


def download(root, directory, name, url, expected_size, expected_sha=None):
    directory = directory.resolve()
    directory.relative_to((root / "models").resolve())
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    if destination.exists():
        data = destination.read_bytes()
        assert len(data) == expected_size
        if expected_sha:
            assert hashlib.sha256(data).hexdigest() == expected_sha, "Preserve existing mismatched file"
        else:
            json.loads(data)
        print(f"Verified existing: {destination}", flush=True)
        return
    temporary = directory / f"{name}.{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    downloaded = 0
    last_report = time.monotonic()
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("xb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                assert downloaded <= expected_size, "Official payload size exceeded"
                if time.monotonic() - last_report >= 15:
                    print(f"{name}: {downloaded}/{expected_size} bytes", flush=True)
                    last_report = time.monotonic()
    assert downloaded == expected_size, (name, downloaded, expected_size)
    sha = digest.hexdigest()
    if expected_sha:
        assert sha == expected_sha, "Official checksum mismatch; do not publish"
    else:
        json.loads(temporary.read_text(encoding="utf-8"))
    assert not destination.exists(), "Do not overwrite another download"
    # Both resolved paths are within the explicit model directory verified above.
    os.replace(temporary, destination)
    print(f"Verified new: {destination} bytes={downloaded} sha256={sha}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("sam2", "hyperswap1a"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.target == "sam2":
        base = "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/de431c4043854a71d8101e17995dfe596bf101a5/"
        directory = root / "models/sam2/sam2.1-hiera-tiny"
        for name, size, sha in (
            ("config.json", 5695, None), ("preprocessor_config.json", 683, None),
            ("processor_config.json", 95, None),
            ("model.safetensors", 155908064, "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7"),
        ):
            download(root, directory, name, base + name, size, sha)
    else:
        download(root, root / "models", "hyperswap_1a_256.onnx",
                 "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx",
                 402742682, "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add")


if __name__ == "__main__":
    main()
