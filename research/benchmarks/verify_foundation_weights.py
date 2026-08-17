"""Verify the exact official foundation archives staged on the SSD."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED = {
    "parakeet": (
        Path(
            "/Volumes/Extreme Pro/echora/checkpoints/base-models/"
            "nvidia-parakeet-tdt-1.1b/parakeet-tdt-1.1b.nemo"
        ),
        4_283_136_000,
        "9c563d52bdffeacbac0c5b894fdea9be82fea3a6bd8bb8018ff57888e2b5d988",
    ),
    "qwen3_asr": (
        Path(
            "/Volumes/Extreme Pro/echora/checkpoints/base-models/"
            "Qwen3-ASR-1.7B-hf/model.safetensors"
        ),
        4_076_193_080,
        "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=sorted(EXPECTED))
    args = parser.parse_args()
    selected = args.models or list(EXPECTED)
    failed = False
    for name in selected:
        path, expected_size, expected_sha = EXPECTED[name]
        if not path.is_file():
            print(f"FAIL {name}: missing {path}")
            failed = True
            continue
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            print(f"FAIL {name}: expected {expected_size} bytes, got {actual_size}")
            failed = True
            continue
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            print(f"FAIL {name}: SHA-256 {actual_sha}, expected {expected_sha}")
            failed = True
            continue
        print(f"OK   {name}: {path} ({actual_sha})")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
