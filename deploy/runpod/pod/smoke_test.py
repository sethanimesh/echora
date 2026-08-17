"""Submit the three preserved clips to a dedicated Echora Pod worker."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[3]
URL = os.environ.get("ECHORA_WORKER_URL", "http://127.0.0.1:8001").rstrip("/")
TOKEN = os.environ.get("ECHORA_WORKER_TOKEN")
EXPECTED_FOUNDATION_SHA256 = "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1"
EXPECTED_ADAPTER_SHA256 = "7cd203cc0cbf479e6afa198cc3895fedc71f1907b870fcb7f1dece0a1e6b2021"


def main() -> None:
    if not TOKEN:
        raise SystemExit("Set ECHORA_WORKER_TOKEN")
    clips = sorted((ROOT / "research" / "benchmarks" / "clips").glob("*.wav"))
    if len(clips) != 3:
        raise SystemExit(f"Expected three clips, found {len(clips)}")
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with httpx.Client(timeout=180) as client:
        health = client.get(f"{URL}/health")
        health.raise_for_status()
        identity = health.json()
        assert identity["foundation_sha256"] == EXPECTED_FOUNDATION_SHA256
        assert identity["adapter_sha256"] == EXPECTED_ADAPTER_SHA256
        print(identity)
        for clip in clips:
            response = client.post(
                f"{URL}/v1/transcribe",
                headers=headers,
                json={
                    "audio_base64": base64.b64encode(clip.read_bytes()).decode("ascii"),
                    "filename": clip.name,
                    "beams": 5,
                },
            )
            response.raise_for_status()
            body = response.json()
            texts = [item["literal_text"] for item in body["hypotheses"]]
            if not texts:
                raise RuntimeError(f"No hypotheses for {clip.name}")
            print(f"{clip.name}: {texts}")
    print("ECHORA_POD_SMOKE_PASSED")


if __name__ == "__main__":
    main()
