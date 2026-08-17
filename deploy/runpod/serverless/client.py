"""Submit one audio file to RunPod asynchronously and print raw hypotheses."""

from __future__ import annotations

import argparse
import base64
import os
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--beams", type=int, default=5)
    args = parser.parse_args()
    endpoint = os.environ.get("RUNPOD_ENDPOINT_ID")
    key = os.environ.get("RUNPOD_API_KEY")
    if not endpoint or not key:
        raise SystemExit("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY")
    if not args.audio.is_file():
        raise SystemExit(f"Audio file does not exist: {args.audio}")
    base = f"https://api.runpod.ai/v2/{endpoint}"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "input": {
            "audio_base64": base64.b64encode(args.audio.read_bytes()).decode("ascii"),
            "filename": args.audio.name,
            "beams": args.beams,
        }
    }
    with httpx.Client(timeout=30) as client:
        response = client.post(f"{base}/run", headers=headers, json=payload)
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"submitted {job_id}")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            status = client.get(f"{base}/status/{job_id}", headers=headers)
            status.raise_for_status()
            body = status.json()
            state = body.get("status")
            if state == "COMPLETED":
                output = body["output"]
                print([item["literal_text"] for item in output["hypotheses"]])
                print(output)
                return
            if state in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                raise RuntimeError(body)
            print(state)
            time.sleep(2)
    raise TimeoutError("RunPod job did not finish within five minutes")


if __name__ == "__main__":
    main()
