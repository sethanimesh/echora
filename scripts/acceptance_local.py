"""Exercise the three preserved personal clips through the running local API."""

from __future__ import annotations

import os
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
API = os.getenv("ECHORA_API_URL", "http://127.0.0.1:8000").rstrip("/")
CONTEXT = os.getenv("ECHORA_ACCEPTANCE_CONTEXT", "home")
EXPECTED_TOP = {
    "20260815-135933.wav": "I want water.",
    "20260815-140043.wav": "I want water please.",
}
KNOWN_FAILURE = "20260815-135957.wav"


def main() -> None:
    clips = sorted((ROOT / "research" / "benchmarks" / "clips").glob("*.wav"))
    if len(clips) != 3:
        raise SystemExit(f"Expected three acceptance clips, found {len(clips)}")
    with httpx.Client(timeout=180) as client:
        health = client.get(f"{API}/api/v1/health")
        health.raise_for_status()
        if not health.json()["model_ready"]:
            raise RuntimeError(health.json()["detail"])
        for clip in clips:
            with clip.open("rb") as handle:
                response = client.post(
                    f"{API}/api/v1/transcriptions",
                    files={"audio": (clip.name, handle, "audio/wav")},
                    data={"context": CONTEXT},
                )
            response.raise_for_status()
            result = response.json()
            literals = [item["literal_text"] for item in result["hypotheses"]]
            assert literals, f"No hypotheses for {clip.name}"
            assert result["beam_weights_are_calibrated_confidence"] is False
            assert result["user_confirmation_required"] is True
            if clip.name in EXPECTED_TOP:
                assert literals[0] == EXPECTED_TOP[clip.name], (clip.name, literals)
                assert result["needs_user_choice"] is False, (clip.name, result["ranker"])
            if clip.name == KNOWN_FAILURE:
                assert result["needs_user_choice"] is True
                assert len(result["messages"]) > 1
            messages = [item["corrected_text"] for item in result["messages"]]
            print(f"{clip.name}: literals={literals} messages={messages}")
    print("ECHORA_LOCAL_ACCEPTANCE_PASSED")


if __name__ == "__main__":
    main()
