#!/usr/bin/env python3
"""Record a short microphone sample and submit it to the local Echora API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_SAMPLE_RATE = 16_000


def parse_arguments() -> argparse.Namespace:
    """Read CLI options without requiring audio packages for help output."""

    parser = argparse.ArgumentParser(
        description="Record audio from the default microphone and transcribe it with Echora."
    )
    parser.add_argument("--seconds", type=float, default=8, help="Recording duration (default: 8).")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("ECHORA_API_URL", DEFAULT_API_URL),
        help=f"Local API base URL (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument("--provider", default="runpod", help="ASR provider name (default: runpod).")
    parser.add_argument("--n-best", type=int, default=5, help="Raw alternatives to request (1–5).")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input devices and exit.",
    )
    return parser.parse_args()


def load_audio_dependencies() -> tuple[Any, Any]:
    """Load optional microphone libraries with a direct installation hint."""

    try:
        import sounddevice
        import soundfile
    except ImportError as error:
        message = (
            "Microphone support is not installed. Run: uv sync --extra recorder --dev\n"
            "On macOS, if PortAudio is missing, run: brew install portaudio"
        )
        raise RuntimeError(message) from error
    return sounddevice, soundfile


def upload_recording(
    recording_path: Path,
    *,
    api_url: str,
    provider: str,
    n_best: int,
) -> dict[str, Any]:
    """Send one WAV recording using the same multipart API used by product clients."""

    with recording_path.open("rb") as recording:
        response = httpx.post(
            f"{api_url.rstrip('/')}/v1/transcriptions",
            data={"provider": provider, "language": "en", "n_best": str(n_best)},
            files={"file": ("recording.wav", recording, "audio/wav")},
            timeout=660,
        )
    response.raise_for_status()
    return response.json()


def main() -> int:
    """Record, upload, display the normalized response, and clean up local temp audio."""

    arguments = parse_arguments()
    if arguments.seconds <= 0:
        print("--seconds must be greater than zero.", file=sys.stderr)
        return 2
    if not 1 <= arguments.n_best <= 5:
        print("--n-best must be between 1 and 5.", file=sys.stderr)
        return 2

    try:
        sounddevice, soundfile = load_audio_dependencies()
        if arguments.list_devices:
            print(sounddevice.query_devices())
            return 0

        print(f"Recording for {arguments.seconds:g} seconds. Speak after the prompt…")
        samples = sounddevice.rec(
            int(arguments.seconds * DEFAULT_SAMPLE_RATE),
            samplerate=DEFAULT_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sounddevice.wait()

        with tempfile.TemporaryDirectory(prefix="echora-recording-") as temporary_directory:
            recording_path = Path(temporary_directory) / "recording.wav"
            soundfile.write(recording_path, samples, DEFAULT_SAMPLE_RATE, subtype="PCM_16")
            print("Uploading to Echora…")
            result = upload_recording(
                recording_path,
                api_url=arguments.api_url,
                provider=arguments.provider,
                n_best=arguments.n_best,
            )
        print(json.dumps(result, indent=2))
        return 0
    except httpx.HTTPStatusError as error:
        detail = error.response.text.strip()
        print(f"Transcription failed: HTTP {error.response.status_code}: {detail}", file=sys.stderr)
        return 1
    except (RuntimeError, httpx.HTTPError) as error:
        print(f"Transcription failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
