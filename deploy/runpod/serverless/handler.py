"""RunPod queue handler for Echora's literal command-v3 ASR model."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import runpod


HF_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B-hf")


def resolve_cached_model(model_name: str) -> Path:
    organization, name = model_name.split("/", 1)
    root = HF_CACHE_ROOT / f"models--{organization}--{name}"
    reference = root / "refs" / "main"
    snapshots = root / "snapshots"
    if reference.is_file():
        candidate = snapshots / reference.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate
    candidates = sorted(path for path in snapshots.glob("*") if path.is_dir()) if snapshots.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"Could not uniquely resolve cached model {model_name} under {root}")


os.environ["ECHORA_FOUNDATION_PATH"] = str(resolve_cached_model(MODEL_NAME))

from app.asr.engine import QwenCommandEngine  # noqa: E402
from app.audio import decode_audio  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.schemas import RemoteTranscriptionRequest  # noqa: E402


SETTINGS = load_settings()
ENGINE = QwenCommandEngine(
    SETTINGS.foundation,
    SETTINGS.adapter,
    SETTINGS.inference_config,
    SETTINGS.device,
)


def handler(job: dict) -> dict:
    request = RemoteTranscriptionRequest.model_validate(job.get("input", {}))
    payload = base64.b64decode(request.audio_base64, validate=True)
    if len(payload) > SETTINGS.max_upload_bytes:
        raise ValueError("Audio upload exceeds the 20 MB limit")
    waveform = decode_audio(payload, request.filename, SETTINGS.max_audio_seconds)
    return ENGINE.transcribe(waveform, request.beams, "runpod").model_dump(mode="json")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
