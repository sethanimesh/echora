"""Runtime configuration with explicit, non-silent backend selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    backend: str
    device: str
    model_root: Path
    foundation: Path
    adapter: Path
    inference_config: Path
    beams: int
    max_audio_seconds: float
    max_upload_bytes: int
    groq_api_key: str | None
    groq_models: tuple[str, ...]
    pod_url: str | None
    pod_token: str | None
    runpod_endpoint_id: str | None
    runpod_api_key: str | None
    remote_timeout_seconds: float

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)


def load_settings() -> Settings:
    model_root = Path(
        os.getenv(
            "ECHORA_MODEL_ROOT",
            str(PROJECT_ROOT / "models" / "echora-qwen3-asr-command-v3"),
        )
    ).expanduser()
    backend = os.getenv("ECHORA_ASR_BACKEND", "local").strip().lower()
    if backend not in {"local", "pod", "runpod"}:
        raise ValueError("ECHORA_ASR_BACKEND must be local, pod, or runpod")
    beams = int(os.getenv("ECHORA_BEAMS", "5"))
    if not 2 <= beams <= 8:
        raise ValueError("ECHORA_BEAMS must be between 2 and 8")
    # Groq enforces rate limits per model, so a fallback chain multiplies the
    # available quota rather than just retrying into the same exhausted bucket.
    groq_models = tuple(
        name.strip()
        for name in os.getenv(
            "ECHORA_GROQ_MODELS",
            "openai/gpt-oss-120b,qwen/qwen3.6-27b,openai/gpt-oss-20b",
        ).split(",")
        if name.strip()
    )
    if not groq_models:
        raise ValueError("ECHORA_GROQ_MODELS must name at least one model")
    return Settings(
        project_root=PROJECT_ROOT,
        backend=backend,
        device=os.getenv("ECHORA_DEVICE", "auto").strip().lower(),
        model_root=model_root,
        foundation=Path(
            os.getenv("ECHORA_FOUNDATION_PATH", str(model_root / "foundation"))
        ).expanduser(),
        adapter=Path(
            os.getenv(
                "ECHORA_ADAPTER_PATH", str(model_root / "adapter" / "adapter.safetensors")
            )
        ).expanduser(),
        inference_config=Path(
            os.getenv(
                "ECHORA_INFERENCE_CONFIG",
                str(model_root / "evaluation" / "qwen_command_v3.json"),
            )
        ).expanduser(),
        beams=beams,
        max_audio_seconds=float(os.getenv("ECHORA_MAX_AUDIO_SECONDS", "45")),
        max_upload_bytes=int(os.getenv("ECHORA_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
        groq_api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ") or None,
        groq_models=groq_models,
        pod_url=os.getenv("ECHORA_POD_URL") or None,
        pod_token=os.getenv("ECHORA_POD_TOKEN") or None,
        runpod_endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID") or None,
        runpod_api_key=os.getenv("RUNPOD_API_KEY") or None,
        remote_timeout_seconds=float(os.getenv("ECHORA_REMOTE_TIMEOUT_SECONDS", "240")),
    )
