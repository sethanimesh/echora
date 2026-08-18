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
    groq_tts_model: str
    groq_tts_voice: str
    speech_autoplay: bool
    pod_url: str | None
    pod_token: str | None
    runpod_endpoint_id: str | None
    runpod_api_key: str | None
    remote_timeout_seconds: float
    personal_enabled: bool
    personal_specializations: bool
    persona_root: Path
    personal_root: Path
    places_path: Path
    embedder_root: Path
    embedder_device: str
    embedder_max_tokens: int
    personal_examples: int
    personal_min_similarity: float
    personal_half_life_days: float
    personal_context_penalty: float
    personal_listener_penalty: float
    personal_time_penalty: float
    personal_max_hints: int
    personal_merge_threshold: float
    personal_store_cap: int
    personal_anchor_share: float

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _ranged(name: str, default: str, low: float, high: float, cast=float):
    """Read a tuning value and refuse it at startup rather than mid-utterance.

    A bad threshold is a configuration mistake, not a runtime degradation: it
    should stop the process now, while only *runtime* personalization failures
    are allowed to fall back silently.
    """
    value = cast(os.getenv(name, default))
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


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
        groq_tts_model=os.getenv("ECHORA_TTS_MODEL", "canopylabs/orpheus-v1-english").strip(),
        groq_tts_voice=os.getenv("ECHORA_TTS_VOICE", "hannah").strip(),
        speech_autoplay=os.getenv("ECHORA_SPEECH_AUTOPLAY", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        pod_url=os.getenv("ECHORA_POD_URL") or None,
        pod_token=os.getenv("ECHORA_POD_TOKEN") or None,
        runpod_endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID") or None,
        runpod_api_key=os.getenv("RUNPOD_API_KEY") or None,
        remote_timeout_seconds=float(os.getenv("ECHORA_REMOTE_TIMEOUT_SECONDS", "240")),
        # Two independent switches: the disambiguation prior is grounding-safe and
        # can stay on while the detail layer, which adds words, is turned off.
        personal_enabled=_flag("ECHORA_PERSONAL_ENABLED", "true"),
        personal_specializations=_flag("ECHORA_PERSONAL_SPECIALIZATION", "true"),
        persona_root=Path(
            os.getenv("ECHORA_PERSONA_ROOT", str(PROJECT_ROOT / "data" / "personas"))
        ).expanduser(),
        personal_root=Path(
            os.getenv("ECHORA_PERSONAL_ROOT", str(PROJECT_ROOT / "data" / "personal"))
        ).expanduser(),
        # Places sit beside the live personal store because that directory is
        # already gitignored and already the one thing a reset deletes. They are
        # deliberately not inside a profile: location has to work when personal
        # context is switched off or no profile is chosen.
        places_path=Path(
            os.getenv(
                "ECHORA_PLACES_PATH",
                str(
                    Path(
                        os.getenv("ECHORA_PERSONAL_ROOT", str(PROJECT_ROOT / "data" / "personal"))
                    )
                    / "settings.json"
                ),
            )
        ).expanduser(),
        embedder_root=Path(
            os.getenv("ECHORA_EMBEDDER_ROOT", str(PROJECT_ROOT / "models" / "echora-minilm-l6-v2"))
        ).expanduser(),
        # CPU on purpose: the 1.7B recognizer owns MPS, and a 22M encoder is a few
        # milliseconds on CPU, so queueing behind it would only add latency.
        embedder_device=os.getenv("ECHORA_EMBEDDER_DEVICE", "cpu").strip().lower(),
        embedder_max_tokens=int(_ranged("ECHORA_EMBEDDER_MAX_TOKENS", "128", 16, 512, int)),
        personal_examples=int(_ranged("ECHORA_PERSONAL_EXAMPLES", "4", 1, 8, int)),
        personal_min_similarity=_ranged("ECHORA_PERSONAL_MIN_SIMILARITY", "0.25", 0.0, 1.0),
        personal_half_life_days=_ranged("ECHORA_PERSONAL_HALF_LIFE_DAYS", "14", 0.5, 3650.0),
        personal_context_penalty=_ranged("ECHORA_PERSONAL_CONTEXT_PENALTY", "0.85", 0.1, 1.0),
        # Harsher than the setting penalty on purpose. A differently-set example
        # still shows the right shape; a differently-addressed one shows the
        # wrong one, and shape is what a few-shot example teaches.
        personal_listener_penalty=_ranged("ECHORA_PERSONAL_LISTENER_PENALTY", "0.5", 0.1, 1.0),
        personal_time_penalty=_ranged("ECHORA_PERSONAL_TIME_PENALTY", "0.95", 0.1, 1.0),
        personal_max_hints=int(_ranged("ECHORA_PERSONAL_MAX_HINTS", "8", 1, 32, int)),
        personal_merge_threshold=_ranged("ECHORA_PERSONAL_MERGE_THRESHOLD", "0.92", 0.5, 1.0),
        personal_store_cap=int(_ranged("ECHORA_PERSONAL_STORE_CAP", "400", 20, 20000, int)),
        # A detail may only ride on a word the beams overwhelmingly agreed on.
        personal_anchor_share=_ranged("ECHORA_PERSONAL_ANCHOR_SHARE", "0.75", 0.5, 1.0),
    )
