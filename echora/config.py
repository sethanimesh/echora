"""Configuration and the composition root.

Everything replaceable is chosen here and nowhere else. No call site anywhere in
the pipeline branches on vendor -- they receive an already-built classifier
through the port. Swapping local for cloud is an env var, not a code change.

    ECHORA_CLASSIFIER=auto        # auto | ollama | groq | offline
    ECHORA_OLLAMA_MODEL=qwen3:4b
    ECHORA_OLLAMA_HOST=http://localhost:11434
    ECHORA_GROQ_MODEL=openai/gpt-oss-120b
    ECHORA_ESCALATE_ON_UNSURE=0   # 1 sends low-confidence spans to the cloud
    ECHORA_TIMEOUT=15.0
    ECHORA_CACHE=1

``auto`` is the shipped default and means: **local first, Groq as fallback,
offline heuristic as the floor.** The chain is total, so classification degrades
but never fails. Fish Audio synthesis is separately enabled by an audio CLI flag
or the voice Haystack pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from echora.classify.port import (
    CachingSpanClassifier, FallbackChain, HeuristicSpanClassifier,
    SentencePriorGuard, SpanClassifier,
)
from echora.tts.fish_audio import DEFAULT_FISH_VOICE_ID

log = logging.getLogger(__name__)

# Module-level defaults. These must NOT be read off the dataclass as
# ``cls.ollama_model``: with slots=True that returns a member_descriptor rather
# than the default value, which silently poisons the config with a garbage model
# name and makes the vendor fail construction.
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT = 15.0
DEFAULT_FISH_TIMEOUT = 30.0


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader. Avoids a python-dotenv dependency for six lines.

    Existing environment variables always win, so an explicit export overrides
    the file rather than the other way round.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(slots=True)
class Config:
    """Library settings. Environment files are loaded only when explicitly supplied.

    The communication app owns separate provider policy and configuration.
    """

    classifier: str = "auto"
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    groq_model: str = DEFAULT_GROQ_MODEL
    escalate_on_unsure: bool = False
    timeout: float = DEFAULT_TIMEOUT
    cache: bool = True
    protected_terms: set[str] = field(default_factory=set)
    fish_api_key: str | None = None
    fish_voice_id: str = DEFAULT_FISH_VOICE_ID
    fish_timeout: float = DEFAULT_FISH_TIMEOUT

    @classmethod
    def from_env(cls, dotenv: Path | None = None) -> Config:
        if dotenv is not None:
            _load_dotenv(dotenv)
        terms = os.environ.get("ECHORA_PROTECTED_TERMS", "")
        return cls(
            classifier=os.environ.get("ECHORA_CLASSIFIER", "auto").lower(),
            ollama_model=os.environ.get("ECHORA_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            ollama_host=os.environ.get("ECHORA_OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            groq_model=os.environ.get("ECHORA_GROQ_MODEL", DEFAULT_GROQ_MODEL),
            escalate_on_unsure=_env_bool("ECHORA_ESCALATE_ON_UNSURE", False),
            timeout=float(os.environ.get("ECHORA_TIMEOUT", str(DEFAULT_TIMEOUT))),
            cache=_env_bool("ECHORA_CACHE", True),
            protected_terms={t.strip() for t in terms.split(",") if t.strip()},
            fish_api_key=(
                os.environ.get("FISH_API_KEY")
                or os.environ.get("FISH_AUDIO_API_KEY")
                or None
            ),
            fish_voice_id=(
                os.environ.get("ECHORA_FISH_VOICE_ID") or DEFAULT_FISH_VOICE_ID
            ),
            fish_timeout=float(
                os.environ.get("ECHORA_FISH_TIMEOUT", str(DEFAULT_FISH_TIMEOUT))
            ),
        )

    def build_synthesizer(self):
        """Construct the optional Fish client without importing its SDK eagerly."""
        from echora.tts.fish_audio import FishAudioSynthesizer

        return FishAudioSynthesizer(
            api_key=self.fish_api_key,
            voice_id=self.fish_voice_id,
            timeout=self.fish_timeout,
        )

    def build_classifier(self) -> SpanClassifier:
        """Assemble the classifier chain described by this config.

        In ``auto`` mode a vendor that cannot be constructed (Ollama not
        installed, no Groq key) is logged and skipped: a missing optional vendor
        should degrade the chain, not stop the device from starting.

        But when a vendor is named **explicitly**, construction failure raises.
        Silently substituting a different classifier for the one that was asked
        for produces confidently wrong output with no indication anything went
        wrong -- which is exactly how a garbage model name went unnoticed here.
        """
        explicit = self.classifier != "auto"
        chain: list[SpanClassifier] = []

        if self.classifier in ("auto", "ollama"):
            try:
                from echora.classify.ollama_classifier import OllamaSpanClassifier
                chain.append(OllamaSpanClassifier(
                    model=self.ollama_model,
                    host=self.ollama_host,
                    timeout=self.timeout,
                ))
            except Exception as exc:  # noqa: BLE001
                if explicit:
                    raise RuntimeError(
                        f"Ollama classifier requested but unavailable: {exc}"
                    ) from exc
                log.info("Ollama classifier unavailable: %s", exc)

        if self.classifier in ("auto", "groq"):
            try:
                from echora.classify.groq_classifier import GroqSpanClassifier
                chain.append(GroqSpanClassifier(
                    model=self.groq_model, timeout=self.timeout,
                ))
            except Exception as exc:  # noqa: BLE001
                if explicit:
                    raise RuntimeError(
                        f"Groq classifier requested but unavailable: {exc}"
                    ) from exc
                log.info("Groq classifier unavailable: %s", exc)

        # Always last, always present: the chain must never be empty and must
        # never raise. This is the guarantee that the device still speaks.
        chain.append(HeuristicSpanClassifier())

        classifier: SpanClassifier = FallbackChain(
            chain, escalate_on_unsure=self.escalate_on_unsure
        )
        # Wraps every backend, including the offline one: the over-prediction of
        # HI on English sentences is a property of the task, not of one vendor.
        classifier = SentencePriorGuard(classifier)
        if self.cache:
            classifier = CachingSpanClassifier(classifier)
        return classifier
