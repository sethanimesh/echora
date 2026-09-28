"""Fish Audio S2.1-Pro Free speech synthesis.

The linguistic pipeline deliberately ends at a normalised speech string.  This
module is the next boundary: it adds trusted delivery instructions and turns
that string into a progressive audio/alignment stream.

Fish's official SDK does not currently expose the timestamped SSE endpoint, so
TTS uses that HTTP endpoint directly while persistent voice enrollment uses the
official SDK.  Both dependencies are imported lazily; the rest of Echora stays
usable without the optional ``fish`` extra.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


FISH_MODEL = "s2.1-pro-free"
FISH_BASE_URL = "https://api.fish.audio"
DEFAULT_FISH_VOICE_ID = "7e711cdecd1f4ba0b720c164148e29f6"
SUPPORTED_FORMATS = ("opus", "mp3", "wav", "pcm")
SUPPORTED_LATENCIES = ("low", "balanced", "normal")


class SpeechStyle(str, Enum):
    """Echora's user-facing assistive delivery presets."""

    SIMPLE = "simple"
    NEUTRAL = "neutral"
    WARM = "warm"
    CONFIDENT = "confident"
    URGENT = "urgent"
    SOFT_PRIVATE = "soft-private"
    SLOW_CLEAR = "slow-clear"
    HAPPY = "happy"
    EMPATHETIC = "empathetic"
    EXCITED = "excited"
    FORMAL = "formal"


STYLE_DIRECTIONS: dict[SpeechStyle, str | None] = {
    SpeechStyle.SIMPLE: "calm, soothing, natural conversational delivery",
    SpeechStyle.NEUTRAL: None,
    SpeechStyle.WARM: "warm, friendly tone",
    SpeechStyle.CONFIDENT: "confident, clear tone",
    SpeechStyle.URGENT: "urgent, firm tone",
    SpeechStyle.SOFT_PRIVATE: "soft, low-volume private tone",
    SpeechStyle.SLOW_CLEAR: "slow, carefully articulated speech",
    SpeechStyle.HAPPY: "happy, bright tone",
    SpeechStyle.EMPATHETIC: "gentle, empathetic tone",
    SpeechStyle.EXCITED: "excited, energetic tone",
    SpeechStyle.FORMAL: "professional, formal tone",
}


class SpeechError(RuntimeError):
    """Stable, vendor-neutral synthesis or enrollment failure."""

    def __init__(
        self, message: str, *, kind: str = "provider", status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class SpeechOptions:
    """Validated options for one Fish Audio synthesis request."""

    voice_id: str | None = None
    style: SpeechStyle | None = None
    direction: str | None = None
    speed: float = 1.0
    volume: float = 0.0
    latency: str = "balanced"
    format: str = "opus"

    def __post_init__(self) -> None:
        if self.style is None and self.direction is None:
            object.__setattr__(self, "style", SpeechStyle.SIMPLE)
        if self.style is not None and not isinstance(self.style, SpeechStyle):
            try:
                object.__setattr__(self, "style", SpeechStyle(self.style))
            except ValueError as exc:
                raise ValueError(f"unknown speech style: {self.style}") from exc
        if self.style is not None and self.direction is not None:
            raise ValueError("style and direction are mutually exclusive")
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        if not -20.0 <= self.volume <= 20.0:
            raise ValueError("volume must be between -20 and 20")
        if self.latency not in SUPPORTED_LATENCIES:
            raise ValueError(
                f"latency must be one of: {', '.join(SUPPORTED_LATENCIES)}"
            )
        if self.format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"format must be one of: {', '.join(SUPPORTED_FORMATS)}"
            )
        if self.voice_id is not None and not self.voice_id.strip():
            raise ValueError("voice_id cannot be empty")
        if self.direction is not None:
            direction = self.direction.strip()
            if not direction:
                raise ValueError("direction cannot be empty")
            if "[" in direction or "]" in direction:
                raise ValueError("direction must not contain square brackets")
            if "\n" in direction or "\r" in direction:
                raise ValueError("direction must be one line")

    @property
    def delivery_direction(self) -> str | None:
        if self.direction is not None:
            return self.direction.strip()
        if self.style is None:
            return None
        return STYLE_DIRECTIONS[self.style]

    def compose(self, speech: str) -> str:
        """Add one trusted S2 direction after rejecting content instructions."""
        if "[" in speech or "]" in speech:
            raise ValueError(
                "message contains square brackets; Fish synthesis rejects "
                "untrusted bracket directions"
            )
        direction = self.delivery_direction
        return speech if direction is None else f"[{direction}] {speech}"


@dataclass(frozen=True, slots=True)
class AlignmentSegment:
    """One word/phrase aligned to the global audio timeline."""

    text: str
    start: float
    end: float
    chunk_seq: int


@dataclass(frozen=True, slots=True)
class SpeechEvent:
    """A progressive audio chunk and optional cumulative alignment snapshot."""

    audio: bytes
    content: str
    chunk_seq: int
    alignment: tuple[AlignmentSegment, ...] | None


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Collected audio and the final replacement alignment for every chunk."""

    audio: bytes
    alignment: tuple[AlignmentSegment, ...]
    format: str
    directed_text: str


@dataclass(frozen=True, slots=True)
class VoiceEnrollment:
    """Identity and state returned after persistent voice creation."""

    id: str
    title: str
    state: str


def collect_speech_events(
    events: Iterable[SpeechEvent], *, format: str, directed_text: str
) -> SynthesisResult:
    """Collect a stream using Fish's replace-by-``chunk_seq`` semantics."""
    audio: list[bytes] = []
    snapshots: dict[int, tuple[AlignmentSegment, ...]] = {}
    for event in events:
        audio.append(event.audio)
        if event.alignment is not None:
            snapshots[event.chunk_seq] = event.alignment
    timeline = tuple(
        segment
        for chunk_seq in sorted(snapshots)
        for segment in snapshots[chunk_seq]
    )
    combined = b"".join(audio)
    if not combined:
        raise SpeechError("Fish Audio returned no audio", kind="protocol")
    return SynthesisResult(
        audio=combined,
        alignment=timeline,
        format=format,
        directed_text=directed_text,
    )


class FishAudioSynthesizer:
    """Timestamped TTS and persistent voice enrollment client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_id: str | None = None,
        timeout: float = 30.0,
        base_url: str = FISH_BASE_URL,
        session: Any | None = None,
        voice_client: Any | None = None,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("FISH_API_KEY")
            or os.environ.get("FISH_AUDIO_API_KEY")
        )
        self._voice_id = voice_id or DEFAULT_FISH_VOICE_ID
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._session = session
        self._voice_client = voice_client

    def _require_key(self) -> str:
        if not self._api_key:
            raise SpeechError(
                "FISH_API_KEY is required for Fish Audio synthesis",
                kind="authentication",
            )
        return self._api_key

    def _http_session(self):
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as exc:
            raise SpeechError(
                "Fish Audio support is not installed; install echora[fish]",
                kind="dependency",
            ) from exc
        self._session = requests.Session()
        return self._session

    @staticmethod
    def _provider_message(response: Any) -> str:
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001 - provider error bodies are not stable
            text = getattr(response, "text", "")
            return str(text).strip() or "Fish Audio request failed"
        if isinstance(payload, dict):
            return str(
                payload.get("message")
                or payload.get("reason")
                or payload.get("detail")
                or "Fish Audio request failed"
            )
        return "Fish Audio request failed"

    @classmethod
    def _raise_for_status(cls, response: Any) -> None:
        status = int(getattr(response, "status_code", 0))
        if status < 400:
            return
        kind = {
            401: "authentication",
            402: "quota",
            422: "validation",
            429: "rate_limit",
        }.get(status, "provider")
        raise SpeechError(
            cls._provider_message(response), kind=kind, status_code=status
        )

    def stream(
        self, speech: str, options: SpeechOptions | None = None
    ) -> Iterator[SpeechEvent]:
        """Yield progressive audio and timestamp snapshots from Fish SSE."""
        resolved = options or SpeechOptions()
        directed_text = resolved.compose(speech)
        voice_id = resolved.voice_id or self._voice_id
        payload: dict[str, Any] = {
            "text": directed_text,
            "prosody": {
                "speed": resolved.speed,
                "volume": resolved.volume,
                "normalize_loudness": True,
            },
            "chunk_length": 200,
            "normalize": False,
            "format": resolved.format,
            "latency": resolved.latency,
            "condition_on_previous_chunks": True,
        }
        if voice_id:
            payload["reference_id"] = voice_id

        response = None
        try:
            response = self._http_session().post(
                f"{self._base_url}/v1/tts/stream/with-timestamp",
                headers={
                    "Authorization": f"Bearer {self._require_key()}",
                    "Content-Type": "application/json",
                    "model": FISH_MODEL,
                },
                json=payload,
                stream=True,
                timeout=self._timeout,
            )
            self._raise_for_status(response)
            for raw_line in response.iter_lines(decode_unicode=True):
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8")
                if not raw_line or raw_line.startswith(":"):
                    continue
                if not raw_line.startswith("data:"):
                    continue
                try:
                    item = json.loads(raw_line.removeprefix("data:").strip())
                    audio = base64.b64decode(item["audio_base64"], validate=True)
                    chunk_seq = int(item["chunk_seq"])
                    content = str(item.get("content", ""))
                    snapshot = item.get("alignment")
                    if snapshot is None:
                        alignment = None
                    else:
                        offset = float(item.get("chunk_audio_offset_sec", 0.0))
                        alignment = tuple(
                            AlignmentSegment(
                                text=str(segment["text"]),
                                start=float(segment["start"]) + offset,
                                end=float(segment["end"]) + offset,
                                chunk_seq=chunk_seq,
                            )
                            for segment in snapshot["segments"]
                        )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise SpeechError(
                        f"invalid Fish Audio stream event: {exc}", kind="protocol"
                    ) from exc
                yield SpeechEvent(
                    audio=audio,
                    content=content,
                    chunk_seq=chunk_seq,
                    alignment=alignment,
                )
        except SpeechError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize HTTP client failures
            raise SpeechError(f"Fish Audio transport failed: {exc}", kind="transport") from exc
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if close is not None:
                    close()

    def synthesize(
        self, speech: str, options: SpeechOptions | None = None
    ) -> SynthesisResult:
        """Collect a complete result while retaining final timestamp snapshots."""
        resolved = options or SpeechOptions()
        directed_text = resolved.compose(speech)
        return collect_speech_events(
            self.stream(speech, resolved),
            format=resolved.format,
            directed_text=directed_text,
        )

    def enroll_voice(
        self,
        audio_paths: Sequence[Path],
        transcripts: Sequence[str],
        *,
        title: str,
    ) -> VoiceEnrollment:
        """Create a private, instantly available persistent TTS voice."""
        if not audio_paths:
            raise ValueError("at least one enrollment sample is required")
        if len(audio_paths) != len(transcripts):
            raise ValueError("each enrollment sample requires one transcript")
        if not title.strip():
            raise ValueError("voice title cannot be empty")
        if any(not transcript.strip() for transcript in transcripts):
            raise ValueError("voice transcripts cannot be empty")

        try:
            voices = [Path(path).read_bytes() for path in audio_paths]
        except OSError as exc:
            raise SpeechError(f"cannot read voice sample: {exc}", kind="validation") from exc

        client = self._voice_client
        if client is None:
            try:
                from fishaudio import FishAudio
            except ImportError as exc:
                raise SpeechError(
                    "Fish Audio support is not installed; install echora[fish]",
                    kind="dependency",
                ) from exc
            client = FishAudio(api_key=self._require_key())

        try:
            voice = client.voices.create(
                title=title.strip(),
                voices=voices,
                texts=[text.strip() for text in transcripts],
                visibility="private",
                train_mode="fast",
                enhance_audio_quality=True,
            )
        except SpeechError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK exception hierarchy is optional
            raise SpeechError(f"Fish Audio enrollment failed: {exc}") from exc

        voice_id = getattr(voice, "id", None) or getattr(voice, "_id", None)
        if not voice_id:
            raise SpeechError("Fish Audio enrollment returned no voice ID", kind="protocol")
        return VoiceEnrollment(
            id=str(voice_id),
            title=str(getattr(voice, "title", title.strip())),
            state=str(getattr(voice, "state", "created")),
        )
