"""Shared API and inference schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CommunicationContext = Literal["general", "home", "care", "outdoors"]


class Hypothesis(BaseModel):
    id: str
    literal_text: str
    sequence_score: float
    search_weight: float = Field(ge=0.0, le=1.0)


class AudioQuality(BaseModel):
    seconds: float
    peak_dbfs: float | None
    rms_dbfs: float | None
    clipped_samples: int
    low_level_warning: bool


class RawAsrResult(BaseModel):
    backend: Literal["local", "pod", "runpod"]
    model: str = "echora-qwen3-asr-command-v3"
    device: str
    hypotheses: list[Hypothesis]
    audio_quality: AudioQuality
    decode_seconds: float


class RankerDecision(BaseModel):
    decision: Literal["selected", "ambiguous"]
    selected_message_id: str | None = None
    display_hypothesis_ids: list[str]
    display_message_ids: list[str] = Field(default_factory=list)
    reason: str
    source: Literal["groq", "unavailable"]
    assistant_model: str | None = None


class MessageCandidate(BaseModel):
    message_id: str
    hypothesis_id: str
    source_hypothesis_ids: list[str]
    source_literals: list[str]
    literal_text: str
    interpreted_intent: str
    corrected_text: str
    repair_status: Literal["unchanged", "corrected", "unavailable"]
    repair_note: str
    # Chosen word -> other variants the recognizer produced at that slot, so the
    # interface can mark an uncertain word inline instead of showing rival sentences.
    word_alternatives: dict[str, list[str]] = Field(default_factory=dict)


class Timing(BaseModel):
    audio_decode_seconds: float
    asr_seconds: float
    ranking_seconds: float
    grammar_seconds: float
    total_seconds: float


class TranscriptionResponse(BaseModel):
    request_id: str
    backend: str
    model: str
    device: str
    context: CommunicationContext
    hypotheses: list[Hypothesis]
    audio_quality: AudioQuality
    ranker: RankerDecision
    messages: list[MessageCandidate]
    recommended_message_id: str | None
    needs_user_choice: bool
    beam_weights_are_calibrated_confidence: bool = False
    user_confirmation_required: bool = True
    timing: Timing
    warnings: list[str]


class RemoteTranscriptionRequest(BaseModel):
    audio_base64: str
    filename: str = "recording.wav"
    beams: int = Field(default=5, ge=2, le=8)


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded", "starting"]
    asr_backend: str
    model_ready: bool
    groq_configured: bool
    detail: str


class ModelInfo(BaseModel):
    name: str
    foundation: str
    adapter_sha256: str
    device: str
    beams: int
    literal_prompt: str
    supported_contexts: list[CommunicationContext]
    beam_weights_are_calibrated_confidence: bool = False
    semantic_repair_is_part_of_asr: bool = False
    limitations: list[str]
