"""HTTP response schemas for transcription records."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SegmentResponse(BaseModel):
    """One timestamped piece of an ASR hypothesis."""

    model_config = ConfigDict(from_attributes=True)

    segment_index: int
    start_seconds: float
    end_seconds: float
    text: str


class HypothesisResponse(BaseModel):
    """One raw ASR hypothesis and any timestamps supplied by its provider."""

    model_config = ConfigDict(from_attributes=True)

    rank: int
    text: str
    score: float
    score_type: str
    segments: list[SegmentResponse]


class TranscriptionResponse(BaseModel):
    """The public representation of a local transcription record."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    provider: str
    language: str
    original_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    created_at: datetime
    completed_at: datetime
    hypotheses: list[HypothesisResponse]


class ProviderCapabilitiesResponse(BaseModel):
    """Public capability information for an installed ASR provider."""

    n_best_maximum: int
    segment_timestamps: bool
    word_timestamps: bool
    language_hint: bool
    acoustic_scores: bool


class ProviderStatusResponse(BaseModel):
    """Safe configuration and feature state for one installed provider."""

    name: str
    configured: bool
    configuration_error: str | None
    capabilities: ProviderCapabilitiesResponse
