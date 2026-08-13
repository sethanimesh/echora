"""Normalized contracts shared by every ASR provider adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class ProviderConfigurationError(Exception):
    """Raised when an installed provider has not received its required settings."""


class ProviderUpstreamError(Exception):
    """Raised when an upstream ASR request cannot return a valid response."""


@dataclass(frozen=True)
class ProviderCapabilities:
    """Features a provider can honestly supply through the normalized API."""

    n_best_maximum: int
    segment_timestamps: bool
    word_timestamps: bool
    language_hint: bool
    acoustic_scores: bool


@dataclass(frozen=True)
class ASRSegmentResult:
    """One timestamped segment returned by a provider."""

    segment_index: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class ASRHypothesisResult:
    """One raw ASR alternative with score semantics explicitly declared."""

    rank: int
    text: str
    score: float
    score_type: str
    segments: list[ASRSegmentResult] = field(default_factory=list)


class ASRProvider(Protocol):
    """Stable adapter interface that lets the API switch ASR vendors."""

    name: str
    capabilities: ProviderCapabilities

    @property
    def configuration_error(self) -> str | None:
        """Explain missing configuration without exposing a secret."""

    async def transcribe(
        self,
        audio_path: Path,
        *,
        original_filename: str,
        language: str,
        n_best: int,
    ) -> list[ASRHypothesisResult]:
        """Return normalized raw ASR hypotheses for the supplied retained audio."""
