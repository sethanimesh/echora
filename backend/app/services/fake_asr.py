"""Deterministic ASR implementation used until a real provider is configured."""

from __future__ import annotations

from pathlib import Path

from app.services.asr.base import ASRHypothesisResult, ProviderCapabilities


class FakeASRProvider:
    """Return a stable result so the local API can be exercised without credentials."""

    name = "fake"
    capabilities = ProviderCapabilities(
        n_best_maximum=1,
        segment_timestamps=False,
        word_timestamps=False,
        language_hint=True,
        acoustic_scores=False,
    )

    @property
    def configuration_error(self) -> None:
        """The fake provider is always available for local testing."""

        return None

    async def transcribe(
        self,
        audio_path: Path,
        *,
        original_filename: str,
        language: str,
        n_best: int,
    ) -> list[ASRHypothesisResult]:
        """Return a deterministic fixture rather than inspecting user audio."""

        return [
            ASRHypothesisResult(
                rank=1,
                text="This is a deterministic fake transcript.",
                score=1.0,
                score_type="deterministic_fixture",
            )
        ]
