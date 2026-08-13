"""Groq Whisper adapter, limited to the one-best capabilities it exposes."""

from __future__ import annotations

from pathlib import Path

import httpx

from app.services.asr.base import (
    ASRHypothesisResult,
    ASRSegmentResult,
    ProviderCapabilities,
    ProviderConfigurationError,
    ProviderUpstreamError,
)


class GroqASRProvider:
    """Adapt Groq's OpenAI-compatible transcription endpoint to Echora's contract."""

    name = "groq"
    capabilities = ProviderCapabilities(
        n_best_maximum=1,
        segment_timestamps=True,
        word_timestamps=True,
        language_hint=True,
        acoustic_scores=False,
    )

    def __init__(
        self,
        *,
        api_key: str | None,
        model_id: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    @property
    def configuration_error(self) -> str | None:
        if not self.api_key:
            return "ECHORA_GROQ_API_KEY is not configured."
        return None

    async def transcribe(
        self,
        audio_path: Path,
        *,
        original_filename: str,
        language: str,
        n_best: int,
    ) -> list[ASRHypothesisResult]:
        if self.configuration_error:
            raise ProviderConfigurationError(self.configuration_error)
        if n_best != 1:
            raise ProviderConfigurationError("Groq accepts only one raw hypothesis.")

        with audio_path.open("rb") as audio_file:
            files = {"file": (original_filename, audio_file, "application/octet-stream")}
            data = {
                "model": self.model_id,
                "language": language,
                "response_format": "verbose_json",
                "timestamp_granularities[]": ["segment", "word"],
            }
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds, transport=self.transport
                ) as client:
                    response = await client.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data=data,
                        files=files,
                    )
                    response.raise_for_status()
            except httpx.HTTPError as error:
                raise ProviderUpstreamError(f"Groq transcription failed: {error}") from error

        try:
            payload = response.json()
            text = payload["text"]
        except (KeyError, TypeError, ValueError) as error:
            message = "Groq returned an invalid transcription response."
            raise ProviderUpstreamError(message) from error

        segments = [
            ASRSegmentResult(
                segment_index=index,
                start_seconds=float(segment["start"]),
                end_seconds=float(segment["end"]),
                text=str(segment["text"]),
            )
            for index, segment in enumerate(payload.get("segments", []))
        ]
        return [
            ASRHypothesisResult(
                rank=1,
                text=str(text),
                score=0.0,
                score_type="unavailable",
                segments=segments,
            )
        ]
