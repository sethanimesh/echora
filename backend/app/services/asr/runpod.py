"""Runpod Serverless adapter for the repository-owned Transformers worker."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from app.services.asr.base import (
    ASRHypothesisResult,
    ASRSegmentResult,
    ProviderCapabilities,
    ProviderConfigurationError,
    ProviderUpstreamError,
)


class RunpodASRProvider:
    """Send retained audio to the dedicated Runpod serverless worker contract."""

    name = "runpod"
    capabilities = ProviderCapabilities(
        n_best_maximum=5,
        segment_timestamps=True,
        word_timestamps=True,
        language_hint=True,
        acoustic_scores=True,
    )

    def __init__(
        self,
        *,
        endpoint_id: str | None,
        api_key: str | None,
        model_id: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    @property
    def configuration_error(self) -> str | None:
        missing = []
        if not self.endpoint_id:
            missing.append("ECHORA_RUNPOD_ENDPOINT_ID")
        if not self.api_key:
            missing.append("ECHORA_RUNPOD_API_KEY")
        if missing:
            return f"{', '.join(missing)} is not configured."
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

        request_body = {
            "input": {
                "audio_base64": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                "filename": original_filename,
                "language": language,
                "model_id": self.model_id,
                "n_best": n_best,
            }
        }
        endpoint_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    endpoint_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request_body,
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise ProviderUpstreamError(f"Runpod transcription failed: {error}") from error

        try:
            payload = response.json()
            output = payload["output"]
            hypotheses = output["hypotheses"]
        except (KeyError, TypeError, ValueError) as error:
            message = "Runpod returned an invalid transcription response."
            raise ProviderUpstreamError(message) from error

        try:
            return [
                ASRHypothesisResult(
                    rank=int(hypothesis["rank"]),
                    text=str(hypothesis["text"]),
                    score=float(hypothesis["score"]),
                    score_type=str(hypothesis["score_type"]),
                    segments=[
                        ASRSegmentResult(
                            segment_index=int(segment["segment_index"]),
                            start_seconds=float(segment["start_seconds"]),
                            end_seconds=float(segment["end_seconds"]),
                            text=str(segment["text"]),
                        )
                        for segment in hypothesis.get("segments", [])
                    ],
                )
                for hypothesis in hypotheses
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderUpstreamError("Runpod returned malformed ASR hypotheses.") from error
