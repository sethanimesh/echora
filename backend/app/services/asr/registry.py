"""Provider discovery, configuration checks, and capability enforcement."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import Depends, HTTPException, status

from app.config import Settings, get_settings
from app.services.asr.base import ASRProvider
from app.services.asr.groq import GroqASRProvider
from app.services.asr.runpod import RunpodASRProvider
from app.services.fake_asr import FakeASRProvider


class ProviderRegistry:
    """Construct every installed provider from one application configuration."""

    def __init__(self, settings: Settings) -> None:
        self.providers: dict[str, ASRProvider] = {
            "fake": FakeASRProvider(),
            "groq": GroqASRProvider(
                api_key=settings.groq_api_key,
                model_id=settings.groq_model_id,
                timeout_seconds=settings.asr_request_timeout_seconds,
            ),
            "runpod": RunpodASRProvider(
                endpoint_id=settings.runpod_endpoint_id,
                api_key=settings.runpod_api_key,
                model_id=settings.runpod_model_id,
                timeout_seconds=settings.asr_request_timeout_seconds,
            ),
        }

    def get(self, requested_provider: str | None) -> ASRProvider:
        """Return the chosen provider and guard configuration/capabilities at the API edge."""

        provider_name = requested_provider or ""
        if not provider_name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="No ASR provider was selected.",
            )
        provider = self.providers.get(provider_name)
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown ASR provider: {provider_name}.",
            )
        return provider

    def provider_statuses(self) -> list[dict[str, object]]:
        """Return safe capability/configuration information for each installed provider."""

        return [
            {
                "name": provider.name,
                "configured": provider.configuration_error is None,
                "configuration_error": provider.configuration_error,
                "capabilities": asdict(provider.capabilities),
            }
            for provider in self.providers.values()
        ]


def get_provider_registry(settings: Settings = Depends(get_settings)) -> ProviderRegistry:
    """FastAPI dependency constructing the current provider registry."""

    return ProviderRegistry(settings)


def enforce_capabilities(provider: ASRProvider, n_best: int) -> None:
    """Reject an incompatible request before retaining audio or calling a vendor."""

    if n_best > provider.capabilities.n_best_maximum:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Requested n_best={n_best}, but provider '{provider.name}' supports at most "
                f"{provider.capabilities.n_best_maximum}."
            ),
        )
