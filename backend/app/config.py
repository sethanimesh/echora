"""Application configuration sourced from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Settings for the local proof-of-concept deployment."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="ECHORA_",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    database_url: str | None = None
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    asr_default_provider: str = "fake"
    asr_request_timeout_seconds: float = Field(default=180, gt=0)
    runpod_endpoint_id: str | None = None
    runpod_api_key: str | None = None
    runpod_model_id: str = "openai/whisper-large-v3"
    groq_api_key: str | None = None
    groq_model_id: str = "whisper-large-v3"

    @model_validator(mode="after")
    def set_default_database_url(self) -> Settings:
        if self.database_url is None:
            self.database_url = f"sqlite:///{self.data_dir / 'echora.db'}"
        if self.asr_default_provider not in {"fake", "groq", "runpod"}:
            raise ValueError("ECHORA_ASR_DEFAULT_PROVIDER must be fake, groq, or runpod.")
        return self

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"


@lru_cache
def get_settings() -> Settings:
    """Return cached process-wide settings."""

    return Settings()
