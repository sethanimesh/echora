"""FastAPI entry point for the Echora backend."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import providers_router
from app.api import router as transcription_router
from app.config import get_settings
from app.database import initialise_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Prepare local POC storage before accepting requests."""

    settings = get_settings()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    initialise_database()
    yield


app = FastAPI(
    title="Echora API",
    version="0.1.0",
    description="Local proof-of-concept backend for Echora speech recognition.",
    lifespan=lifespan,
)
app.include_router(transcription_router)
app.include_router(providers_router)


@app.get("/healthz", tags=["system"])
def healthcheck() -> dict[str, str]:
    """Report that the service is accepting requests."""

    settings = get_settings()
    return {
        "status": "ok",
        "service": "echora-api",
        "environment": settings.environment,
    }
