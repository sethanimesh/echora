"""Raw-ASR FastAPI service for a persistent dedicated RunPod Pod."""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

from .asr import LocalAsrBackend
from .asr.engine import EXPECTED_ADAPTER_SHA256, EXPECTED_FOUNDATION_SHA256
from .audio import AudioValidationError, decode_audio
from .config import load_settings
from .schemas import RawAsrResult, RemoteTranscriptionRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    if settings.backend != "local":
        raise RuntimeError("The Pod worker requires ECHORA_ASR_BACKEND=local")
    app.state.settings = settings
    app.state.asr = LocalAsrBackend(settings)
    yield


app = FastAPI(title="Echora raw ASR worker", version="0.1.0", lifespan=lifespan)


async def authorize(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("ECHORA_WORKER_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="ECHORA_WORKER_TOKEN is not configured")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid worker token")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ready",
        "model": "echora-qwen3-asr-command-v3",
        "device": app.state.asr.device,
        "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
    }


@app.post("/v1/transcribe", response_model=RawAsrResult, dependencies=[Depends(authorize)])
async def transcribe(request: RemoteTranscriptionRequest) -> RawAsrResult:
    try:
        payload = base64.b64decode(request.audio_base64, validate=True)
        waveform = decode_audio(
            payload,
            request.filename,
            app.state.settings.max_audio_seconds,
        )
    except (ValueError, AudioValidationError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    result = await app.state.asr.transcribe(waveform, request.beams)
    return result.model_copy(update={"backend": "pod"})
