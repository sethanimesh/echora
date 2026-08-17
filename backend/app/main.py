"""FastAPI application for the local-first Echora demo."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .asr import create_backend
from .asr.engine import EXPECTED_ADAPTER_SHA256
from .audio import AudioValidationError, decode_audio
from .config import Settings, load_settings
from .messaging import GroqMessageChain, GroqSpeech
from .schemas import (
    CommunicationContext,
    HealthResponse,
    ModelInfo,
    SpeechAudio,
    SpeechRequest,
    Timing,
    TranscriptionResponse,
)


SUPPORTED_CONTEXTS: tuple[CommunicationContext, ...] = ("general", "home", "care", "outdoors")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.asr = None
    app.state.asr_error = None
    try:
        app.state.asr = create_backend(settings)
    except Exception as error:
        app.state.asr_error = f"{type(error).__name__}: {error}"
    app.state.message_chain = GroqMessageChain(settings)
    app.state.speech = GroqSpeech(settings)
    yield


app = FastAPI(
    title="Echora API",
    version="0.1.0",
    description="Literal dysarthric ASR followed by evidence-constrained message assistance.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _settings() -> Settings:
    return app.state.settings


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = _settings()
    ready = app.state.asr is not None
    return HealthResponse(
        status="ready" if ready and settings.groq_configured else "degraded",
        asr_backend=settings.backend,
        model_ready=ready,
        groq_configured=settings.groq_configured,
        detail=(
            f"{settings.backend} ASR and Groq are ready"
            if ready and settings.groq_configured
            else app.state.asr_error
            or "ASR is ready; Groq is not configured, so raw candidates will be shown"
        ),
    )


@app.get("/api/v1/model", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    settings = _settings()
    try:
        config = json.loads(settings.inference_config.read_text(encoding="utf-8"))
        prompt = config["literal_prompt"]
    except Exception:
        prompt = "Transcribe only the words actually spoken."
    device = app.state.asr.device if app.state.asr is not None else "unavailable"
    return ModelInfo(
        name="echora-qwen3-asr-command-v3",
        foundation="Qwen/Qwen3-ASR-1.7B-hf",
        adapter_sha256=EXPECTED_ADAPTER_SHA256,
        device=device,
        beams=settings.beams,
        literal_prompt=prompt,
        supported_contexts=list(SUPPORTED_CONTEXTS),
        limitations=[
            "TORGO contains only eight dysarthric speakers in the evaluation design.",
            "Synthetic command compositions do not create new speaker acoustics.",
            "Beam search weights are relative search evidence, not calibrated confidence.",
            "The message assistant cannot recover words absent from every ASR hypothesis.",
        ],
    )


@app.post("/api/v1/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(...),
    context: str = Form(default="general"),
) -> TranscriptionResponse:
    settings = _settings()
    if context not in SUPPORTED_CONTEXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported context. Choose one of: {', '.join(SUPPORTED_CONTEXTS)}",
        )
    if app.state.asr is None:
        raise HTTPException(status_code=503, detail=app.state.asr_error or "ASR is unavailable")
    started = time.perf_counter()
    data = await audio.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio upload exceeds the 20 MB limit")
    decode_started = time.perf_counter()
    try:
        waveform = await asyncio.to_thread(
            decode_audio, data, audio.filename or "recording.wav", settings.max_audio_seconds
        )
    except AudioValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    audio_decode_seconds = round(time.perf_counter() - decode_started, 3)
    try:
        raw = await app.state.asr.transcribe(waveform, settings.beams)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"ASR failed: {type(error).__name__}: {error}") from error
    selected_context: CommunicationContext = context  # type: ignore[assignment]
    chain = await app.state.message_chain.run(raw.hypotheses, selected_context)
    recommended = chain.ranker.selected_message_id
    warnings = list(chain.warnings)
    # One surviving message means nothing is left to disambiguate, so it is spoken
    # on arrival. Synthesizing it here rather than in a follow-up request is what
    # lets the browser play it the instant the screen renders.
    speech: SpeechAudio | None = None
    speech_seconds = 0.0
    if settings.speech_autoplay and chain.ranker.decision == "selected" and chain.messages:
        speech_started = time.perf_counter()
        speech = await app.state.speech.synthesize(chain.messages[0].corrected_text)
        speech_seconds = round(time.perf_counter() - speech_started, 3)
    if raw.audio_quality.low_level_warning:
        warnings.append("Recording level is low; move closer to the microphone")
    return TranscriptionResponse(
        request_id=str(uuid.uuid4()),
        backend=raw.backend,
        model=raw.model,
        device=raw.device,
        context=selected_context,
        hypotheses=raw.hypotheses,
        audio_quality=raw.audio_quality,
        ranker=chain.ranker,
        messages=chain.messages,
        recommended_message_id=recommended,
        needs_user_choice=chain.ranker.decision == "ambiguous",
        speech=speech,
        timing=Timing(
            audio_decode_seconds=audio_decode_seconds,
            asr_seconds=raw.decode_seconds,
            ranking_seconds=chain.ranking_seconds,
            grammar_seconds=chain.grammar_seconds,
            speech_seconds=speech_seconds,
            total_seconds=round(time.perf_counter() - started, 3),
        ),
        warnings=warnings,
    )


@app.post("/api/v1/speech", response_model=SpeechAudio)
async def speak(request: SpeechRequest) -> SpeechAudio:
    """Synthesize a message the speaker asked to hear again, or edited by hand.

    A 503 here is not an error the speaker should see: the interface falls back to
    the browser's own voice, so the message is still spoken.
    """
    audio = await app.state.speech.synthesize(request.text)
    if audio is None:
        raise HTTPException(status_code=503, detail="Groq speech is unavailable")
    return audio
