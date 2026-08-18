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
from .personal import USER_PROFILE_ID, Personalizer
from .places import PlaceStore, default_settings
from .schemas import (
    AcceptedMessageRequest,
    AcceptedMessageResponse,
    CommunicationContext,
    HealthResponse,
    Listener,
    ModelInfo,
    PersonaSummary,
    PlaceSettings,
    PlaceSettingsRequest,
    ProfileRequest,
    ProfileResponse,
    SpeechAudio,
    SpeechRequest,
    Timing,
    TranscriptionResponse,
    default_listener,
)


SUPPORTED_CONTEXTS: tuple[CommunicationContext, ...] = ("general", "home", "care", "outdoors")
SUPPORTED_LISTENERS: tuple[Listener, ...] = ("familiar", "unfamiliar")


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
    # Personal context loads like the recognizer -- eagerly, and never fatally.
    # Without it every message is produced exactly as it was before this layer.
    app.state.personal = None
    app.state.personal_error = None
    if settings.personal_enabled:
        try:
            app.state.personal = Personalizer(settings)
        except Exception as error:
            app.state.personal_error = f"{type(error).__name__}: {error}"
    # Places are not part of the personal layer on purpose: choosing a setting by
    # location has to keep working when personal context is off or no profile is
    # picked, so this store stands on its own and never fails the app.
    app.state.places = PlaceStore(settings.places_path)
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
        personal_ready=app.state.personal is not None,
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
        supported_listeners=list(SUPPORTED_LISTENERS),
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
    persona: str = Form(default=""),
    listener: str = Form(default=""),
) -> TranscriptionResponse:
    settings = _settings()
    if context not in SUPPORTED_CONTEXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported context. Choose one of: {', '.join(SUPPORTED_CONTEXTS)}",
        )
    # Empty means the caller has nothing to say about who is listening, which is
    # the ordinary case for a plain setting with no place behind it. A value the
    # server does not know is a different matter and is refused, exactly as an
    # unknown setting is: the stance is part of the prompt.
    if listener and listener not in SUPPORTED_LISTENERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported listener. Choose one of: {', '.join(SUPPORTED_LISTENERS)}",
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
    # Who is listening, strongest claim first: what the caller sent (a place the
    # speaker tapped, which is the only thing that knows the room they are
    # actually in), then what the profile says it usually is for this setting,
    # then the setting's own default. Every step falls back, so an absent or
    # unloadable profile lands exactly where the setting alone would.
    selected_listener: Listener = default_listener(selected_context)
    if listener:
        selected_listener = listener  # type: ignore[assignment]
    elif persona and app.state.personal is not None:
        declared = app.state.personal.listener_for(persona, selected_context)
        if declared is not None:
            selected_listener = declared
    # An unknown profile is a warning, never a rejection. Unlike the setting, it is
    # not part of the prompt, and a stale value in the interface must never stand
    # between the speaker and a spoken message.
    brief = None
    profile_warning: str | None = None
    if persona and app.state.personal is not None:
        brief = await app.state.personal.brief(
            raw.hypotheses, selected_context, persona, selected_listener
        )
        if persona not in app.state.personal.known_ids():
            profile_warning = f"Unknown profile '{persona}'; this message was not personalized"
    chain = await app.state.message_chain.run(
        raw.hypotheses, selected_context, brief=brief, listener=selected_listener
    )
    recommended = chain.ranker.selected_message_id
    warnings = list(chain.warnings)
    if profile_warning:
        warnings.append(profile_warning)
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
        listener=selected_listener,
        persona=persona or None,
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


@app.get("/api/v1/personas", response_model=list[PersonaSummary])
async def personas() -> list[PersonaSummary]:
    """The profiles the interface can offer. Empty when personal context is unavailable."""
    if app.state.personal is None:
        return []
    return app.state.personal.personas()


@app.post("/api/v1/accepted", response_model=AcceptedMessageResponse)
async def accepted(request: AcceptedMessageRequest) -> AcceptedMessageResponse:
    """Record a message the speaker settled on, so later utterances can learn from it.

    Never an error. Failing to remember something is not worth telling a speaker
    about mid-conversation, and the message has already been said.
    """
    if app.state.personal is None:
        return AcceptedMessageResponse(stored=False, reason="personal context is unavailable")
    stored, merged, reason = await app.state.personal.remember(
        request.persona,
        request.context,
        request.heard,
        request.message,
        request.listener,
    )
    return AcceptedMessageResponse(stored=stored, merged=merged, reason=reason)


@app.get("/api/v1/places", response_model=PlaceSettings)
async def read_places() -> PlaceSettings:
    """The known places and whether location may choose between them.

    Never an error. A first run, or a document that cannot be read, returns the
    three shipped places with detection off -- which is exactly how the
    interface behaved before places existed.
    """
    try:
        return await asyncio.to_thread(app.state.places.load)
    except Exception:
        return default_settings()


@app.post("/api/v1/places", response_model=PlaceSettings)
async def write_places(request: PlaceSettingsRequest) -> PlaceSettings:
    """Replace the whole document, and answer with what was actually stored.

    One writer and one validation path. The built-ins are re-inserted and their
    wording and context restored by `places.normalize`, so no request can leave
    the speaker with fewer than the three places that ship, and the response is
    the truth rather than an echo of what was asked for.
    """
    try:
        return await asyncio.to_thread(
            app.state.places.save,
            PlaceSettings(auto_detect=request.auto_detect, places=request.places),
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Places could not be saved: {error}")


@app.get("/api/v1/profile", response_model=ProfileResponse)
async def read_profile(persona: str = USER_PROFILE_ID) -> ProfileResponse:
    """What Echora currently thinks it knows, so the speaker can see and change it."""
    if app.state.personal is None:
        return ProfileResponse(saved=False, reason="personal context is unavailable")
    profile = app.state.personal.describe(persona)
    if profile is None:
        return ProfileResponse(saved=False, reason="no such profile")
    return ProfileResponse(
        saved=True,
        lexicon_size=len(profile.lexicon),
        specialization_size=len(profile.specializations),
    )


@app.post("/api/v1/profile", response_model=ProfileResponse)
async def write_profile(request: ProfileRequest) -> ProfileResponse:
    """Save the onboarding answers. Only the speaker's own profile is writable."""
    if app.state.personal is None:
        raise HTTPException(status_code=503, detail="Personal context is unavailable")
    profile, refused = await asyncio.to_thread(
        app.state.personal.save_user_profile,
        request.people,
        request.places,
        request.things,
        [(item.word, item.wording) for item in request.details],
    )
    return ProfileResponse(
        saved=True,
        lexicon_size=len(profile.lexicon),
        specialization_size=len(profile.specializations),
        refused=refused,
        reason="saved",
    )
