"""HTTP routes for local transcription records."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.database import get_db
from app.models import ASRHypothesis, ASRSegment, Transcription
from app.schemas import ProviderStatusResponse, TranscriptionResponse
from app.services.asr.base import ProviderConfigurationError, ProviderUpstreamError
from app.services.asr.registry import (
    ProviderRegistry,
    enforce_capabilities,
    get_provider_registry,
)
from app.services.storage import delete_stored_audio, save_upload

router = APIRouter(prefix="/v1/transcriptions", tags=["transcriptions"])
providers_router = APIRouter(prefix="/v1/providers", tags=["providers"])


def get_transcription_or_404(database: Session, transcription_id: str) -> Transcription:
    """Load a transcription record and nested results, or raise a standard 404."""

    statement = (
        database.query(Transcription)
        .options(selectinload(Transcription.hypotheses).selectinload(ASRHypothesis.segments))
        .filter(Transcription.id == transcription_id)
    )
    transcription = statement.one_or_none()
    if transcription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not found.",
        )
    return transcription


@router.post("", response_model=TranscriptionResponse, status_code=status.HTTP_201_CREATED)
async def create_transcription(
    file: UploadFile = File(...),
    provider_name: str | None = Form(None, alias="provider"),
    language: str = Form("en"),
    n_best: int = Form(1, ge=1, le=5),
    database: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    provider_registry: ProviderRegistry = Depends(get_provider_registry),
) -> Transcription:
    """Persist audio locally and create a normalized raw-ASR result."""

    if language != "en":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only English (en) is supported by this proof of concept.",
        )
    asr_provider = provider_registry.get(provider_name or settings.asr_default_provider)
    enforce_capabilities(asr_provider, n_best)
    if asr_provider.configuration_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=asr_provider.configuration_error,
        )

    storage_name, size_bytes, digest = await save_upload(file, settings)
    audio_path = settings.audio_dir / storage_name
    try:
        hypotheses = await asr_provider.transcribe(
            audio_path,
            original_filename=Path(file.filename or "audio").name,
            language=language,
            n_best=n_best,
        )
    except ProviderConfigurationError as error:
        delete_stored_audio(settings.audio_dir, storage_name)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except ProviderUpstreamError as error:
        delete_stored_audio(settings.audio_dir, storage_name)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    transcription = Transcription(
        provider=asr_provider.name,
        language=language,
        original_filename=Path(file.filename or "audio").name,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=size_bytes,
        sha256=digest,
        audio_storage_name=storage_name,
    )
    transcription.hypotheses = [
        ASRHypothesis(
            rank=hypothesis.rank,
            text=hypothesis.text,
            score=hypothesis.score,
            score_type=hypothesis.score_type,
            segments=[
                ASRSegment(
                    segment_index=segment.segment_index,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    text=segment.text,
                )
                for segment in hypothesis.segments
            ],
        )
        for hypothesis in hypotheses
    ]
    database.add(transcription)
    try:
        database.commit()
    except Exception:
        database.rollback()
        delete_stored_audio(settings.audio_dir, storage_name)
        raise
    database.refresh(transcription)
    return get_transcription_or_404(database, transcription.id)


@router.get("/{transcription_id}", response_model=TranscriptionResponse)
def get_transcription(transcription_id: str, database: Session = Depends(get_db)) -> Transcription:
    """Retrieve one stored transcription result without exposing its audio path."""

    return get_transcription_or_404(database, transcription_id)


@router.delete("/{transcription_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transcription(
    transcription_id: str,
    database: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Remove both the local audio object and its persisted result."""

    transcription = get_transcription_or_404(database, transcription_id)
    delete_stored_audio(settings.audio_dir, transcription.audio_storage_name)
    database.delete(transcription)
    database.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@providers_router.get("", response_model=list[ProviderStatusResponse])
def list_providers(
    provider_registry: ProviderRegistry = Depends(get_provider_registry),
) -> list[dict[str, object]]:
    """List installed providers and their honest normalized capabilities."""

    return provider_registry.provider_statuses()
