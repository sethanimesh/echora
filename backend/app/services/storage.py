"""Local-only audio storage used by the proof of concept."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.config import Settings

ALLOWED_AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm"}
ALLOWED_AUDIO_CONTENT_TYPES = {
    "application/octet-stream",
    "audio/flac",
    "audio/m4a",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
}
CHUNK_SIZE = 1024 * 1024


def validate_upload(file: UploadFile) -> str:
    """Validate the filename and declared media type before local persistence."""

    original_filename = file.filename or ""
    extension = Path(original_filename).suffix.lower()
    content_type = file.content_type or "application/octet-stream"

    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio file extension.",
        )
    if content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio content type.",
        )
    return extension


async def iter_limited_chunks(file: UploadFile, max_upload_bytes: int) -> AsyncIterator[bytes]:
    """Yield upload chunks and stop before a file exceeds the configured limit."""

    total = 0
    while chunk := await file.read(CHUNK_SIZE):
        total += len(chunk)
        if total > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio upload exceeds the {max_upload_bytes}-byte limit.",
            )
        yield chunk


async def save_upload(file: UploadFile, settings: Settings) -> tuple[str, int, str]:
    """Write an upload locally and return storage name, bytes, and SHA-256 digest."""

    extension = validate_upload(file)
    storage_name = f"{uuid4()}{extension}"
    destination = settings.audio_dir / storage_name
    digest = hashlib.sha256()
    size_bytes = 0

    try:
        with destination.open("xb") as audio_file:
            async for chunk in iter_limited_chunks(file, settings.max_upload_bytes):
                audio_file.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    return storage_name, size_bytes, digest.hexdigest()


def delete_stored_audio(audio_dir: Path, storage_name: str) -> None:
    """Delete a local POC audio object without accepting a user-controlled path."""

    (audio_dir / Path(storage_name).name).unlink(missing_ok=True)
