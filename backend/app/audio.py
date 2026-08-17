"""Audio decoding, normalization, validation, and quality descriptors."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .schemas import AudioQuality


SAMPLE_RATE = 16_000
ALLOWED_SUFFIXES = {".wav", ".webm", ".ogg", ".oga", ".m4a", ".mp3", ".mp4"}


class AudioValidationError(ValueError):
    pass


def _soundfile_decode(data: bytes) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return np.asarray(audio), int(rate)


def _ffmpeg_decode(data: bytes, suffix: str) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise AudioValidationError("This audio format requires ffmpeg, but ffmpeg is unavailable")
    with tempfile.TemporaryDirectory(prefix="echora-audio-") as directory:
        source = Path(directory) / f"input{suffix}"
        target = Path(directory) / "normalized.wav"
        source.write_bytes(data)
        process = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                "-y",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if process.returncode != 0 or not target.is_file():
            detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "decode failed"
            raise AudioValidationError(f"Could not decode the recording: {detail}")
        return _soundfile_decode(target.read_bytes())


def decode_audio(data: bytes, filename: str, max_seconds: float) -> np.ndarray:
    if not data:
        raise AudioValidationError("The recording is empty")
    suffix = Path(filename or "recording.wav").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise AudioValidationError(
            "Unsupported audio type. Use WAV, WebM, OGG, M4A, MP3, or MP4 audio"
        )
    try:
        audio, rate = _soundfile_decode(data)
    except Exception:
        audio, rate = _ffmpeg_decode(data, suffix)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not len(audio):
        raise AudioValidationError("The recording contains no audio samples")
    audio = np.nan_to_num(audio.astype(np.float32), copy=False)
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE).astype(np.float32)
    seconds = len(audio) / SAMPLE_RATE
    if seconds > max_seconds:
        raise AudioValidationError(
            f"Recording is {seconds:.1f}s; the maximum is {max_seconds:.0f}s"
        )
    if seconds < 0.15:
        raise AudioValidationError("The recording is too short")
    if float(np.max(np.abs(audio))) < 1e-5:
        raise AudioValidationError("No audible signal was detected")
    return np.ascontiguousarray(audio)


def encode_wav(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def audio_quality(audio: np.ndarray) -> AudioQuality:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0

    def dbfs(value: float) -> float | None:
        return round(20 * math.log10(value), 2) if value > 0 else None

    return AudioQuality(
        seconds=round(len(audio) / SAMPLE_RATE, 3),
        peak_dbfs=dbfs(peak),
        rms_dbfs=dbfs(rms),
        clipped_samples=int(np.sum(np.abs(audio) > 0.99)),
        low_level_warning=bool(peak > 0 and 20 * math.log10(peak) < -30.0),
    )
