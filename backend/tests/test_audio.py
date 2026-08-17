from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.audio import AudioValidationError, audio_quality, decode_audio, encode_wav


def tone(seconds: float = 0.5) -> np.ndarray:
    samples = np.arange(int(16_000 * seconds), dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 220 * samples / 16_000)).astype(np.float32)


def test_wav_round_trip_and_quality() -> None:
    audio = tone()
    decoded = decode_audio(encode_wav(audio), "voice.wav", 45)
    assert decoded.dtype == np.float32
    assert len(decoded) == len(audio)
    quality = audio_quality(decoded)
    assert quality.seconds == 0.5
    assert quality.peak_dbfs is not None


def test_stereo_is_mixed_and_resampled() -> None:
    source = np.stack([tone(), tone() * 0.5], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, source[::2], 8_000, format="WAV")
    decoded = decode_audio(buffer.getvalue(), "stereo.wav", 45)
    assert 7_900 <= len(decoded) <= 8_100


@pytest.mark.parametrize(
    ("payload", "filename", "message"),
    [
        (b"", "voice.wav", "empty"),
        (b"not audio", "voice.xyz", "Unsupported"),
        (encode_wav(np.zeros(16_000, dtype=np.float32)), "voice.wav", "audible"),
    ],
)
def test_invalid_audio(payload: bytes, filename: str, message: str) -> None:
    with pytest.raises(AudioValidationError, match=message):
        decode_audio(payload, filename, 45)


def test_duration_limit() -> None:
    with pytest.raises(AudioValidationError, match="maximum"):
        decode_audio(encode_wav(tone(1.1)), "voice.wav", 1.0)
