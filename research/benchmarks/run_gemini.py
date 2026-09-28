"""Gemini audio-understanding as a literal ASR recognizer.

This runner exists to answer one question with numbers: does a large cloud
multimodal model transcribe dysarthric speech better than the tuned
command-v3 adapter?  It is a measurement, not a deployment path -- Gemini
returns one fluent string per utterance, and the rest of Echora reasons over
beams, so a favourable WER here would still not drop into `backend/app/asr/`
unchanged.

It uses the REST endpoint directly rather than google-genai so the benchmark
adds no dependency to the project venv.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import sys
import time
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

import common


MODEL = os.environ.get("ECHORA_GEMINI_MODEL", "gemini-3.5-flash")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# The instruction the adapter itself was trained with, copied verbatim from
# research/training/configs/qwen_command_v3.json. Using anything richer here --
# a vocabulary, a hint that these are short commands -- would hand Gemini a
# prior the local model never got, and the comparison would stop meaning
# anything.
LITERAL_PROMPT = (
    "Transcribe only the words actually spoken. Keep incomplete or unusual word "
    "sequences literal. Do not add missing words."
)
INSTRUCTION = (
    LITERAL_PROMPT
    + " The audio is English. Reply with the transcription only: no quotes, no "
    "commentary, no explanation, and no note about audio quality. If you cannot "
    "make out any words, reply with an empty line."
)

MAX_ATTEMPTS = 6
WORKERS = int(os.environ.get("ECHORA_GEMINI_WORKERS", "4"))
# Free-tier AI Studio meters requests per minute per model, and for the capable
# flash models that allowance is 5. Pacing to it costs nothing -- the limit, not
# concurrency, sets the wall clock -- and keeps the run off the retry path,
# where a 429 otherwise parks a worker for the better part of a minute.
REQUESTS_PER_MINUTE = float(os.environ.get("ECHORA_GEMINI_RPM", "5"))


class _Pacer:
    """Space requests across threads by at least one RPM slot."""

    def __init__(self, per_minute: float) -> None:
        self._interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)


def retry_delay(payload: dict) -> float | None:
    """The wait the API itself asked for, which beats guessing at a backoff."""
    for detail in payload.get("error", {}).get("details", []):
        if detail.get("@type", "").endswith("RetryInfo"):
            value = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                return float(value)
            except ValueError:
                return None
    return None


def is_daily_quota(payload: dict) -> bool:
    """A per-day exhaustion never clears inside a run; spinning on it is waste."""
    for detail in payload.get("error", {}).get("details", []):
        if detail.get("@type", "").endswith("QuotaFailure"):
            for violation in detail.get("violations", []):
                if "PerDay" in violation.get("quotaId", ""):
                    return True
    return False


def api_key() -> str:
    key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY", "").strip()
    if key:
        return key
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GOOGLE_AI_STUDIO_API_KEY":
                return value.strip().strip('"').strip("'")
    raise RuntimeError("GOOGLE_AI_STUDIO_API_KEY is not set and not present in .env")


def encode_wav(audio: np.ndarray) -> bytes:
    """16 kHz mono float32 -> 16-bit PCM WAV, the format common.load_audio yields."""
    clipped = np.clip(audio, -1.0, 1.0)
    samples = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(common.SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def clean(text: str) -> str:
    """Strip the wrapper a chat model adds around a transcript.

    Only formatting is removed. Nothing here repairs or second-guesses the
    words, because the words are exactly what is being measured.
    """
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    # Occasional "Transcription: foo" / "The audio says: foo" preambles.
    text = re.sub(r"^(transcription|transcript|the audio says)\s*:\s*", "", text, flags=re.I)
    return text.strip()


class DailyQuotaExhausted(RuntimeError):
    """Raised so a spent daily allowance stops the run instead of spinning."""


def _warn(attempt: int, detail: str) -> None:
    """A stalled run must say why. An overloaded model backs off for minutes,
    and a silent retry loop is indistinguishable from a hang."""
    print(f"  retry {attempt + 1}/{MAX_ATTEMPTS}: {detail}", file=sys.stderr, flush=True)


def load_batch(device: str):
    key = api_key()
    url = ENDPOINT.format(model=MODEL)
    pacer = _Pacer(REQUESTS_PER_MINUTE)

    def transcribe_one(client: httpx.Client, audio: np.ndarray) -> str:
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": INSTRUCTION},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(encode_wav(audio)).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
        }
        last_error = ""
        for attempt in range(MAX_ATTEMPTS):
            pacer.wait()
            try:
                response = client.post(url, params={"key": key}, json=body, timeout=180)
            except httpx.HTTPError as error:  # transport failures are worth retrying
                last_error = str(error)
                _warn(attempt, last_error)
                time.sleep(min(2**attempt, 30) + random.random())
                continue
            if response.status_code == 200:
                payload = response.json()
                candidates = payload.get("candidates") or []
                if not candidates:
                    # A safety block or an empty generation. Recorded as an
                    # empty transcript rather than dropped, so the denominator
                    # stays honest.
                    return ""
                parts = candidates[0].get("content", {}).get("parts") or []
                return clean("".join(part.get("text", "") for part in parts))
            if response.status_code in {429, 500, 503, 504}:
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                if is_daily_quota(payload):
                    raise DailyQuotaExhausted(
                        "Gemini free-tier daily request quota is exhausted. "
                        "Rerun the same command tomorrow -- evaluate_manifest resumes "
                        "from the rows already in the output TSV."
                    )
                asked = retry_delay(payload)
                pause = asked if asked is not None else min(2**attempt, 30)
                last_error = f"{response.status_code}, waiting {pause:.0f}s"
                _warn(attempt, last_error)
                time.sleep(pause + random.random())
                continue
            raise RuntimeError(f"Gemini {response.status_code}: {response.text[:400]}")
        raise RuntimeError(f"Gemini failed after {MAX_ATTEMPTS} attempts -- {last_error}")

    def transcribe_batch(audios):
        # The harness hands us a list; the work is network-bound, so a batch is
        # just a concurrency window.
        with httpx.Client(http2=False) as client:
            if len(audios) == 1:
                return [transcribe_one(client, audios[0])]
            with ThreadPoolExecutor(max_workers=min(WORKERS, len(audios))) as pool:
                return list(pool.map(lambda item: transcribe_one(client, item), audios))

    return transcribe_batch


def load(device: str):
    transcribe_batch = load_batch(device)
    return lambda audio: transcribe_batch([audio])[0]


if __name__ == "__main__":
    common.run("gemini", MODEL, "Gemini cloud audio understanding (no dysarthria adaptation)", load)
