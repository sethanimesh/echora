"""Groq speech synthesis for a message the speaker is about to communicate.

Speech is an enhancement, never a dependency. Every failure here -- no API key,
a rate limit, a network fault -- returns None so the caller falls back to the
browser's local voice. A stroke survivor waiting to be heard is worse off with
an error than with a plainer voice.
"""

from __future__ import annotations

import asyncio
import base64
import re

from groq import Groq

from ..config import Settings
from ..schemas import SpeechAudio


# Orpheus reads bracketed text as a vocal direction ("[cheerful]") rather than
# speaking it, and rejects input beyond this length.
MAX_SPEECH_CHARS = 200
# Synthesis runs inside the transcription request, so a stalled TTS call must not
# hold the message back. The browser voice takes over well before this elapses.
SPEECH_TIMEOUT_SECONDS = 12.0
_DIRECTION = re.compile(r"\[[^\]]*\]")


def _speakable(text: str) -> str:
    """Strip anything the model would swallow as a direction, then bound length."""
    cleaned = " ".join(_DIRECTION.sub(" ", text).split())
    if len(cleaned) <= MAX_SPEECH_CHARS:
        return cleaned
    # Cut on a word boundary so the voice does not stop mid-word.
    head = cleaned[:MAX_SPEECH_CHARS]
    spaced, _, _ = head.rpartition(" ")
    return (spaced or head).rstrip()


class GroqSpeech:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None

    async def synthesize(self, text: str) -> SpeechAudio | None:
        spoken = _speakable(text or "")
        if not self.client or not spoken:
            return None
        try:
            audio = await asyncio.to_thread(self._create, spoken)
        except Exception:
            # Deliberately not retried. A per-minute throttle would make the
            # speaker wait and a daily cap will not clear inside this request, so
            # both cases hand straight over to the local browser voice.
            return None
        return SpeechAudio(
            audio_base64=base64.b64encode(audio).decode("ascii"),
            media_type="audio/wav",
            voice=self.settings.groq_tts_voice,
            model=self.settings.groq_tts_model,
        )

    def _create(self, text: str) -> bytes:
        response = self.client.audio.speech.create(
            model=self.settings.groq_tts_model,
            voice=self.settings.groq_tts_voice,
            input=text,
            response_format="wav",
            timeout=SPEECH_TIMEOUT_SECONDS,
        )
        return response.read()
