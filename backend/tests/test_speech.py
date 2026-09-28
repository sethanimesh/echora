"""Groq speech synthesis: what reaches the model, and what happens when it fails."""

from __future__ import annotations

import base64

import pytest

from app.config import load_settings
from app.messaging.speech import MAX_SPEECH_CHARS, GroqSpeech, _speakable


class RecordingClient:
    def __init__(self, audio: bytes = b"RIFFwav") -> None:
        self.calls: list[dict] = []
        self.audio = audio

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleBinaryResponse(self.audio)


class SimpleBinaryResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload


class ExplodingClient:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise self.error


def _speech(**overrides) -> GroqSpeech:
    settings = load_settings()
    settings = settings.__class__(**{**settings.__dict__, "groq_api_key": "test-key", **overrides})
    return GroqSpeech(settings)


def _attach(speech: GroqSpeech, client) -> None:
    """Stand in for the `client.audio.speech` chain the SDK exposes."""

    class Audio:
        speech = client

    class Client:
        audio = Audio()

    speech.client = Client()


def test_a_vocal_direction_is_never_left_in_the_text() -> None:
    # Orpheus reads "[cheerful]" as an instruction rather than speaking it, so a
    # bracket that survived from ASR would silently change how the voice sounds.
    assert _speakable("Please [cheerful] help me.") == ""
    assert _speakable("  I  want   water. ") == "I want water."


def test_long_text_uses_full_text_device_fallback() -> None:
    spoken = _speakable("word " * 80)
    assert spoken == ""


@pytest.mark.parametrize("text", ["", "   ", "[only a direction]"])
async def test_nothing_speakable_is_never_sent(text: str) -> None:
    speech = _speech()
    client = RecordingClient()
    _attach(speech, client)
    assert await speech.synthesize(text) is None
    assert client.calls == []


async def test_an_unconfigured_key_falls_back_instead_of_failing() -> None:
    speech = _speech(groq_api_key=None)
    assert speech.client is None
    assert await speech.synthesize("My leg hurts.") is None


async def test_the_configured_model_and_voice_reach_the_request() -> None:
    speech = _speech(groq_tts_model="test/model", groq_tts_voice="troy")
    client = RecordingClient(b"RIFFwav")
    _attach(speech, client)
    audio = await speech.synthesize("My leg hurts.")
    assert audio is not None
    assert base64.b64decode(audio.audio_base64) == b"RIFFwav"
    assert audio.voice == "troy"
    assert audio.model == "test/model"
    assert client.calls[0]["input"] == "My leg hurts."
    assert client.calls[0]["response_format"] == "wav"


async def test_a_rate_limit_hands_over_without_retrying() -> None:
    # Waiting out a throttle would leave the speaker in silence, and a daily cap
    # will not clear inside this request. One attempt, then the browser voice.
    speech = _speech()
    client = ExplodingClient(RuntimeError("429 rate limit reached for tokens per day"))
    _attach(speech, client)
    assert await speech.synthesize("My leg hurts.") is None
    assert client.calls == 1
