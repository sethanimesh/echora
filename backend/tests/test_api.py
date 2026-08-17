from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from app import main
from app.audio import encode_wav
from app.messaging.groq_chain import MessageChainResult
from app.schemas import (
    AudioQuality,
    Hypothesis,
    MessageCandidate,
    PersonalBrief,
    PersonaSummary,
    RankerDecision,
    RawAsrResult,
    SpeechAudio,
)


class FakeBackend:
    device = "test/float32"

    async def transcribe(self, audio, beams):
        return RawAsrResult(
            backend="local",
            device=self.device,
            hypotheses=[Hypothesis(id="h1", literal_text="I water", sequence_score=-0.2, search_weight=1)],
            audio_quality=AudioQuality(seconds=0.5, peak_dbfs=-3, rms_dbfs=-9, clipped_samples=0, low_level_warning=False),
            decode_seconds=0.1,
        )


class FailingBackend:
    device = "remote-gpu"

    async def transcribe(self, audio, beams):
        raise TimeoutError("cloud worker is still warming")


class FakeChain:
    def __init__(self, settings):
        pass

    async def run(self, hypotheses, context="general", brief=None):
        return MessageChainResult(
            ranker=RankerDecision(
                decision="selected",
                selected_message_id="m1",
                display_hypothesis_ids=["h1"],
                display_message_ids=["m1"],
                reason="Only candidate",
                source="groq",
            ),
            messages=[
                MessageCandidate(
                    message_id="m1",
                    hypothesis_id="h1",
                    source_hypothesis_ids=["h1"],
                    source_literals=["I water"],
                    literal_text="I water",
                    interpreted_intent="i water",
                    corrected_text="I would like some water.",
                    repair_status="corrected",
                    repair_note="Heard as \u201ci water\u201d.",
                )
            ],
            ranking_seconds=0,
            grammar_seconds=0,
            warnings=[],
        )


def _candidate(message_id: str, text: str) -> MessageCandidate:
    return MessageCandidate(
        message_id=message_id,
        hypothesis_id="h1",
        source_hypothesis_ids=["h1"],
        source_literals=["I water"],
        literal_text="I water",
        interpreted_intent="i water",
        corrected_text=text,
        repair_status="corrected",
        repair_note="Heard as \u201ci water\u201d.",
    )


class AmbiguousChain:
    """Two readings survived, so the speaker still has to choose between them."""

    def __init__(self, settings):
        pass

    async def run(self, hypotheses, context="general", brief=None):
        messages = [_candidate("m1", "I would like some water."), _candidate("m2", "I would like to wait.")]
        return MessageChainResult(
            ranker=RankerDecision(
                decision="ambiguous",
                display_hypothesis_ids=["h1"],
                display_message_ids=["m1", "m2"],
                reason="These could be different messages; please choose.",
                source="groq",
            ),
            messages=messages,
        )


class FakeSpeech:
    spoken: list[str] = []

    def __init__(self, settings):
        pass

    async def synthesize(self, text):
        FakeSpeech.spoken.append(text)
        return SpeechAudio(audio_base64="UklGRg==", voice="hannah", model="test-tts")


class SilentSpeech:
    """Groq TTS could not answer. The message must still be returned."""

    def __init__(self, settings):
        pass

    async def synthesize(self, text):
        return None


class FakePersonal:
    """Stands in for the personal layer so no test loads the real encoder weights."""

    briefs: list[tuple[str, str]] = []
    remembered: list[tuple[str, str, str]] = []

    def __init__(self, settings):
        pass

    def known_ids(self):
        return {"krishnan", "user"}

    def personas(self):
        return [PersonaSummary(id="krishnan", label="Krishnan", history_size=3)]

    async def brief(self, hypotheses, context, profile_id):
        FakePersonal.briefs.append((profile_id, context))
        if profile_id not in self.known_ids():
            return None
        return PersonalBrief(profile_id=profile_id, profile_label="Krishnan", speaker_note="note")

    async def remember(self, profile_id, context, heard, message):
        FakePersonal.remembered.append((profile_id, heard, message))
        return True, False, "stored"


def client(monkeypatch, chain=FakeChain, speech=FakeSpeech, personal=FakePersonal):
    FakeSpeech.spoken = []
    FakePersonal.briefs = []
    FakePersonal.remembered = []
    monkeypatch.setattr(main, "create_backend", lambda settings: FakeBackend())
    monkeypatch.setattr(main, "GroqMessageChain", chain)
    monkeypatch.setattr(main, "GroqSpeech", speech)
    monkeypatch.setattr(main, "Personalizer", personal)
    return TestClient(main.app)


def _post(api, context: str | None = None, persona: str | None = None):
    audio = np.sin(np.linspace(0, 100, 8_000)).astype(np.float32) * 0.1
    form = {}
    if context:
        form["context"] = context
    if persona:
        form["persona"] = persona
    return api.post(
        "/api/v1/transcriptions",
        files={"audio": ("voice.wav", encode_wav(audio), "audio/wav")},
        data=form or None,
    )


def test_health_and_transcription_contract(monkeypatch) -> None:
    with client(monkeypatch) as api:
        assert api.get("/api/v1/health").status_code == 200
        audio = np.sin(np.linspace(0, 100, 8_000)).astype(np.float32) * 0.1
        response = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", encode_wav(audio), "audio/wav")},
            data={"context": "home"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["hypotheses"][0]["literal_text"] == "I water"
        assert body["messages"][0]["literal_text"] == "I water"
        assert body["messages"][0]["corrected_text"] == "I would like some water."
        assert body["context"] == "home"
        assert body["recommended_message_id"] == "m1"
        assert body["beam_weights_are_calibrated_confidence"] is False
        assert body["user_confirmation_required"] is False


def test_invalid_context_is_rejected(monkeypatch) -> None:
    with client(monkeypatch) as api:
        response = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", encode_wav(np.ones(8_000, dtype=np.float32) * 0.1), "audio/wav")},
            data={"context": "spaceship"},
        )
        assert response.status_code == 422


def test_invalid_and_oversized_audio(monkeypatch) -> None:
    with client(monkeypatch) as api:
        invalid = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.xyz", b"bad", "application/octet-stream")},
        )
        assert invalid.status_code == 422
        main.app.state.settings = SimpleNamespace(
            **{**main.app.state.settings.__dict__, "max_upload_bytes": 3}
        )
        oversized = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", b"1234", "audio/wav")},
        )
        assert oversized.status_code == 413


def test_silence_and_duration_are_rejected(monkeypatch) -> None:
    with client(monkeypatch) as api:
        silence = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("silent.wav", encode_wav(np.zeros(8_000, dtype=np.float32)), "audio/wav")},
        )
        assert silence.status_code == 422
        main.app.state.settings = SimpleNamespace(
            **{**main.app.state.settings.__dict__, "max_audio_seconds": 0.2}
        )
        long_recording = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("long.wav", encode_wav(np.ones(8_000, dtype=np.float32) * 0.1), "audio/wav")},
        )
        assert long_recording.status_code == 422


def test_model_and_cloud_worker_unavailable_are_explicit(monkeypatch) -> None:
    monkeypatch.setattr(main, "create_backend", lambda settings: (_ for _ in ()).throw(RuntimeError("missing model")))
    monkeypatch.setattr(main, "GroqMessageChain", FakeChain)
    with TestClient(main.app) as api:
        assert api.get("/api/v1/health").json()["model_ready"] is False
        response = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", encode_wav(np.ones(8_000, dtype=np.float32) * 0.1), "audio/wav")},
        )
        assert response.status_code == 503
        assert "missing model" in response.json()["detail"]

    monkeypatch.setattr(main, "create_backend", lambda settings: FailingBackend())
    with TestClient(main.app) as api:
        response = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", encode_wav(np.ones(8_000, dtype=np.float32) * 0.1), "audio/wav")},
        )
        assert response.status_code == 503
        assert "TimeoutError" in response.json()["detail"]


def test_one_surviving_message_is_synthesized_with_the_result(monkeypatch) -> None:
    """An unambiguous message ships its own audio, so the browser can speak it at once."""
    with client(monkeypatch) as api:
        body = _post(api).json()
        assert body["ranker"]["decision"] == "selected"
        assert body["speech"]["audio_base64"] == "UklGRg=="
        assert body["speech"]["media_type"] == "audio/wav"
        # The realized message is spoken, never the raw literal.
        assert FakeSpeech.spoken == ["I would like some water."]


def test_an_ambiguous_result_is_never_spoken_for_the_speaker(monkeypatch) -> None:
    with client(monkeypatch, chain=AmbiguousChain) as api:
        body = _post(api).json()
        assert body["needs_user_choice"] is True
        assert body["speech"] is None
        assert FakeSpeech.spoken == []


def test_failed_synthesis_still_returns_the_message(monkeypatch) -> None:
    with client(monkeypatch, speech=SilentSpeech) as api:
        response = _post(api)
        assert response.status_code == 200
        body = response.json()
        assert body["speech"] is None
        assert body["messages"][0]["corrected_text"] == "I would like some water."


def test_autoplay_can_be_switched_off(monkeypatch) -> None:
    with client(monkeypatch) as api:
        main.app.state.settings = SimpleNamespace(
            **{**main.app.state.settings.__dict__, "speech_autoplay": False}
        )
        body = _post(api).json()
        assert body["speech"] is None
        assert FakeSpeech.spoken == []


def test_the_speech_endpoint_serves_audio_and_reports_unavailability(monkeypatch) -> None:
    with client(monkeypatch) as api:
        ok = api.post("/api/v1/speech", json={"text": "My leg hurts."})
        assert ok.status_code == 200
        assert ok.json()["audio_base64"] == "UklGRg=="
        assert FakeSpeech.spoken == ["My leg hurts."]

    # A 503 is the signal for the interface to fall back to the browser voice.
    with client(monkeypatch, speech=SilentSpeech) as api:
        assert api.post("/api/v1/speech", json={"text": "My leg hurts."}).status_code == 503


# ---------------------------------------------------------------- personal context


def test_a_profile_reaches_the_message_chain(monkeypatch) -> None:
    with client(monkeypatch) as api:
        response = _post(api, context="home", persona="krishnan")
    assert response.status_code == 200
    assert response.json()["persona"] == "krishnan"
    assert FakePersonal.briefs == [("krishnan", "home")]


def test_a_request_without_a_profile_never_asks_the_personal_layer(monkeypatch) -> None:
    with client(monkeypatch) as api:
        response = _post(api)
    assert response.status_code == 200
    assert response.json()["persona"] is None
    assert FakePersonal.briefs == []


def test_an_unknown_profile_is_a_warning_and_still_speaks(monkeypatch) -> None:
    """A stale value in the interface must never stand between a speaker and their message."""
    with client(monkeypatch) as api:
        response = _post(api, persona="nobody")
    body = response.json()
    assert response.status_code == 200
    assert body["messages"][0]["corrected_text"] == "I would like some water."
    assert any("nobody" in warning for warning in body["warnings"])


def test_the_personal_layer_failing_to_load_leaves_transcription_working(monkeypatch) -> None:
    class Broken:
        def __init__(self, settings):
            raise RuntimeError("no embedding bundle")

    with client(monkeypatch, personal=Broken) as api:
        assert api.get("/api/v1/health").json()["personal_ready"] is False
        assert api.get("/api/v1/personas").json() == []
        response = _post(api, persona="krishnan")
    assert response.status_code == 200
    assert response.json()["messages"][0]["corrected_text"] == "I would like some water."


def test_an_accepted_message_is_handed_to_the_store(monkeypatch) -> None:
    with client(monkeypatch) as api:
        response = api.post(
            "/api/v1/accepted",
            json={"persona": "krishnan", "context": "home", "heard": "water", "message": "I would like some water."},
        )
    assert response.json() == {"stored": True, "merged": False, "reason": "stored"}
    assert FakePersonal.remembered == [("krishnan", "water", "I would like some water.")]


def test_remembering_is_never_an_error_when_the_layer_is_down(monkeypatch) -> None:
    class Broken:
        def __init__(self, settings):
            raise RuntimeError("no embedding bundle")

    with client(monkeypatch, personal=Broken) as api:
        response = api.post(
            "/api/v1/accepted",
            json={"persona": "krishnan", "heard": "water", "message": "I would like some water."},
        )
    assert response.status_code == 200
    assert response.json()["stored"] is False
