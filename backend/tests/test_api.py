from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from app import main
from app.audio import encode_wav
from app.messaging.groq_chain import MessageChainResult
from app.schemas import AudioQuality, Hypothesis, MessageCandidate, RankerDecision, RawAsrResult


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

    async def run(self, hypotheses, context="general"):
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


def client(monkeypatch):
    monkeypatch.setattr(main, "create_backend", lambda settings: FakeBackend())
    monkeypatch.setattr(main, "GroqMessageChain", FakeChain)
    return TestClient(main.app)


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
        assert body["user_confirmation_required"] is True


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
