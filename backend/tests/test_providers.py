import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.asr.base import ProviderUpstreamError
from app.services.asr.groq import GroqASRProvider
from app.services.asr.runpod import RunpodASRProvider


def test_list_providers_reports_capabilities_without_exposing_secrets() -> None:
    with TestClient(app) as client:
        response = client.get("/v1/providers")

    assert response.status_code == 200
    providers = {provider["name"]: provider for provider in response.json()}
    assert providers["fake"]["configured"] is True
    assert isinstance(providers["runpod"]["configured"], bool)
    assert "api_key" not in providers["runpod"]
    assert providers["runpod"]["capabilities"]["n_best_maximum"] == 5
    assert providers["groq"]["capabilities"]["n_best_maximum"] == 1


def test_rejects_groq_multi_hypothesis_before_upload_or_upstream_call() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/transcriptions",
            data={"provider": "groq", "n_best": "2"},
            files={"file": ("example.wav", b"not-real-audio", "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Requested n_best=2, but provider 'groq' supports at most 1."
    )


@pytest.mark.anyio
async def test_groq_contract_maps_verbose_json_to_normalized_result(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.groq.com/openai/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer groq-test-key"
        return httpx.Response(
            200,
            json={
                "text": "Hello there",
                "segments": [{"start": 0.0, "end": 1.2, "text": "Hello there"}],
            },
        )

    provider = GroqASRProvider(
        api_key="groq-test-key",
        model_id="whisper-large-v3",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )
    result = await provider.transcribe(
        audio_path,
        original_filename="sample.wav",
        language="en",
        n_best=1,
    )

    assert result[0].text == "Hello there"
    assert result[0].score_type == "unavailable"
    assert result[0].segments[0].end_seconds == 1.2


@pytest.mark.anyio
async def test_runpod_contract_maps_worker_output_to_normalized_result(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.runpod.ai/v2/runpod-test/runsync"
        request_payload = json.loads(request.content)
        assert request_payload["input"]["n_best"] == 2
        return httpx.Response(
            200,
            json={
                "output": {
                    "hypotheses": [
                        {
                            "rank": 1,
                            "text": "I want tea",
                            "score": -0.2,
                            "score_type": "sequence_logprob",
                            "segments": [
                                {
                                    "segment_index": 0,
                                    "start_seconds": 0.0,
                                    "end_seconds": 1.0,
                                    "text": "I want tea",
                                }
                            ],
                        }
                    ]
                }
            },
        )

    provider = RunpodASRProvider(
        endpoint_id="runpod-test",
        api_key="runpod-test-key",
        model_id="openai/whisper-large-v3",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )
    result = await provider.transcribe(
        audio_path,
        original_filename="sample.wav",
        language="en",
        n_best=2,
    )

    assert result[0].text == "I want tea"
    assert result[0].score == -0.2
    assert result[0].segments[0].start_seconds == 0.0


@pytest.mark.anyio
async def test_runpod_contract_surfaces_worker_validation_error(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")

    provider = RunpodASRProvider(
        endpoint_id="runpod-test",
        api_key="runpod-test-key",
        model_id="openai/whisper-large-v3",
        timeout_seconds=5,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "output": {
                        "error": {
                            "code": "invalid_input",
                            "message": "Audio could not be decoded.",
                        }
                    }
                },
            )
        ),
    )

    with pytest.raises(ProviderUpstreamError, match="Audio could not be decoded"):
        await provider.transcribe(
            audio_path,
            original_filename="sample.wav",
            language="en",
            n_best=1,
        )
