"""Tests for the manual microphone recorder's API request contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record_and_transcribe.py"
SPEC = importlib.util.spec_from_file_location("record_and_transcribe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RECORDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECORDER
SPEC.loader.exec_module(RECORDER)


def test_upload_recording_uses_echora_multipart_contract(
    tmp_path: Path, monkeypatch
) -> None:
    recording_path = tmp_path / "recording.wav"
    recording_path.write_bytes(b"not-real-wav")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content
        return httpx.Response(201, json={"id": "recording-id", "provider": "runpod"})

    transport = httpx.MockTransport(handler)

    def mocked_post(*args, **kwargs):
        with httpx.Client(transport=transport) as client:
            return client.post(*args, **kwargs)

    monkeypatch.setattr(httpx, "post", mocked_post)

    response = RECORDER.upload_recording(
        recording_path,
        api_url="http://127.0.0.1:8000/",
        provider="runpod",
        n_best=5,
    )

    assert response["id"] == "recording-id"
    assert captured["url"] == "http://127.0.0.1:8000/v1/transcriptions"
    body = bytes(captured["body"])
    assert b'name="provider"' in body
    assert b"runpod" in body
    assert b'name="n_best"' in body
    assert b"recording.wav" in body
