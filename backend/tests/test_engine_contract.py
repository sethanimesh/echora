from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from app.asr import backends
from app.asr.engine import QwenCommandEngine, build_hypotheses, normalize, select_device, sha256
from app.config import load_settings
from app.schemas import AudioQuality, Hypothesis, RawAsrResult


def test_normalize_and_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"echora")
    assert normalize("I want water, please.") == "i want water please"
    assert sha256(artifact) == "b9515efa129ca3881bbaddac145d3d7c4222264db1138ae025d812d4c8bf8e17"


def test_cpu_must_be_explicit() -> None:
    device, dtype = select_device("cpu")
    assert device == "cpu"
    assert str(dtype) == "torch.float32"
    with pytest.raises(ValueError):
        select_device("magic")


def test_device_selection_never_silently_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No supported accelerator"):
        select_device("auto")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        select_device("cuda")
    with pytest.raises(RuntimeError, match="Apple MPS is unavailable"):
        select_device("mps")


def test_five_beams_are_deduplicated_without_rewriting_literals() -> None:
    result = build_hypotheses(
        ["I water", "I water.", " high water ", "", "I want water"],
        [-0.1, -0.2, -0.3, -0.4, -0.5],
    )
    assert [item.literal_text for item in result] == ["I water", "high water", "I want water"]
    assert [item.id for item in result] == ["h1", "h2", "h3"]
    assert abs(sum(item.search_weight for item in result) - 1.0) < 1e-5


def test_incorrect_adapter_format_is_rejected_before_model_load(tmp_path: Path) -> None:
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    (foundation / "model.safetensors").write_bytes(b"placeholder")
    adapter = tmp_path / "adapter.safetensors"
    save_file({"placeholder": torch.zeros(1)}, adapter, metadata={"format": "wrong"})
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not an Echora command-v3 artifact"):
        QwenCommandEngine(foundation, adapter, config, device_request="cpu", verify_hashes=False)


@pytest.mark.asyncio
async def test_local_backend_serializes_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEngine:
        active = 0
        peak = 0
        device = "cpu"
        dtype = "float32"

        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, beams, backend):
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            time.sleep(0.03)
            type(self).active -= 1
            return RawAsrResult(
                backend="local",
                device="cpu/float32",
                hypotheses=[Hypothesis(id="h1", literal_text="water", sequence_score=-1, search_weight=1)],
                audio_quality=AudioQuality(seconds=1, peak_dbfs=-2, rms_dbfs=-8, clipped_samples=0, low_level_warning=False),
                decode_seconds=0.03,
            )

    monkeypatch.setattr(backends, "QwenCommandEngine", FakeEngine)
    backend = backends.LocalAsrBackend(load_settings())
    await asyncio.gather(
        backend.transcribe(np.ones(160, dtype=np.float32), 5),
        backend.transcribe(np.ones(160, dtype=np.float32), 5),
    )
    assert FakeEngine.peak == 1


def test_remote_payload_is_small_wav() -> None:
    payload = backends._RemoteBackend.payload(np.ones(16_000, dtype=np.float32) * 0.1, 5)
    assert payload["filename"] == "normalized.wav"
    assert payload["beams"] == 5
    assert len(payload["audio_base64"]) < 100_000


def test_remote_backends_require_explicit_configuration() -> None:
    settings = load_settings()
    pod_settings = settings.__class__(
        **{**settings.__dict__, "backend": "pod", "pod_url": None, "pod_token": None}
    )
    with pytest.raises(RuntimeError, match="ECHORA_POD_URL"):
        backends.PodAsrBackend(pod_settings)
    runpod_settings = settings.__class__(
        **{
            **settings.__dict__,
            "backend": "runpod",
            "runpod_endpoint_id": None,
            "runpod_api_key": None,
        }
    )
    with pytest.raises(RuntimeError, match="RUNPOD_ENDPOINT_ID"):
        backends.RunpodAsrBackend(runpod_settings)
