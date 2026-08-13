import base64
from pathlib import Path

import pytest

from runpod_worker.worker import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    WorkerConfig,
    WorkerInputError,
    resolve_cached_snapshot_path,
    segments_from_timestamped_text,
    validate_input,
)


def test_generation_limit_leaves_room_for_whisper_decoder_start_tokens() -> None:
    assert MAX_NEW_TOKENS == 445
    assert MAX_NEW_TOKENS + 3 == 448


def test_validate_input_accepts_a_small_english_wav_request() -> None:
    payload = {
        "audio_base64": base64.b64encode(b"audio").decode("ascii"),
        "filename": "utterance.wav",
        "language": "en",
        "model_id": "openai/whisper-large-v3",
        "n_best": 5,
    }

    audio_bytes, suffix, language, n_best = validate_input(payload, WorkerConfig())

    assert audio_bytes == b"audio"
    assert suffix == ".wav"
    assert language == "en"
    assert n_best == 5


def test_validate_input_rejects_unknown_language() -> None:
    payload = {
        "audio_base64": base64.b64encode(b"audio").decode("ascii"),
        "filename": "utterance.wav",
        "language": "hi",
        "model_id": "openai/whisper-large-v3",
        "n_best": 1,
    }

    with pytest.raises(WorkerInputError, match="English"):
        validate_input(payload, WorkerConfig())


def test_timestamp_tokens_become_non_invented_segments() -> None:
    segments = segments_from_timestamped_text("<|0.00|> Hello<|0.52|> there<|1.00|>")

    assert segments == [
        {
            "segment_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 0.52,
            "text": "Hello",
        },
        {
            "segment_index": 1,
            "start_seconds": 0.52,
            "end_seconds": 1.0,
            "text": "there",
        },
    ]


def test_resolve_cached_snapshot_path_uses_runpod_main_reference(tmp_path) -> None:
    model_root = tmp_path / "models--openai--whisper-large-v3"
    revision = "cache-revision"
    snapshot_path = model_root / "snapshots" / revision
    snapshot_path.mkdir(parents=True)
    (model_root / "refs").mkdir()
    (model_root / "refs" / "main").write_text(revision)

    resolved, resolved_revision = resolve_cached_snapshot_path(MODEL_ID, cache_root=tmp_path)

    assert resolved == snapshot_path
    assert resolved_revision == revision


def test_resolve_cached_snapshot_path_falls_back_for_older_cache_layouts(tmp_path) -> None:
    snapshot_path = tmp_path / "models--openai--whisper-large-v3" / "snapshots" / "old-cache"
    snapshot_path.mkdir(parents=True)

    resolved, resolved_revision = resolve_cached_snapshot_path(MODEL_ID, cache_root=tmp_path)

    assert resolved == snapshot_path
    assert resolved_revision == "old-cache"


def test_resolve_cached_snapshot_path_fails_when_runpod_cache_is_missing(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="cached model mount"):
        resolve_cached_snapshot_path(MODEL_ID, cache_root=tmp_path)


def test_dockerfile_uses_an_adjacent_serverless_entrypoint() -> None:
    worker_directory = Path(__file__).parents[1] / "runpod_worker"

    assert 'runpod.serverless.start({"handler": handler})' in (
        worker_directory / "handler.py"
    ).read_text()
    assert "Loaded model {runtime.config.model_id}" in (worker_directory / "handler.py").read_text()
    assert 'CMD ["python", "-u", "handler.py"]' in (
        worker_directory / "Dockerfile"
    ).read_text()
    assert "FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime" in (
        worker_directory / "Dockerfile"
    ).read_text()
