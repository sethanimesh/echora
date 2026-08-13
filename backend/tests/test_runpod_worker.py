import base64

import pytest

from runpod_worker.worker import (
    WorkerConfig,
    WorkerInputError,
    segments_from_timestamped_text,
    validate_input,
)


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
