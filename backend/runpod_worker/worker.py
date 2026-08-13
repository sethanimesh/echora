"""Runpod handler for pinned Hugging Face Transformers Whisper Large-v3 inference.

This worker deliberately uses Hugging Face Transformers and PyTorch. It does
not use faster-whisper, automatic speech endpointing, denoising, or VAD: input
audio is only decoded and resampled after the caller has completed recording.
"""

from __future__ import annotations

import base64
import binascii
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "openai/whisper-large-v3"
HF_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 120.0
SAMPLE_RATE = 16_000
TIMESTAMP_TOKEN = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")
SPECIAL_TOKEN = re.compile(r"<\|[^|]+\|>")


class WorkerInputError(ValueError):
    """An invalid request that should be returned to the caller without retrying."""


def resolve_cached_snapshot_path(
    model_id: str,
    *,
    cache_root: Path = HF_CACHE_ROOT,
) -> tuple[Path, str]:
    """Resolve the immutable snapshot selected by Runpod's managed model cache."""

    if "/" not in model_id:
        raise RuntimeError(f"Invalid Hugging Face model ID: {model_id}.")
    organization, model_name = model_id.split("/", maxsplit=1)
    model_root = cache_root / f"models--{organization}--{model_name}"
    snapshots_directory = model_root / "snapshots"
    main_ref = model_root / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text().strip()
        snapshot_path = snapshots_directory / revision
        if snapshot_path.is_dir():
            return snapshot_path, revision

    snapshots = sorted(path for path in snapshots_directory.glob("*") if path.is_dir())
    if snapshots:
        snapshot_path = snapshots[0]
        return snapshot_path, snapshot_path.name
    raise RuntimeError(
        "Whisper was not found in Runpod's cached model mount. "
        f"Expected model root: {model_root}. Configure cached model '{model_id}' and redeploy."
    )


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable model settings for this deployed worker image."""

    model_id: str = MODEL_ID
    max_audio_bytes: int = MAX_AUDIO_BYTES
    max_duration_seconds: float = MAX_DURATION_SECONDS


def validate_input(payload: dict[str, Any], config: WorkerConfig) -> tuple[bytes, str, str, int]:
    """Validate and decode the transport payload before allocating GPU work."""

    if payload.get("model_id") != config.model_id:
        raise WorkerInputError(f"This worker only serves {config.model_id}.")
    if payload.get("language") != "en":
        raise WorkerInputError("This worker currently supports only English (en).")

    n_best = payload.get("n_best", 1)
    if not isinstance(n_best, int) or not 1 <= n_best <= 5:
        raise WorkerInputError("n_best must be an integer from 1 to 5.")

    encoded_audio = payload.get("audio_base64")
    if not isinstance(encoded_audio, str):
        raise WorkerInputError("audio_base64 is required.")
    try:
        audio_bytes = base64.b64decode(encoded_audio, validate=True)
    except (binascii.Error, ValueError) as error:
        raise WorkerInputError("audio_base64 is not valid base64.") from error
    if not audio_bytes:
        raise WorkerInputError("Audio must not be empty.")
    if len(audio_bytes) > config.max_audio_bytes:
        raise WorkerInputError(f"Audio exceeds the {config.max_audio_bytes}-byte limit.")

    suffix = Path(str(payload.get("filename") or "audio.wav")).suffix.lower()
    if suffix not in {".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm"}:
        raise WorkerInputError("Unsupported audio file extension.")
    return audio_bytes, suffix, "en", n_best


def segments_from_timestamped_text(timestamped_text: str) -> list[dict[str, Any]]:
    """Convert Whisper timestamp tokens into text spans without inventing alignment."""

    matches = list(TIMESTAMP_TOKEN.finditer(timestamped_text))
    segments: list[dict[str, Any]] = []
    for index, match in enumerate(matches[:-1]):
        next_match = matches[index + 1]
        text = SPECIAL_TOKEN.sub("", timestamped_text[match.end() : next_match.start()]).strip()
        start_seconds = float(match.group(1))
        end_seconds = float(next_match.group(1))
        if text and end_seconds >= start_seconds:
            segments.append(
                {
                    "segment_index": len(segments),
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "text": text,
                }
            )
    return segments


def get_output_value(output: Any, key: str, default: Any = None) -> Any:
    """Read an output value from either Transformers ModelOutput or a dictionary."""

    if isinstance(output, dict):
        return output.get(key, default)
    return getattr(output, key, default)


def to_python(value: Any) -> Any:
    """Convert tensors and tensor-like objects to ordinary Python collections."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class WhisperRuntime:
    """Loads the pinned GPU model once and normalizes its raw generation output."""

    def __init__(self, config: WorkerConfig | None = None) -> None:
        self.config = config or WorkerConfig()
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.model_revision: str | None = None

    def load(self) -> None:
        """Load the exact cached Transformers checkpoint during worker startup."""

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for this Runpod worker.")
        model_path, self.model_revision = resolve_cached_snapshot_path(self.config.model_id)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda")
        self.model.eval()

    def decode_audio(self, audio_bytes: bytes, suffix: str) -> tuple[Any, float]:
        """Decode to mono 16 kHz PCM without removing pauses or changing source audio."""

        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="echora-asr-") as temporary_directory:
            source_path = Path(temporary_directory) / f"source{suffix}"
            pcm_path = Path(temporary_directory) / "decoded.wav"
            source_path.write_bytes(audio_bytes)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(source_path),
                        "-ar",
                        str(SAMPLE_RATE),
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        str(pcm_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                raise WorkerInputError("Audio could not be decoded.") from error
            samples, sample_rate = sf.read(pcm_path, dtype="float32", always_2d=False)

        duration_seconds = len(samples) / sample_rate
        if duration_seconds <= 0:
            raise WorkerInputError("Audio contains no decodable samples.")
        if duration_seconds > self.config.max_duration_seconds:
            raise WorkerInputError(
                f"Audio exceeds the {self.config.max_duration_seconds:g}-second duration limit."
            )
        return samples, duration_seconds

    def transcribe(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate up to five raw beam-search hypotheses and timestamp spans."""

        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("WhisperRuntime.load() must run before transcribe().")
        audio_bytes, suffix, language, n_best = validate_input(payload, self.config)
        samples, duration_seconds = self.decode_audio(audio_bytes, suffix)
        model_inputs = self.processor(
            samples,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,
            return_attention_mask=True,
        ).to("cuda", self.torch.float16)

        generation_kwargs: dict[str, Any] = {
            "language": language,
            "task": "transcribe",
            "num_beams": n_best,
            "num_return_sequences": n_best,
            "max_new_tokens": 448,
            "condition_on_prev_tokens": False,
            "return_timestamps": True,
            "return_dict_in_generate": True,
        }
        if duration_seconds > 30:
            generation_kwargs["return_segments"] = True

        with self.torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generation_kwargs)

        sequences = get_output_value(generated, "sequences", generated)
        sequence_list = to_python(sequences)
        texts = self.processor.batch_decode(sequences, skip_special_tokens=True)
        raw_scores = to_python(get_output_value(generated, "sequences_scores", []))
        scores = raw_scores if isinstance(raw_scores, list) else []

        hypotheses = []
        for index, (sequence, text) in enumerate(zip(sequence_list, texts, strict=True)):
            timestamped_text = self.processor.tokenizer.decode(
                sequence,
                skip_special_tokens=False,
                decode_with_timestamps=True,
            )
            score = float(scores[index]) if index < len(scores) else 0.0
            hypotheses.append(
                {
                    "rank": index + 1,
                    "text": text.strip(),
                    "score": score,
                    "score_type": "sequence_logprob" if index < len(scores) else "unavailable",
                    "segments": segments_from_timestamped_text(timestamped_text),
                }
            )

        return {
            "model_id": self.config.model_id,
            "model_revision": self.model_revision,
            "duration_seconds": duration_seconds,
            "hypotheses": hypotheses,
        }


runtime = WhisperRuntime()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Job entry point with user-safe input errors and retryable unexpected failures."""

    try:
        return runtime.transcribe(job.get("input", {}))
    except WorkerInputError as error:
        return {"error": {"code": "invalid_input", "message": str(error)}}
