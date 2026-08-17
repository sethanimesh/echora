"""Official Qwen3-ASR 1.7B foundation checkpoint.

This runner deliberately refuses to download a model implicitly. Echora's
weights live on the external SSD locally; cloud jobs must point the same
environment variable at an explicitly staged copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

import common


MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
EXPECTED_WEIGHT_BYTES = 4_076_193_080
DEFAULT_MODEL_PATH = Path(
    "/Volumes/Extreme Pro/echora/checkpoints/base-models/Qwen3-ASR-1.7B-hf"
)


def model_source() -> Path:
    source = Path(os.environ.get("ECHORA_QWEN3_ASR_MODEL", DEFAULT_MODEL_PATH))
    weights = source / "model.safetensors"
    if not source.is_dir() or not weights.is_file():
        raise FileNotFoundError(
            f"Qwen foundation weights are not staged at {source}. "
            "Set ECHORA_QWEN3_ASR_MODEL to the exact local checkpoint directory."
        )
    if weights.stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"Incomplete Qwen weight file at {weights}: "
            f"expected {EXPECTED_WEIGHT_BYTES} bytes, got {weights.stat().st_size}."
        )
    return source


def inference_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_batch(device: str):
    source = model_source()
    dtype = inference_dtype(device)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        source,
        dtype=dtype,
        local_files_only=True,
    )
    model.to(device).eval()

    def transcribe_batch(audios):
        # TORGO and the external clips are known-English evaluation data. We
        # force only the language, never vocabulary, context, or expected text.
        inputs = processor.apply_transcription_request(
            audio=audios,
            language=["English"] * len(audios),
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return [
            text.strip()
            for text in processor.decode(
                generated_ids,
                return_format="transcription_only",
            )
        ]

    return transcribe_batch


def load(device: str):
    transcribe_batch = load_batch(device)

    def transcribe(audio):
        return transcribe_batch([audio])[0]

    return transcribe


if __name__ == "__main__":
    common.run(
        "qwen3_asr",
        MODEL,
        "Qwen3-ASR 1.7B, official foundation weights (no dysarthria adaptation)",
        load,
    )
