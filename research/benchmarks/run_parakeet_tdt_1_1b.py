"""Official Parakeet-TDT 1.1B foundation checkpoint through NVIDIA NeMo."""

from __future__ import annotations

import os
from pathlib import Path

import torch

import common


MODEL = "nvidia/parakeet-tdt-1.1b"
DEFAULT_MODEL_PATH = Path(
    "/Volumes/Extreme Pro/echora/checkpoints/base-models/"
    "nvidia-parakeet-tdt-1.1b/parakeet-tdt-1.1b.nemo"
)


def model_source() -> Path:
    source = Path(os.environ.get("ECHORA_PARAKEET_TDT_1_1B", DEFAULT_MODEL_PATH))
    if not source.is_file():
        raise FileNotFoundError(
            f"Parakeet foundation weights are not staged at {source}. "
            "Set ECHORA_PARAKEET_TDT_1_1B to the exact .nemo checkpoint."
        )
    return source


def load_batch(device: str):
    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError as error:
        raise RuntimeError(
            "Parakeet-TDT 1.1B requires the cloud NeMo environment; "
            "install nemo_toolkit[asr] there before running this benchmark."
        ) from error

    model = ASRModel.restore_from(restore_path=str(model_source()), map_location=device)
    model.to(device).eval()

    def transcribe_batch(audios):
        with torch.inference_mode():
            outputs = model.transcribe(
                audio=audios,
                batch_size=len(audios),
                return_hypotheses=True,
            )
        return [
            (item.text if hasattr(item, "text") else str(item)).strip()
            for item in outputs
        ]

    return transcribe_batch


def load(device: str):
    transcribe_batch = load_batch(device)

    def transcribe(audio):
        return transcribe_batch([audio])[0]

    return transcribe


if __name__ == "__main__":
    common.run(
        "parakeet_tdt_1_1b",
        MODEL,
        "Parakeet-TDT 1.1B, official foundation weights (no dysarthria adaptation)",
        load,
    )
