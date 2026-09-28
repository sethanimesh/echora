"""The deployed Echora recognizer: Qwen3-ASR 1.7B plus the command-v3 adapter.

The other runners here score foundations. This one scores what the application
actually ships, by driving `backend.app.asr.engine` rather than a copy of it,
so a benchmark number and a production transcript come from one code path.

It emits all five beams -- rank 1 into `transcript`, the rest into
`alternatives` -- because top-1 WER is not the whole claim Echora makes. The
beams are the evidence the message chain reasons over, so `metrics.py`'s
top-k oracle column is the number that describes what the speaker can reach.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import common


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL = "echora-qwen3-asr-command-v3"
BEAMS = int(os.environ.get("ECHORA_BENCH_BEAMS", "5"))


def load_batch(device: str):
    from backend.app.asr.engine import QwenCommandEngine
    from backend.app.config import load_settings

    settings = load_settings()
    engine = QwenCommandEngine(
        settings.foundation,
        settings.adapter,
        settings.inference_config,
        device if device != "cpu" else settings.device,
    )

    def transcribe_batch(audios):
        # Beam search decodes one utterance at a time here, matching how the
        # application calls it; batching would change the numbers being compared.
        predictions = []
        for audio in audios:
            result = engine.transcribe(audio, BEAMS, "local")
            predictions.append(
                common.Prediction(
                    tuple(
                        common.Hypothesis(item.literal_text, item.sequence_score)
                        for item in result.hypotheses
                    )
                )
            )
        return predictions

    return transcribe_batch


def load(device: str):
    transcribe_batch = load_batch(device)
    return lambda audio: transcribe_batch([audio])[0]


if __name__ == "__main__":
    common.run("echora_v3", MODEL, "Qwen3-ASR 1.7B + command-v3 adapter (deployed)", load)
