"""Stock nvidia/parakeet-ctc-1.1b literal-ASR baseline.

CTC is useful as a deliberately different decoder from TDT: it is monotonic,
does not autoregressively rewrite earlier words, and often exposes acoustic
confusions more directly.  This script does not use a language model or the
Echora candidate ranker.
"""

import torch
from transformers import ParakeetForCTC, ParakeetProcessor

import common

MODEL = "nvidia/parakeet-ctc-1.1b"


def load_batch(device):
    processor = ParakeetProcessor.from_pretrained(MODEL)
    # The checkpoint advertises bfloat16, while the processor emits float32.
    # Force one dtype for CPU/MPS; otherwise the first convolution rejects it.
    model = ParakeetForCTC.from_pretrained(MODEL, dtype=torch.float32)
    model.to(device).eval()

    def transcribe_batch(audios):
        inputs = processor(
            audios,
            sampling_rate=common.SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            token_ids = model(**inputs).logits.argmax(dim=-1)
        return [
            text.strip()
            for text in processor.batch_decode(token_ids, skip_special_tokens=True)
        ]

    return transcribe_batch


def load(device):
    transcribe_batch = load_batch(device)

    def transcribe(audio):
        return transcribe_batch([audio])[0]

    return transcribe


if __name__ == "__main__":
    common.run("parakeet_ctc", MODEL, "Parakeet-CTC 1.1B, stock (no TORGO adaptation)", load)
