"""nvidia/parakeet-tdt-0.6b-v3

NOT a TORGO fine-tune. This is the stock NVIDIA model, here because plan.md
names Parakeet-TDT the leading fine-tuning candidate and you should see what the
un-adapted starting point does on your voice before adapting anything.

Two deliberate substitutions from plan.md:

* 0.6b-v3 rather than tdt-1.1b. The 1.1b repo ships only a .nemo archive, which
  needs the full NVIDIA NeMo toolkit; transformers cannot read it. 0.6b-v3 ships
  safetensors and loads natively. It is also newer and scores better on the Open
  ASR leaderboard, so this is a substitution, not a downgrade.
* TDT decoding returns per-token `durations` alongside `sequences`. The current
  native Transformers implementation is greedy-only, so this runner does not
  pretend those durations form an N-best list. Beam alternatives require NeMo
  or a separate verified decoder implementation.

~2.5 GB. Needs librosa, which the other scripts do not.
"""

import torch
from transformers import ParakeetForTDT, ParakeetProcessor

import common

MODEL = "nvidia/parakeet-tdt-0.6b-v3"


def load_batch(device):
    processor = ParakeetProcessor.from_pretrained(MODEL)
    model = ParakeetForTDT.from_pretrained(MODEL)
    model.to(device).eval()

    def transcribe_batch(audios):
        inputs = processor(
            audios,
            sampling_rate=common.SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs)
        # generate() returns a greedy sequence + durations, not an N-best beam.
        return [
            text.strip()
            for text in processor.batch_decode(out.sequences, skip_special_tokens=True)
        ]

    return transcribe_batch


def load(device):
    transcribe_batch = load_batch(device)

    def transcribe(audio):
        return transcribe_batch([audio])[0]

    return transcribe


if __name__ == "__main__":
    common.run("parakeet", MODEL, "Parakeet-TDT 0.6b v3, stock (no TORGO adaptation)", load)
