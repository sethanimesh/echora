"""macarious/torgo_xlsr_finetune_M04

wav2vec2-large-xlsr-53 with a CTC head, fine-tuned leave-one-speaker-out with
TORGO speaker M04 held out. Reported eval WER 0.2742. This is the CTC family
representative: no language model, no decoder, so it transcribes what it hears
phonetically and will not invent fluent English the way Whisper does.

That property is why it is worth testing for Echora even though its absolute WER
is unremarkable -- a CTC model failing loudly is more useful to a confirmation
loop than a seq2seq model hallucinating confidently. ~1.3 GB.
"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import common

MODEL = "macarious/torgo_xlsr_finetune_M04"


def load(device):
    processor = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL)
    model.to(device).eval()

    def transcribe(audio):
        values = processor(
            audio, sampling_rate=common.SAMPLE_RATE, return_tensors="pt"
        ).input_values.to(device)
        with torch.no_grad():
            logits = model(values).logits
        return processor.batch_decode(logits.argmax(dim=-1))[0].strip()

    return transcribe


if __name__ == "__main__":
    common.run("macarious", MODEL, "wav2vec2-large-xlsr-53 + CTC", load)
