"""sqrk/torgo-mms1ball-Nov29

facebook/mms-1b-all fine-tuned on TORGO, reported eval WER 0.4196. The card is
auto-generated ("unknown dataset"), but the number is the honest one in this
family: it sits right in the range the published literature reports for
speaker-independent TORGO, which is why this is here and the same account's
whisper-large-v3 run (claiming ~5% WER) is not.

Largest model in the set at ~3.9 GB, and the slowest on CPU. The tokenizer lives
in the upstream MMS repo, not this one, so both are fetched.
"""

import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

import common

MODEL = "sqrk/torgo-mms1ball-Nov29"
UPSTREAM = "facebook/mms-1b-all"


def load(device):
    processor = AutoProcessor.from_pretrained(UPSTREAM, target_lang="eng")
    # No target_lang here on purpose: this checkpoint's model.safetensors already
    # holds the fine-tuned English adapter, and asking transformers to reload it
    # fails because the repo ships adapter.eng.bin rather than a safetensors file.
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
    common.run("sqrk_mms", MODEL, "MMS-1B-all + CTC", load)
