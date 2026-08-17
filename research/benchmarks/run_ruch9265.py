"""ruch9265/distil-whisper-torgo

Full encoder+decoder fine-tune of whisper-tiny on the TORGO mirror
(13,239 train / 1,656 val / 1,655 test), 2000 steps at lr 5e-6. Card documents
the split sizes and hyperparameters but reports no WER, and the split is a
random utterance split, so treat its accuracy on your voice as the real test.
~150 MB.
"""

import torch
from transformers import WhisperForConditionalGeneration

import common

MODEL = "ruch9265/distil-whisper-torgo"


def load(device):
    model = WhisperForConditionalGeneration.from_pretrained(MODEL)
    processor = common.whisper_processor(MODEL, model.config.num_mel_bins)
    model.to(device).eval()

    def transcribe(audio):
        features = processor(
            audio, sampling_rate=common.SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            ids = model.generate(
                input_features=features, language="en", task="transcribe", max_new_tokens=128
            )
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    return transcribe


if __name__ == "__main__":
    common.run("ruch9265", MODEL, "whisper-tiny, full fine-tune", load)
