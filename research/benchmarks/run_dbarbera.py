"""dbarbera/whisper-small-torgo-dysarthria-lora

LoRA adapter on the whisper-small decoder (q_proj, v_proj, rank 16, 1.77M params).
The only TORGO model on the Hub with a real model card: speaker-disjoint splits,
stated normalisation, per-speaker results. Author reports 40.7% WER on dysarthric
utterances, 16.9% overall. ~1 GB (base model) + 7 MB (adapter).
"""

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

import common

BASE = "openai/whisper-small"
ADAPTER = "dbarbera/whisper-small-torgo-dysarthria-lora"


def load(device):
    model = WhisperForConditionalGeneration.from_pretrained(BASE)
    processor = common.whisper_processor(BASE, model.config.num_mel_bins)
    model = PeftModel.from_pretrained(model, ADAPTER)
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
    common.run("dbarbera", ADAPTER, "whisper-small + LoRA on decoder", load)
