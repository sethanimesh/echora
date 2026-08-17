"""Fine-tune Parakeet-CTC on one leakage-safe TORGO outer fold.

Designed for one NVIDIA GPU. Run once per outer test speaker; this script sees
train and development manifests only and never evaluates the outer test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    EarlyStoppingCallback,
    ParakeetForCTC,
    ParakeetProcessor,
    Trainer,
    TrainingArguments,
)

try:
    from .augment import AugmentConfig, augment
except ImportError:  # direct execution: python research/training/finetune_parakeet_ctc.py
    from augment import AugmentConfig, augment

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "research" / "benchmarks"))
from metrics import aggregate, score  # noqa: E402

DEFAULT_MODEL = "nvidia/parakeet-ctc-1.1b"


def read_manifest(path: Path, max_duration: float) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = [record for record in records if 0 < float(record["duration"]) <= max_duration]
    if not records:
        raise ValueError(f"no usable records in {path}")
    return records


def audit_split(train: list[dict], dev: list[dict]) -> dict:
    train_speakers = {record["speaker"] for record in train}
    dev_speakers = {record["speaker"] for record in dev}
    overlap = train_speakers & dev_speakers
    if overlap:
        raise ValueError(f"speaker leakage between train/dev: {sorted(overlap)}")
    train_ids = {record["id"] for record in train}
    dev_ids = {record["id"] for record in dev}
    if train_ids & dev_ids:
        raise ValueError("utterance leakage between train/dev")
    return {
        "train_utterances": len(train),
        "dev_utterances": len(dev),
        "train_speakers": sorted(train_speakers),
        "dev_speakers": sorted(dev_speakers),
        "train_conditions": dict(Counter(record["condition"] for record in train)),
        "shared_text_ids": len(
            {record["text_id"] for record in train} & {record["text_id"] for record in dev}
        ),
    }


def balanced_weights(records: list[dict], dysarthric_fraction: float) -> list[float]:
    """Equalize speakers within a requested dysarthric/control sampling mix."""
    if not 0 < dysarthric_fraction < 1:
        raise ValueError("dysarthric_fraction must be between zero and one")
    speaker_counts = Counter(record["speaker"] for record in records)
    speakers_by_condition: dict[str, set[str]] = defaultdict(set)
    for record in records:
        speakers_by_condition[record["condition"]].add(record["speaker"])
    required = {"dysarthric", "control"}
    if set(speakers_by_condition) != required:
        raise ValueError("training manifest must contain dysarthric and control speakers")
    condition_mass = {"dysarthric": dysarthric_fraction, "control": 1 - dysarthric_fraction}
    return [
        condition_mass[record["condition"]]
        / len(speakers_by_condition[record["condition"]])
        / speaker_counts[record["speaker"]]
        for record in records
    ]


class ManifestDataset(Dataset):
    def __init__(self, records: list[dict], apply_augmentation: bool):
        self.records = records
        self.apply_augmentation = apply_augmentation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return {**self.records[index], "apply_augmentation": self.apply_augmentation}


class AudioCollator:
    def __init__(
        self,
        processor: ParakeetProcessor,
        seed: int,
        augmentation: AugmentConfig,
    ):
        self.processor = processor
        self.rng = np.random.default_rng(seed)
        self.augmentation = augmentation

    def __call__(self, records: list[dict]) -> dict[str, torch.Tensor]:
        audio_batch: list[np.ndarray] = []
        for record in records:
            audio, sample_rate = sf.read(record["audio_filepath"], dtype="float32")
            if sample_rate != 16000:
                raise ValueError(f"expected 16 kHz: {record['audio_filepath']}")
            if audio.ndim != 1:
                audio = audio.mean(axis=1)
            if record["apply_augmentation"]:
                audio = augment(audio, sample_rate, self.rng, self.augmentation)
            audio_batch.append(audio)
        inputs = self.processor(
            audio_batch,
            sampling_rate=16000,
            text=[record["text"] for record in records],
            padding=True,
            return_tensors="pt",
        )
        # The shared processor creates this for transducers; CTC does not use it.
        inputs.pop("decoder_input_ids", None)
        return dict(inputs)


class BalancedTrainer(Trainer):
    def __init__(self, *args, sampling_weights: list[float], sampler_seed: int, **kwargs):
        self.sampling_weights = sampling_weights
        self.sampler_seed = sampler_seed
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        generator = torch.Generator().manual_seed(self.sampler_seed)
        return WeightedRandomSampler(
            self.sampling_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )


def set_trainable_blocks(model: ParakeetForCTC, unfreeze_last: int) -> tuple[int, int]:
    blocks = model.encoder.layers
    if not 0 <= unfreeze_last <= len(blocks):
        raise ValueError(f"--unfreeze-last must be between 0 and {len(blocks)}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.ctc_head.parameters():
        parameter.requires_grad = True
    if unfreeze_last:
        for block in blocks[-unfreeze_last:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def metric_function(processor: ParakeetProcessor, dev_records: list[dict]):
    def compute_metrics(prediction) -> dict[str, float]:
        predicted_ids = prediction.predictions
        if isinstance(predicted_ids, tuple):
            predicted_ids = predicted_ids[0]
        hypotheses = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        references = processor.batch_decode(prediction.label_ids, skip_special_tokens=True)
        scores = [score(reference, hypothesis) for reference, hypothesis in zip(references, hypotheses)]
        overall = aggregate(scores)
        by_speaker: dict[str, list] = defaultdict(list)
        if len(scores) != len(dev_records):
            raise ValueError("evaluation output order/length does not match the dev manifest")
        for record, item in zip(dev_records, scores):
            by_speaker[record["speaker"]].append(item)
        macro_wer = float(np.mean([aggregate(items).wer for items in by_speaker.values()]))
        return {"wer": overall.wer, "cer": overall.cer, "macro_wer": macro_wer}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--unfreeze-last", type=int, default=10)
    parser.add_argument("--dysarthric-fraction", type=float, default=0.85)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="audit manifests without loading a model")
    parser.add_argument("--allow-cpu", action="store_true", help="debug only; real training needs CUDA")
    args = parser.parse_args()

    train_records = read_manifest(args.train_manifest, args.max_duration)
    dev_records = read_manifest(args.dev_manifest, args.max_duration)
    audit = audit_split(train_records, dev_records)
    weights = balanced_weights(train_records, args.dysarthric_fraction)
    audit["target_dysarthric_fraction"] = args.dysarthric_fraction
    audit["max_duration"] = args.max_duration
    print(json.dumps(audit, indent=2))
    if args.dry_run:
        return
    if not torch.cuda.is_available() and not args.allow_cpu:
        parser.error("CUDA is required for this 1.1B fine-tune; use --dry-run to audit locally")

    torch.manual_seed(args.seed)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    processor = ParakeetProcessor.from_pretrained(args.model)
    model = ParakeetForCTC.from_pretrained(args.model, dtype=dtype)
    trainable, total = set_trainable_blocks(model, args.unfreeze_last)
    print(f"trainable parameters: {trainable:,}/{total:,} ({trainable / total:.1%})")

    augmentation = AugmentConfig()
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "train_manifest": str(args.train_manifest.resolve()),
        "dev_manifest": str(args.dev_manifest.resolve()),
        "output": str(args.output.resolve()),
        "augmentation": asdict(augmentation),
        "audit": audit,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
    (args.output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n"
    )
    processor.save_pretrained(args.output)

    train_dataset = ManifestDataset(train_records, apply_augmentation=True)
    dev_dataset = ManifestDataset(dev_records, apply_augmentation=False)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        do_train=True,
        do_eval=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=0.05,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_wer",
        greater_is_better=False,
        eval_accumulation_steps=4,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
    )

    def logits_to_ids(logits, _labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=AudioCollator(processor, args.seed, augmentation),
        processing_class=processor,
        compute_metrics=metric_function(processor, dev_records),
        preprocess_logits_for_metrics=logits_to_ids,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        sampling_weights=weights,
        sampler_seed=args.seed,
    )
    trainer.train()
    trainer.save_model(args.output / "best")
    processor.save_pretrained(args.output / "best")
    metrics = trainer.evaluate()
    (args.output / "final_dev_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
