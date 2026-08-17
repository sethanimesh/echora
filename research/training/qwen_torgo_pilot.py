"""End-to-end partial Qwen3-ASR adaptation for universal dysarthric speech.

This script owns the complete paid pilot: input auditing, frozen-foundation
baselines, deterministic speaker-balanced sampling, training-only waveform
augmentation, resumable partial checkpoints, development-WER selection,
ordinary-English retention gates, protected test evaluation and a deployable
adapter export.  It produces literal ASR transcripts only; no semantic ranker
or personal context is imported anywhere in this module.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "research" / "benchmarks"))
from metrics import aggregate, normalize, score  # noqa: E402

try:
    from .augment import AugmentConfig, augment
except ImportError:
    from augment import AugmentConfig, augment


MANIFEST_NAMES = (
    "torgo_train",
    "torgo_dev",
    "torgo_test",
    "normal_train",
    "normal_dev",
    "normal_test",
    "personal",
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *values: object) -> int:
    payload = ":".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def audit_data(config: dict, manifests: dict[str, list[dict]]) -> dict:
    for name in MANIFEST_NAMES:
        if name not in manifests:
            raise ValueError(f"Required manifest is missing: {name}")
        for row in manifests[name]:
            required = {"id", "speaker", "condition", "duration", "text", "audio_filepath"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"{name}:{row.get('id', '?')} is missing {sorted(missing)}")
            if not Path(row["audio_filepath"]).is_file():
                raise FileNotFoundError(f"Missing staged audio: {row['audio_filepath']}")
            if not 0 < float(row["duration"]) <= float(config["maximum_audio_seconds"]):
                raise ValueError(f"Invalid duration in {name}: {row['id']}")
            if not str(row["text"]).strip():
                raise ValueError(f"Empty transcript in {name}: {row['id']}")

    torgo_train = manifests["torgo_train"]
    torgo_dev = manifests["torgo_dev"]
    torgo_test = manifests["torgo_test"]
    if {row["speaker"] for row in torgo_dev} != {config["development_speaker"]}:
        raise ValueError("Unexpected TORGO development speaker")
    if {row["speaker"] for row in torgo_test} != {config["outer_test_speaker"]}:
        raise ValueError("Unexpected TORGO outer-test speaker")
    held_out = {config["development_speaker"], config["outer_test_speaker"]}
    if held_out & {row["speaker"] for row in torgo_train}:
        raise ValueError("Held-out TORGO speaker leaked into training")
    test_prompts = {row.get("text_id") for row in torgo_test}
    if None in test_prompts:
        raise ValueError("TORGO test text IDs are required")
    overlap = test_prompts & {row.get("text_id") for row in torgo_train + torgo_dev}
    if overlap and not config.get("allow_outer_test_prompt_overlap", False):
        raise ValueError("Outer-test prompt leaked into strict train/dev manifests")

    dysarthric = [row for row in torgo_train if row["condition"] == "dysarthric"]
    control = [row for row in torgo_train if row["condition"] == "control"]
    if not dysarthric or not control:
        raise ValueError("TORGO train needs dysarthric and control recordings")
    normal_sets = [
        {row["speaker"] for row in manifests[name]}
        for name in ("normal_train", "normal_dev", "normal_test")
    ]
    if any(normal_sets[a] & normal_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Normal-speech speakers overlap across train/dev/test")
    all_ids = [row["id"] for name in MANIFEST_NAMES for row in manifests[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Utterance IDs overlap across pilot manifests")
    return {
        "outer_test_prompt_overlap_count": len(overlap),
        "prompt_overlap_allowed_and_disclosed": bool(
            config.get("allow_outer_test_prompt_overlap", False)
        ),
        "dysarthric_training_utterances": len(dysarthric),
        "dysarthric_training_speakers": sorted({row["speaker"] for row in dysarthric}),
        "torgo_control_pool_utterances": len(control),
        "normal_training_pool_utterances": len(manifests["normal_train"]),
        "development_utterances": len(torgo_dev),
        "outer_test_utterances": len(torgo_test),
        "normal_dev_utterances": len(manifests["normal_dev"]),
        "normal_test_utterances": len(manifests["normal_test"]),
        "personal_smoke_utterances": len(manifests["personal"]),
    }


def validate_foundation(config: dict, model_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The paid pilot requires CUDA; refusing CPU or MPS fallback")
    gpu = torch.cuda.get_device_name(0)
    required_gpu = config["required_gpu_name_fragment"]
    if required_gpu.lower() not in gpu.lower():
        raise RuntimeError(f"The frozen pilot requires {required_gpu}; detected {gpu}")
    weight = model_dir / config["model_weight_file"]
    if not weight.is_file():
        raise FileNotFoundError(f"Foundation weight is missing: {weight}")
    if weight.stat().st_size != int(config["model_weight_bytes"]):
        raise RuntimeError("Foundation byte count does not match the frozen configuration")
    if sha256(weight) != config["model_weight_sha256"]:
        raise RuntimeError("Foundation SHA-256 does not match the frozen configuration")


def configure_trainable_parameters(model, config: dict) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.model.audio_tower.layers
    selected = [int(index) for index in config["trainable_audio_layer_indices"]]
    if len(layers) != 24:
        raise RuntimeError(f"Expected 24 Qwen audio layers, found {len(layers)}")
    if selected != [20, 21, 22, 23]:
        raise RuntimeError(f"Pilot trainable layers changed unexpectedly: {selected}")
    for index in selected:
        for parameter in layers[index].parameters():
            parameter.requires_grad = True
    if config["train_multimodal_projector"] is not True:
        raise RuntimeError("The multimodal projector must remain trainable")
    for parameter in model.model.multi_modal_projector.parameters():
        parameter.requires_grad = True
    prefixes = tuple(
        [f"model.audio_tower.layers.{index}." for index in selected]
        + ["model.multi_modal_projector."]
    )
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name in names if not name.startswith(prefixes)]
    if unexpected or not names:
        raise RuntimeError(f"Unexpected trainable parameter set: {unexpected[:5]}")
    return names


def set_partial_training_mode(model, config: dict) -> None:
    model.eval()
    for index in config["trainable_audio_layer_indices"]:
        model.model.audio_tower.layers[int(index)].train()
    model.model.multi_modal_projector.train()


def load_model(config: dict, model_dir: Path):
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    configure_trainable_parameters(model, config)
    return processor, model


def load_audio(row: dict) -> np.ndarray:
    audio, _ = librosa.load(row["audio_filepath"], sr=16_000, mono=True, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio for {row['id']}")
    return audio


def prepare_training_example(
    processor, row: dict, augmentation: AugmentConfig, apply_probability: float, epoch: int, seed: int
) -> dict[str, torch.Tensor]:
    audio = load_audio(row)
    if row["condition"] == "dysarthric":
        rng = np.random.default_rng(stable_seed(seed, epoch, row["id"], "augment"))
        if rng.random() < apply_probability:
            audio = augment(audio, 16_000, rng, augmentation)
    conversation = [
        [
            {"role": "user", "content": [{"type": "audio", "audio": audio}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"language English<asr_text>{row['text'].strip()}",
                    }
                ],
            },
        ]
    ]
    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        processor_kwargs={"output_labels": True, "sampling_rate": 16_000},
    )
    return {key: value for key, value in batch.items() if torch.is_tensor(value)}


def to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}


def forward_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(**to_cuda(batch)).loss
    if loss is None or not torch.isfinite(loss):
        value = None if loss is None else float(loss.detach().cpu())
        raise FloatingPointError(f"Non-finite loss: {value}")
    return loss


def select_speaker_balanced(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count <= 0:
        return []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["speaker"]].append(row)
    speakers = list(grouped)
    random.Random(seed).shuffle(speakers)
    for speaker in speakers:
        random.Random(stable_seed(seed, speaker)).shuffle(grouped[speaker])
    offsets = {speaker: 0 for speaker in speakers}
    selected = []
    while len(selected) < count:
        for speaker in speakers:
            values = grouped[speaker]
            selected.append(values[offsets[speaker] % len(values)])
            offsets[speaker] += 1
            if len(selected) == count:
                break
    return selected


def select_coverage_slice(rows: list[dict], count: int, epoch: int, seed: int) -> list[dict]:
    """Rotate through a pool so all records are seen before any are repeated."""
    if count <= 0:
        return []
    ordered = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest())
    start = (epoch - 1) * count
    return [ordered[(start + offset) % len(ordered)] for offset in range(count)]


def epoch_schedule(manifests: dict[str, list[dict]], config: dict, epoch: int) -> list[dict]:
    dysarthric = [
        row for row in manifests["torgo_train"] if row["condition"] == "dysarthric"
    ]
    controls = [row for row in manifests["torgo_train"] if row["condition"] == "control"]
    seed = stable_seed(int(config["seed"]), epoch, "schedule")
    schedule = list(dysarthric)
    schedule.extend(
        select_coverage_slice(
            controls,
            int(config["torgo_control_examples_per_epoch"]),
            epoch,
            int(config["seed"]) ^ 0xC011,
        )
    )
    schedule.extend(
        select_speaker_balanced(
            manifests["normal_train"], int(config["normal_examples_per_epoch"]), seed ^ 0xA11CE
        )
    )
    random.Random(seed).shuffle(schedule)
    if Counter(row["id"] for row in schedule if row["condition"] == "dysarthric") != Counter(
        row["id"] for row in dysarthric
    ):
        raise AssertionError("Every eligible dysarthric row must appear exactly once per epoch")
    return schedule


def transcribe(model, processor, audio: np.ndarray, max_new_tokens: int) -> str:
    inputs = processor.apply_transcription_request(
        audio=[audio], language=["English"]
    ).to(model.device, model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    decoded = processor.decode(generated, return_format="transcription_only")
    return decoded[0].strip()


def score_predictions(rows: list[dict], predictions: list[str]) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("Prediction count does not match manifest")
    scored = [score(row["text"], prediction) for row, prediction in zip(rows, predictions)]
    total = aggregate(scored)
    by_speaker: dict[str, list] = defaultdict(list)
    for row, item in zip(rows, scored):
        by_speaker[row["speaker"]].append(item)
    speaker_wer = {
        speaker: aggregate(items).wer for speaker, items in sorted(by_speaker.items())
    }
    return {
        "utterances": len(rows),
        "micro_wer": total.wer,
        "macro_speaker_wer": sum(speaker_wer.values()) / len(speaker_wer),
        "micro_cer": total.cer,
        "word_substitutions": total.word_substitutions,
        "word_deletions": total.word_deletions,
        "word_insertions": total.word_insertions,
        "reference_words": total.reference_words,
        "deletion_rate": total.word_deletions / total.reference_words,
        "empty_outputs": sum(not normalize(prediction) for prediction in predictions),
        "empty_rate": sum(not normalize(prediction) for prediction in predictions) / len(rows),
        "speaker_wer": speaker_wer,
    }


def evaluate(
    model,
    processor,
    rows: list[dict],
    output: Path,
    max_new_tokens: int,
    label: str,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions, decode_seconds = [], []
    temporary = output.with_suffix(output.suffix + ".tmp")
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "id",
                "speaker",
                "condition",
                "audio_filepath",
                "audio_seconds",
                "decode_seconds",
                "reference",
                "transcript",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for index, row in enumerate(rows, 1):
                audio = load_audio(row)
                started = time.perf_counter()
                prediction = transcribe(model, processor, audio, max_new_tokens)
                elapsed = time.perf_counter() - started
                predictions.append(prediction)
                decode_seconds.append(elapsed)
                writer.writerow(
                    {
                        "id": row["id"],
                        "speaker": row["speaker"],
                        "condition": row["condition"],
                        "audio_filepath": row["audio_filepath"],
                        "audio_seconds": f"{len(audio) / 16000:.4f}",
                        "decode_seconds": f"{elapsed:.4f}",
                        "reference": row["text"],
                        "transcript": prediction,
                    }
                )
                handle.flush()
                if index % 50 == 0 or index == len(rows):
                    print(f"eval {label}: {index}/{len(rows)}", flush=True)
    finally:
        model.config.use_cache = previous_cache
    temporary.replace(output)
    metrics = score_predictions(rows, predictions)
    metrics["decode_seconds"] = sum(decode_seconds)
    metrics["audio_seconds"] = sum(float(row["duration"]) for row in rows)
    metrics["rtf"] = metrics["decode_seconds"] / metrics["audio_seconds"]
    metrics["predictions_sha256"] = sha256(output)
    print(
        f"eval {label}: WER={metrics['micro_wer']:.2%} CER={metrics['micro_cer']:.2%} "
        f"deletions={metrics['deletion_rate']:.2%} empty={metrics['empty_rate']:.2%}",
        flush=True,
    )
    return metrics


def cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def optimizer_to_cuda(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to("cuda")


def partial_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_partial_state(model, saved: dict[str, torch.Tensor]) -> None:
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(trainable) != set(saved):
        raise RuntimeError("Saved adapter parameter names do not match the frozen recipe")
    with torch.no_grad():
        for name, parameter in trainable.items():
            if parameter.shape != saved[name].shape:
                raise RuntimeError(f"Saved adapter shape mismatch: {name}")
            parameter.copy_(saved[name].to(parameter.device, parameter.dtype))


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    metadata: dict,
) -> None:
    payload = {
        "format": "echora-qwen-partial-pilot-v1",
        **metadata,
        "trainable_state": partial_state(model),
        "optimizer_state": cpu_copy(optimizer.state_dict()),
        "scheduler_state": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path: Path, model, optimizer, scheduler, identity: dict) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "echora-qwen-partial-pilot-v1":
        raise RuntimeError("Unknown pilot checkpoint format")
    for key in ("run_id", "base_model_sha256", "config_sha256", "manifest_sha256"):
        if saved.get(key) != identity[key]:
            raise RuntimeError(f"Checkpoint {key} does not match this run")
    load_partial_state(model, saved["trainable_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
    optimizer_to_cuda(optimizer)
    scheduler.load_state_dict(saved["scheduler_state"])
    torch.set_rng_state(saved["torch_rng_state"])
    torch.cuda.set_rng_state(saved["cuda_rng_state"])
    return saved


def make_optimizer(model, config: dict):
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        fused=True,
    )


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = round(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def development_guard(candidate: dict, baseline: dict, config: dict) -> tuple[bool, list[str]]:
    reasons = []
    normal_limit = (
        baseline["normal_dev"]["micro_wer"]
        + float(config["maximum_normal_dev_wer_degradation_absolute"])
    )
    if candidate["normal_dev"]["micro_wer"] > normal_limit:
        reasons.append("normal-dev WER guard")
    deletion_limit = (
        baseline["torgo_dev"]["deletion_rate"]
        + float(config["maximum_dev_deletion_rate_increase_absolute"])
    )
    if candidate["torgo_dev"]["deletion_rate"] > deletion_limit:
        reasons.append("dysarthric deletion-rate guard")
    empty_limit = (
        baseline["torgo_dev"]["empty_rate"]
        + float(config["maximum_dev_empty_rate_increase_absolute"])
    )
    if candidate["torgo_dev"]["empty_rate"] > empty_limit:
        reasons.append("dysarthric empty-output guard")
    return not reasons, reasons


def export_adapter(path: Path, model, config: dict, identity: dict, best: dict) -> dict:
    state = {name: value.contiguous() for name, value in partial_state(model).items()}
    metadata = {
        "format": "echora-qwen3-asr-partial-v1",
        "run_id": config["run_id"],
        "base_model_id": config["model_id"],
        "base_model_revision": config["model_revision"],
        "base_model_sha256": config["model_weight_sha256"],
        "config_sha256": identity["config_sha256"],
        "manifest_sha256": identity["manifest_sha256"],
        "selected_epoch": str(best["epoch"]),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_safetensors(state, temporary, metadata=metadata)
    temporary.replace(path)
    descriptor = {
        **metadata,
        "trainable_audio_layer_indices": config["trainable_audio_layer_indices"],
        "train_multimodal_projector": config["train_multimodal_projector"],
        "adapter_file": path.name,
        "adapter_sha256": sha256(path),
    }
    path.with_name("adapter_config.json").write_text(
        json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
    )
    return descriptor


def combined_manifest_sha(paths: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(sha256(paths[name]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def train(
    config_path: Path,
    data_dir: Path,
    model_dir: Path,
    output: Path,
    resume: Path | None,
) -> dict:
    config = read_json(config_path)
    manifest_paths = {
        name: data_dir / "manifests" / f"{name}.jsonl" for name in MANIFEST_NAMES
    }
    manifests = {name: read_jsonl(path) for name, path in manifest_paths.items()}
    audit = audit_data(config, manifests)
    validate_foundation(config, model_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    identity = {
        "run_id": config["run_id"],
        "base_model_sha256": config["model_weight_sha256"],
        "config_sha256": sha256(config_path),
        "manifest_sha256": combined_manifest_sha(manifest_paths),
    }
    run_manifest = {
        **identity,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "manifest_files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in manifest_paths.items()
        },
        "audit": audit,
        "config": config,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )

    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]) % (2**32))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    print(f"run       {config['run_id']}", flush=True)
    print(f"model     {config['model_id']} @ {config['model_revision']}", flush=True)
    print(f"gpu       {torch.cuda.get_device_name(0)}", flush=True)
    print(f"data      {json.dumps(audit, sort_keys=True)}", flush=True)
    processor, model = load_model(config, model_dir)
    trainable_names = [name for name, item in model.named_parameters() if item.requires_grad]
    trainable_count = sum(item.numel() for item in model.parameters() if item.requires_grad)
    total_count = sum(item.numel() for item in model.parameters())
    print(
        f"trainable {trainable_count:,}/{total_count:,} ({trainable_count / total_count:.2%})",
        flush=True,
    )

    max_tokens = int(config["generation_max_new_tokens"])
    baseline_path = output / "baseline.json"
    if baseline_path.is_file():
        baseline = read_json(baseline_path)
        if any(baseline.get(key) != identity[key] for key in identity):
            raise RuntimeError("Existing baseline belongs to different config or manifests")
        print("baseline  reusing verified baseline.json", flush=True)
    else:
        print("baseline  frozen Qwen development and smoke evaluation", flush=True)
        baseline = {
            **identity,
            "torgo_dev": evaluate(
                model,
                processor,
                manifests["torgo_dev"],
                predictions_dir / "baseline_torgo_dev.tsv",
                max_tokens,
                "baseline/torgo-dev",
            ),
            "normal_dev": evaluate(
                model,
                processor,
                manifests["normal_dev"],
                predictions_dir / "baseline_normal_dev.tsv",
                max_tokens,
                "baseline/normal-dev",
            ),
            "personal": evaluate(
                model,
                processor,
                manifests["personal"],
                predictions_dir / "baseline_personal.tsv",
                max_tokens,
                "baseline/personal-smoke",
            ),
        }
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    epoch_size = len(epoch_schedule(manifests, config, 1))
    accumulation = int(config["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(epoch_size / accumulation)
    total_steps = steps_per_epoch * int(config["maximum_epochs"])
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, total_steps, float(config["warmup_ratio"]))
    augmentation_values = dict(config["augmentation"])
    apply_probability = float(augmentation_values.pop("apply_probability"))
    augmentation = AugmentConfig(**augmentation_values)

    state = {
        "next_epoch": 1,
        "next_micro_index": 0,
        "global_step": 0,
        "best": None,
        "epochs_without_improvement": 0,
        "history": [],
    }
    if resume is not None:
        print(f"resume    {resume}", flush=True)
        saved = restore_checkpoint(resume, model, optimizer, scheduler, identity)
        state.update({key: saved[key] for key in state})

    latest_path = output / "checkpoint-latest.pt"
    best_path = output / "checkpoint-best.pt"
    stop_reason = "maximum_epochs"
    for epoch in range(int(state["next_epoch"]), int(config["maximum_epochs"]) + 1):
        schedule = epoch_schedule(manifests, config, epoch)
        start_index = int(state["next_micro_index"]) if epoch == int(state["next_epoch"]) else 0
        if not 0 <= start_index <= len(schedule):
            raise RuntimeError("Checkpoint microbatch position is invalid")
        epoch_losses = []
        print(
            f"epoch {epoch}/{config['maximum_epochs']} examples={len(schedule)} "
            f"resume_index={start_index}",
            flush=True,
        )
        for chunk_start in range(start_index, len(schedule), accumulation):
            chunk = schedule[chunk_start : chunk_start + accumulation]
            set_partial_training_mode(model, config)
            optimizer.zero_grad(set_to_none=True)
            chunk_losses = []
            for row in chunk:
                batch = prepare_training_example(
                    processor,
                    row,
                    augmentation,
                    apply_probability,
                    epoch,
                    int(config["seed"]),
                )
                loss = forward_loss(model, batch)
                (loss / len(chunk)).backward()
                chunk_losses.append(float(loss.detach().cpu()))
                del batch, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [item for item in model.parameters() if item.requires_grad],
                float(config["maximum_gradient_norm"]),
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            state["next_epoch"] = epoch
            state["next_micro_index"] = min(chunk_start + len(chunk), len(schedule))
            epoch_losses.extend(chunk_losses)
            if state["global_step"] % 10 == 0:
                print(
                    f"step {state['global_step']}/{total_steps} "
                    f"loss={sum(chunk_losses) / len(chunk_losses):.5f} "
                    f"grad={float(grad_norm):.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if state["global_step"] % int(config["checkpoint_every_optimizer_steps"]) == 0:
                save_checkpoint(latest_path, model, optimizer, scheduler, {**identity, **state})
                print(f"saved     resumable {latest_path} at step {state['global_step']}", flush=True)

        model.eval()
        epoch_metrics = {
            "epoch": epoch,
            "global_step": state["global_step"],
            "training_mean_loss": sum(epoch_losses) / len(epoch_losses),
            "torgo_dev": evaluate(
                model,
                processor,
                manifests["torgo_dev"],
                predictions_dir / f"epoch_{epoch:02d}_torgo_dev.tsv",
                max_tokens,
                f"epoch-{epoch}/torgo-dev",
            ),
            "normal_dev": evaluate(
                model,
                processor,
                manifests["normal_dev"],
                predictions_dir / f"epoch_{epoch:02d}_normal_dev.tsv",
                max_tokens,
                f"epoch-{epoch}/normal-dev",
            ),
        }
        eligible, reasons = development_guard(epoch_metrics, baseline, config)
        epoch_metrics["selection_guard_passed"] = eligible
        epoch_metrics["selection_guard_failures"] = reasons
        state["history"].append(epoch_metrics)
        improved = eligible and (
            state["best"] is None
            or epoch_metrics["torgo_dev"]["macro_speaker_wer"]
            < state["best"]["torgo_dev"]["macro_speaker_wer"]
        )
        if improved:
            state["best"] = epoch_metrics
            state["epochs_without_improvement"] = 0
            save_checkpoint(best_path, model, optimizer, scheduler, {**identity, **state})
            print(f"selected  epoch {epoch} as best guarded development checkpoint", flush=True)
        else:
            state["epochs_without_improvement"] += 1
            print(
                f"not best  epoch {epoch}; guard={eligible} reasons={reasons or ['higher WER']}",
                flush=True,
            )

        state["next_epoch"] = epoch + 1
        state["next_micro_index"] = 0
        save_checkpoint(latest_path, model, optimizer, scheduler, {**identity, **state})
        (output / "training_history.json").write_text(
            json.dumps(state["history"], indent=2) + "\n", encoding="utf-8"
        )
        if (
            epoch >= int(config["minimum_epochs"])
            and state["epochs_without_improvement"] >= int(config["early_stopping_patience"])
        ):
            stop_reason = "development_early_stopping"
            print(f"stopping   {stop_reason}", flush=True)
            break

    if state["best"] is None or not best_path.is_file():
        result = {
            "status": "failed_no_guarded_checkpoint",
            **identity,
            "baseline": baseline,
            "history": state["history"],
            "test_evaluation_performed": False,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError("No checkpoint passed the normal-speech/deletion/empty-output guardrails")

    best_saved = torch.load(best_path, map_location="cpu", weights_only=False)
    load_partial_state(model, best_saved["trainable_state"])
    best = state["best"]
    dev_relative_gain = (
        baseline["torgo_dev"]["macro_speaker_wer"]
        - best["torgo_dev"]["macro_speaker_wer"]
    ) / baseline["torgo_dev"]["macro_speaker_wer"]
    adapted_personal = evaluate(
        model,
        processor,
        manifests["personal"],
        predictions_dir / "adapted_personal.tsv",
        max_tokens,
        "best/personal-smoke",
    )
    adapter = export_adapter(output / "adapter.safetensors", model, config, identity, best)
    if dev_relative_gain < float(config["required_relative_dev_wer_improvement"]):
        result = {
            "status": "failed_development_improvement_gate",
            **identity,
            "stop_reason": stop_reason,
            "baseline": baseline,
            "best_development": best,
            "relative_dev_wer_improvement": dev_relative_gain,
            "adapted_personal": adapted_personal,
            "adapter": adapter,
            "test_evaluation_performed": False,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "elapsed_seconds": time.monotonic() - started,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError(
            f"Development WER gain {dev_relative_gain:.2%} did not reach the required "
            f"{float(config['required_relative_dev_wer_improvement']):.2%}; outer test remains sealed"
        )

    print("test      development gate passed; unsealing M04 and normal test once", flush=True)
    adapted_test = {
        "torgo_test": evaluate(
            model,
            processor,
            manifests["torgo_test"],
            predictions_dir / "adapted_torgo_test.tsv",
            max_tokens,
            "adapted/torgo-test",
        ),
        "normal_test": evaluate(
            model,
            processor,
            manifests["normal_test"],
            predictions_dir / "adapted_normal_test.tsv",
            max_tokens,
            "adapted/normal-test",
        ),
    }

    del optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    print("reload    foundation for protected test baselines", flush=True)
    processor, foundation = load_model(config, model_dir)
    foundation.eval()
    foundation_test = {
        "torgo_test": evaluate(
            foundation,
            processor,
            manifests["torgo_test"],
            predictions_dir / "foundation_torgo_test.tsv",
            max_tokens,
            "foundation/torgo-test",
        ),
        "normal_test": evaluate(
            foundation,
            processor,
            manifests["normal_test"],
            predictions_dir / "foundation_normal_test.tsv",
            max_tokens,
            "foundation/normal-test",
        ),
    }
    load_partial_state(foundation, best_saved["trainable_state"])
    reloaded_personal = evaluate(
        foundation,
        processor,
        manifests["personal"],
        predictions_dir / "reloaded_adapter_personal.tsv",
        max_tokens,
        "reloaded-adapter/personal-smoke",
    )
    adapted_lines = (predictions_dir / "adapted_personal.tsv").read_text().splitlines()
    reloaded_lines = (predictions_dir / "reloaded_adapter_personal.tsv").read_text().splitlines()
    adapted_transcripts = [row["transcript"] for row in csv.DictReader(adapted_lines, delimiter="\t")]
    reloaded_transcripts = [row["transcript"] for row in csv.DictReader(reloaded_lines, delimiter="\t")]
    if adapted_transcripts != reloaded_transcripts:
        raise RuntimeError("Fresh foundation + exported adapter changed personal transcripts")

    normal_test_degradation = (
        adapted_test["normal_test"]["micro_wer"]
        - foundation_test["normal_test"]["micro_wer"]
    )
    test_relative_gain = (
        foundation_test["torgo_test"]["macro_speaker_wer"]
        - adapted_test["torgo_test"]["macro_speaker_wer"]
    ) / foundation_test["torgo_test"]["macro_speaker_wer"]
    retention_passed = normal_test_degradation <= float(
        config["maximum_normal_dev_wer_degradation_absolute"]
    )
    test_failures = []
    if test_relative_gain < float(config["required_relative_test_wer_improvement_for_success"]):
        test_failures.append("outer-test WER improvement")
    if adapted_test["torgo_test"]["deletion_rate"] > (
        foundation_test["torgo_test"]["deletion_rate"]
        + float(config["maximum_test_deletion_rate_increase_absolute"])
    ):
        test_failures.append("outer-test deletion rate")
    if adapted_test["torgo_test"]["empty_rate"] > (
        foundation_test["torgo_test"]["empty_rate"]
        + float(config["maximum_test_empty_rate_increase_absolute"])
    ):
        test_failures.append("outer-test empty-output rate")
    if not retention_passed:
        final_status = "failed_normal_test_retention"
    elif test_failures:
        final_status = "failed_outer_test_generalization"
    else:
        final_status = "passed"
    result = {
        "status": final_status,
        **identity,
        "gpu": torch.cuda.get_device_name(0),
        "stop_reason": stop_reason,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "trainable_parameter_names": trainable_names,
        "baseline_development": baseline,
        "best_development": best,
        "relative_dev_wer_improvement": dev_relative_gain,
        "foundation_test": foundation_test,
        "adapted_test": adapted_test,
        "relative_torgo_test_wer_improvement": test_relative_gain,
        "normal_test_wer_degradation_absolute": normal_test_degradation,
        "outer_test_guard_failures": test_failures,
        "personal_baseline": baseline["personal"],
        "personal_adapted": adapted_personal,
        "adapter_reload_transcripts_identical": True,
        "adapter": adapter,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256(best_path),
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_sha256": sha256(latest_path),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.monotonic() - started,
        "test_evaluation_performed": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"result    status={result['status']} dev_gain={dev_relative_gain:.2%} "
        f"test_gain={test_relative_gain:.2%} normal_test_delta={normal_test_degradation:+.2%}",
        flush=True,
    )
    if not retention_passed:
        raise RuntimeError("Best dysarthric checkpoint failed the independent normal-test retention guard")
    if test_failures:
        raise RuntimeError(f"Best checkpoint failed outer-test generalization: {test_failures}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.data, args.model, args.output, args.resume)
