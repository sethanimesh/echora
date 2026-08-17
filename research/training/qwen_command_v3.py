"""Large literal-command adaptation of Qwen3-ASR using the available data only.

The run starts from the exact official foundation, trains the top four audio
layers/projector plus a small decoder LoRA, and adds same-speaker TORGO word
compositions with controlled gaps and stretching.  It reports raw beam
alternatives; no semantic ranker, grammar repair, or personal context is used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .augment import AugmentConfig, augment
    from .qwen_command_model import (
        load_model,
        load_partial_state,
        load_safetensors_adapter,
        partial_state,
        save_adapter,
        set_training_mode,
    )
    from .qwen_torgo_pilot import (
        aggregate,
        combined_manifest_sha,
        cpu_copy,
        load_audio,
        normalize,
        optimizer_to_cuda,
        read_json,
        read_jsonl,
        score,
        select_coverage_slice,
        select_speaker_balanced,
        sha256,
        stable_seed,
        validate_foundation,
    )
except ImportError:
    from augment import AugmentConfig, augment
    from qwen_command_model import (
        load_model,
        load_partial_state,
        load_safetensors_adapter,
        partial_state,
        save_adapter,
        set_training_mode,
    )
    from qwen_torgo_pilot import (
        aggregate,
        combined_manifest_sha,
        cpu_copy,
        load_audio,
        normalize,
        optimizer_to_cuda,
        read_json,
        read_jsonl,
        score,
        select_coverage_slice,
        select_speaker_balanced,
        sha256,
        stable_seed,
        validate_foundation,
    )


MANIFEST_NAMES = (
    "torgo_train",
    "torgo_dev",
    "torgo_test",
    "normal_train",
    "normal_dev",
    "normal_test",
    "personal",
    "command_train",
    "command_dev",
    "command_test",
    "command_diagnostic",
)
CHECKPOINT_FORMAT = "echora-qwen-command-v3"


def audit_data(config: dict, manifests: dict[str, list[dict]]) -> dict:
    required = {"id", "speaker", "condition", "duration", "text", "audio_filepath"}
    for name in MANIFEST_NAMES:
        for row in manifests[name]:
            missing = required - set(row)
            if missing:
                raise ValueError(f"{name}:{row.get('id', '?')} missing {sorted(missing)}")
            if not Path(row["audio_filepath"]).is_file():
                raise FileNotFoundError(row["audio_filepath"])
            if not 0 < float(row["duration"]) <= float(config["maximum_audio_seconds"]):
                raise ValueError(f"Invalid duration: {name}:{row['id']}")
            if not normalize(str(row["text"])):
                raise ValueError(f"Empty normalized transcript: {name}:{row['id']}")

    train_speakers = {
        row["speaker"] for row in manifests["torgo_train"] if row["condition"] == "dysarthric"
    }
    dev_speakers = {row["speaker"] for row in manifests["torgo_dev"]}
    test_speakers = {row["speaker"] for row in manifests["torgo_test"]}
    if dev_speakers != {config["development_speaker"]}:
        raise ValueError("Unexpected TORGO development speaker")
    if test_speakers != {config["outer_test_speaker"]}:
        raise ValueError("Unexpected TORGO test speaker")
    if train_speakers & (dev_speakers | test_speakers) or dev_speakers & test_speakers:
        raise ValueError("TORGO speakers overlap")
    command_sets = {
        name: manifests[f"command_{name}"] for name in ("train", "dev", "test")
    }
    command_speakers = {name: {row["speaker"] for row in rows} for name, rows in command_sets.items()}
    if command_speakers != {"train": train_speakers, "dev": dev_speakers, "test": test_speakers}:
        raise ValueError("Command speakers do not mirror TORGO speaker splits")
    phrases = {name: {normalize(row["text"]) for row in rows} for name, rows in command_sets.items()}
    if any(phrases[a] & phrases[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Command phrases overlap across splits")
    if not all(row.get("synthetic_composition") is True for rows in command_sets.values() for row in rows):
        raise ValueError("Command manifests must disclose synthetic composition")
    all_ids = [row["id"] for name in MANIFEST_NAMES[:-1] for row in manifests[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Training/evaluation manifest IDs overlap")
    normal_speakers = [
        {row["speaker"] for row in manifests[name]}
        for name in ("normal_train", "normal_dev", "normal_test")
    ]
    if any(normal_speakers[a] & normal_speakers[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Normal speakers overlap")
    return {
        "dysarthric_train_speakers": sorted(train_speakers),
        "dysarthric_train_utterances": sum(
            row["condition"] == "dysarthric" for row in manifests["torgo_train"]
        ),
        "command_splits": {
            name: {
                "utterances": len(rows),
                "speakers": sorted(command_speakers[name]),
                "phrases": len(phrases[name]),
            }
            for name, rows in command_sets.items()
        },
        "speaker_disjoint": True,
        "command_phrase_disjoint": True,
        "personal_training_use": False,
    }


def to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.to("cuda", non_blocking=True) for name, value in batch.items()}


def prepare_training_example(
    processor,
    row: dict,
    prompt: str,
    augmentation: AugmentConfig,
    apply_probability: float,
    epoch: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    audio = load_audio(row)
    if str(row["condition"]).startswith("dysarthric"):
        rng = np.random.default_rng(stable_seed(seed, epoch, row["id"], "augment"))
        if rng.random() < apply_probability:
            audio = augment(audio, 16_000, rng, augmentation)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
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
    batch = processor.apply_chat_template(
        [messages],
        tokenize=True,
        return_dict=True,
        processor_kwargs={"output_labels": True, "sampling_rate": 16_000},
    )
    return {name: value for name, value in batch.items() if torch.is_tensor(value)}


def forward_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(**to_cuda(batch)).loss
    if loss is None or not torch.isfinite(loss):
        raise FloatingPointError("Qwen command loss is non-finite")
    return loss


def command_count(config: dict, epoch: int) -> int:
    curriculum = [int(value) for value in config["command_examples_per_epoch"]]
    return curriculum[min(epoch - 1, len(curriculum) - 1)]


def epoch_schedule(manifests: dict[str, list[dict]], config: dict, epoch: int) -> list[dict]:
    dysarthric = [
        row for row in manifests["torgo_train"] if row["condition"] == "dysarthric"
    ]
    controls = [row for row in manifests["torgo_train"] if row["condition"] == "control"]
    seed = stable_seed(int(config["seed"]), epoch, "command-v3-schedule")
    schedule = list(dysarthric)
    schedule.extend(
        select_coverage_slice(
            manifests["command_train"], command_count(config, epoch), epoch, seed ^ 0xC0A11D
        )
    )
    schedule.extend(
        select_coverage_slice(
            controls, int(config["torgo_control_examples_per_epoch"]), epoch, seed ^ 0xC011
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
        raise AssertionError("Every original dysarthric row must appear once per epoch")
    return schedule


def transcribe(model, processor, audio: np.ndarray, prompt: str | None, max_tokens: int) -> str:
    inputs = processor.apply_transcription_request(
        audio=[audio], language=["English"], prompt=prompt
    ).to(model.device, model.dtype)
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
    finally:
        model.config.use_cache = previous_cache
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.decode(generated, return_format="transcription_only")[0].strip()


def score_predictions(rows: list[dict], hypotheses: list[str]) -> dict:
    scored = [score(row["text"], hypothesis) for row, hypothesis in zip(rows, hypotheses)]
    total = aggregate(scored)
    by_speaker: dict[str, list] = defaultdict(list)
    by_slice: dict[str, list] = defaultdict(list)
    for row, item in zip(rows, scored):
        by_speaker[row["speaker"]].append(item)
        by_slice[row.get("slice", "unspecified")].append(item)
    exact = [normalize(row["text"]) == normalize(hypothesis) for row, hypothesis in zip(rows, hypotheses)]
    return {
        "utterances": len(rows),
        "micro_wer": total.wer,
        "micro_cer": total.cer,
        "exact_match_rate": sum(exact) / len(exact),
        "word_substitutions": total.word_substitutions,
        "word_deletions": total.word_deletions,
        "word_insertions": total.word_insertions,
        "reference_words": total.reference_words,
        "deletion_rate": total.word_deletions / total.reference_words,
        "empty_outputs": sum(not normalize(value) for value in hypotheses),
        "empty_rate": sum(not normalize(value) for value in hypotheses) / len(hypotheses),
        "speaker_wer": {
            speaker: aggregate(values).wer for speaker, values in sorted(by_speaker.items())
        },
        "slice_wer": {
            name: aggregate(values).wer for name, values in sorted(by_slice.items())
        },
    }


def evaluate(
    model,
    processor,
    rows: list[dict],
    output: Path,
    prompt: str | None,
    max_tokens: int,
    label: str,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions, elapsed = [], []
    temporary = output.with_suffix(output.suffix + ".tmp")
    model.eval()
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "id", "speaker", "condition", "slice", "audio_filepath", "audio_seconds",
            "decode_seconds", "reference", "transcript",
        ]
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            audio = load_audio(row)
            started = time.perf_counter()
            hypothesis = transcribe(model, processor, audio, prompt, max_tokens)
            duration = time.perf_counter() - started
            predictions.append(hypothesis)
            elapsed.append(duration)
            writer.writerow(
                {
                    "id": row["id"],
                    "speaker": row["speaker"],
                    "condition": row["condition"],
                    "slice": row.get("slice", "unspecified"),
                    "audio_filepath": row["audio_filepath"],
                    "audio_seconds": f"{len(audio) / 16000:.4f}",
                    "decode_seconds": f"{duration:.4f}",
                    "reference": row["text"],
                    "transcript": hypothesis,
                }
            )
            handle.flush()
            if index % 50 == 0 or index == len(rows):
                print(f"eval {label}: {index}/{len(rows)}", flush=True)
    temporary.replace(output)
    metrics = score_predictions(rows, predictions)
    metrics["decode_seconds"] = sum(elapsed)
    metrics["audio_seconds"] = sum(float(row["duration"]) for row in rows)
    metrics["rtf"] = metrics["decode_seconds"] / metrics["audio_seconds"]
    metrics["predictions_sha256"] = sha256(output)
    print(
        f"eval {label}: WER={metrics['micro_wer']:.2%} CER={metrics['micro_cer']:.2%} "
        f"exact={metrics['exact_match_rate']:.2%}",
        flush=True,
    )
    return metrics


def nbest(model, processor, audio, prompt: str | None, beams: int, max_tokens: int) -> list[dict]:
    inputs = processor.apply_transcription_request(
        audio=[audio], language=["English"], prompt=prompt
    ).to(model.device, model.dtype)
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=beams,
                num_return_sequences=beams,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        model.config.use_cache = previous_cache
    ids = generated.sequences[:, inputs["input_ids"].shape[1] :]
    texts = processor.decode(ids, return_format="transcription_only")
    scores = generated.sequences_scores.detach().float().cpu()
    probabilities = torch.softmax(scores, dim=0).tolist()
    output, seen = [], set()
    for text, value, probability in zip(texts, scores.tolist(), probabilities):
        cleaned = text.strip()
        key = normalize(cleaned)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "text": cleaned,
                "sequence_score": value,
                "relative_beam_probability": probability,
            }
        )
    return output


def summarize_nbest(rows: list[dict]) -> dict:
    top1, oracle, exact_topk = [], [], 0
    for row in rows:
        hypotheses = json.loads(row["hypotheses"])
        texts = [item["text"] for item in hypotheses] or [""]
        values = [score(row["reference"], text) for text in texts]
        top1.append(values[0])
        oracle.append(min(values, key=lambda item: (item.word_errors, item.character_errors)))
        exact_topk += any(normalize(row["reference"]) == normalize(text) for text in texts)
    top1_total, oracle_total = aggregate(top1), aggregate(oracle)
    return {
        "utterances": len(rows),
        "top1_wer": top1_total.wer,
        "top1_cer": top1_total.cer,
        "topk_oracle_wer": oracle_total.wer,
        "topk_oracle_cer": oracle_total.cer,
        "exact_in_topk_rate": exact_topk / len(rows),
    }


def diagnostic_category(row: dict) -> str:
    value = row.get("slice", "unspecified")
    if value.startswith("command_"):
        return "command_all"
    if value.startswith("torgo_"):
        return "torgo_original"
    if value.startswith("pause_"):
        return "pause_stress"
    if value == "personal_original":
        return "personal_original"
    return value


def evaluate_nbest(
    model,
    processor,
    rows: list[dict],
    output: Path,
    prompt: str | None,
    beams: int,
    max_tokens: int,
    label: str,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    recorded = []
    model.eval()
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = ["id", "speaker", "slice", "category", "audio_filepath", "reference", "top1", "hypotheses"]
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            values = nbest(model, processor, load_audio(row), prompt, beams, max_tokens)
            item = {
                "id": row["id"],
                "speaker": row["speaker"],
                "slice": row.get("slice", "unspecified"),
                "category": diagnostic_category(row),
                "audio_filepath": row["audio_filepath"],
                "reference": row["text"],
                "top1": values[0]["text"] if values else "",
                "hypotheses": json.dumps(values, ensure_ascii=False),
            }
            writer.writerow(item)
            handle.flush()
            recorded.append(item)
            if index % 25 == 0 or index == len(rows):
                print(f"diagnostic {label}: {index}/{len(rows)}", flush=True)
    temporary.replace(output)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in recorded:
        grouped[row["category"]].append(row)
    summary = {
        "beams": beams,
        "overall": summarize_nbest(recorded),
        "categories": {name: summarize_nbest(values) for name, values in sorted(grouped.items())},
        "slices": {
            name: summarize_nbest([row for row in recorded if row["slice"] == name])
            for name in sorted({row["slice"] for row in recorded})
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def make_optimizer(model, groups: dict, config: dict):
    parameters = dict(model.named_parameters())
    return torch.optim.AdamW(
        [
            {
                "params": [parameters[name] for name in groups["audio"]],
                "lr": float(config["audio_learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
            },
            {
                "params": [parameters[name] for name in groups["lora"]],
                "lr": float(config["decoder_lora_learning_rate"]),
                "weight_decay": 0.0,
            },
        ],
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


def save_checkpoint(path: Path, model, optimizer, scheduler, identity: dict, state: dict) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        **identity,
        **state,
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
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError("Unknown command-v3 checkpoint format")
    for name, value in identity.items():
        if saved.get(name) != value:
            raise RuntimeError(f"Checkpoint {name} does not match this run")
    load_partial_state(model, saved["trainable_state"], exact=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    optimizer_to_cuda(optimizer)
    scheduler.load_state_dict(saved["scheduler_state"])
    torch.set_rng_state(saved["torch_rng_state"])
    torch.cuda.set_rng_state(saved["cuda_rng_state"])
    return saved


def guard(candidate: dict, baseline: dict, config: dict) -> tuple[bool, list[str]]:
    failures = []
    if candidate["torgo_dev"]["micro_wer"] > baseline["torgo_dev"]["micro_wer"] + float(
        config["maximum_torgo_dev_wer_degradation_absolute"]
    ):
        failures.append("TORGO-dev WER")
    if candidate["normal_dev"]["micro_wer"] > baseline["normal_dev"]["micro_wer"] + float(
        config["maximum_normal_dev_wer_degradation_absolute"]
    ):
        failures.append("normal-dev WER")
    if candidate["torgo_dev"]["deletion_rate"] > baseline["torgo_dev"]["deletion_rate"] + float(
        config["maximum_dev_deletion_rate_increase_absolute"]
    ):
        failures.append("TORGO-dev deletion rate")
    command_relative = (
        baseline["command_dev"]["micro_wer"] - candidate["command_dev"]["micro_wer"]
    ) / max(baseline["command_dev"]["micro_wer"], 1e-9)
    if command_relative < float(config["required_relative_command_dev_wer_improvement"]):
        failures.append("command-dev improvement")
    return not failures, failures


def objective(metrics: dict) -> float:
    return (
        metrics["command_dev"]["micro_wer"]
        + 0.50 * metrics["torgo_dev"]["micro_wer"]
        + 0.10 * metrics["normal_dev"]["micro_wer"]
    )


def personal_i_water(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if normalize(row["reference"]) == "i water" and row["slice"] == "personal_original":
                values = json.loads(row["hypotheses"])
                texts = [item["text"] for item in values]
                return {
                    "reference": row["reference"],
                    "top1": texts[0] if texts else "",
                    "top5": texts,
                    "literal_present": any(normalize(value) == "i water" for value in texts),
                }
    raise RuntimeError("Protected I-water row is missing from diagnostic")


def train(
    config_path: Path,
    data_dir: Path,
    model_dir: Path,
    v1_adapter: Path,
    output: Path,
    resume: Path | None,
) -> dict:
    config = read_json(config_path)
    manifest_paths = {name: data_dir / "manifests" / f"{name}.jsonl" for name in MANIFEST_NAMES}
    manifests = {name: read_jsonl(path) for name, path in manifest_paths.items()}
    data_audit = audit_data(config, manifests)
    validate_foundation(config, model_dir)
    if sha256(v1_adapter) != config["v1_adapter_sha256"]:
        raise RuntimeError("Verified v1 baseline adapter SHA-256 mismatch")

    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions"
    diagnostics = output / "diagnostics"
    predictions.mkdir(exist_ok=True)
    diagnostics.mkdir(exist_ok=True)
    identity = {
        "run_id": config["run_id"],
        "base_model_sha256": config["model_weight_sha256"],
        "v1_adapter_sha256": config["v1_adapter_sha256"],
        "config_sha256": sha256(config_path),
        "manifest_sha256": combined_manifest_sha(manifest_paths),
    }
    (output / "run_manifest.json").write_text(
        json.dumps({**identity, "audit": data_audit, "config": config}, indent=2) + "\n",
        encoding="utf-8",
    )

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    prompt = str(config["literal_prompt"])
    max_tokens = int(config["generation_max_new_tokens"])
    print(f"run       {config['run_id']}", flush=True)
    print(f"gpu       {torch.cuda.get_device_name(0)}", flush=True)
    print("init      exact official foundation; v1 is baseline only", flush=True)

    processor, model, parameter_groups = load_model(config, model_dir)
    foundation_state = partial_state(model)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    accumulation = int(config["gradient_accumulation_steps"])
    total_steps = sum(
        math.ceil(len(epoch_schedule(manifests, config, epoch)) / accumulation)
        for epoch in range(1, int(config["maximum_epochs"]) + 1)
    )
    optimizer = make_optimizer(model, parameter_groups, config)
    scheduler = make_scheduler(optimizer, total_steps, float(config["warmup_ratio"]))

    state: dict[str, Any] = {
        "next_epoch": 1,
        "next_micro_index": 0,
        "global_step": 0,
        "history": [],
        "best_candidate": None,
        "best_deployable": None,
        "epochs_without_improvement": 0,
    }
    if resume is not None:
        saved = restore_checkpoint(resume, model, optimizer, scheduler, identity)
        for key in state:
            state[key] = saved[key]
        print(
            f"resume    epoch={state['next_epoch']} index={state['next_micro_index']} "
            f"step={state['global_step']}",
            flush=True,
        )

    baseline_path = output / "baseline.json"
    if baseline_path.is_file():
        baseline = read_json(baseline_path)
        if any(baseline.get(name) != value for name, value in identity.items()):
            raise RuntimeError("Existing baseline belongs to another command-v3 run")
    else:
        print("baseline  foundation with literal prompt", flush=True)
        foundation = {
            "torgo_dev": evaluate(model, processor, manifests["torgo_dev"], predictions / "foundation_torgo_dev.tsv", prompt, max_tokens, "foundation/torgo-dev"),
            "command_dev": evaluate(model, processor, manifests["command_dev"], predictions / "foundation_command_dev.tsv", prompt, max_tokens, "foundation/command-dev"),
            "normal_dev": evaluate(model, processor, manifests["normal_dev"], predictions / "foundation_normal_dev.tsv", prompt, max_tokens, "foundation/normal-dev"),
        }
        load_safetensors_adapter(model, v1_adapter, exact=False)
        print("baseline  verified v1 without prompt", flush=True)
        v1_no_prompt = {
            "torgo_dev": evaluate(model, processor, manifests["torgo_dev"], predictions / "v1_no_prompt_torgo_dev.tsv", None, max_tokens, "v1-no-prompt/torgo-dev"),
            "command_dev": evaluate(model, processor, manifests["command_dev"], predictions / "v1_no_prompt_command_dev.tsv", None, max_tokens, "v1-no-prompt/command-dev"),
            "normal_dev": evaluate(model, processor, manifests["normal_dev"], predictions / "v1_no_prompt_normal_dev.tsv", None, max_tokens, "v1-no-prompt/normal-dev"),
        }
        print("baseline  verified v1 with literal prompt", flush=True)
        v1_prompt = {
            "torgo_dev": evaluate(model, processor, manifests["torgo_dev"], predictions / "v1_prompt_torgo_dev.tsv", prompt, max_tokens, "v1-prompt/torgo-dev"),
            "command_dev": evaluate(model, processor, manifests["command_dev"], predictions / "v1_prompt_command_dev.tsv", prompt, max_tokens, "v1-prompt/command-dev"),
            "normal_dev": evaluate(model, processor, manifests["normal_dev"], predictions / "v1_prompt_normal_dev.tsv", prompt, max_tokens, "v1-prompt/normal-dev"),
        }
        load_partial_state(model, foundation_state, exact=True)
        baseline = {**identity, "foundation_prompt": foundation, "v1_no_prompt": v1_no_prompt, "v1_prompt": v1_prompt}
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    latest_path = output / "checkpoint-latest.pt"
    candidate_path = output / "checkpoint-best-candidate.pt"
    deployable_path = output / "checkpoint-best-deployable.pt"
    augmentation_values = dict(config["augmentation"])
    apply_probability = float(augmentation_values.pop("apply_probability"))
    augmentation = AugmentConfig(**augmentation_values)

    def run_epoch(epoch: int, start_index: int) -> dict:
        schedule = epoch_schedule(manifests, config, epoch)
        losses = []
        print(
            f"epoch {epoch} examples={len(schedule)} commands={command_count(config, epoch)} "
            f"resume_index={start_index}",
            flush=True,
        )
        for chunk_start in range(start_index, len(schedule), accumulation):
            chunk = schedule[chunk_start : chunk_start + accumulation]
            set_training_mode(model, config)
            optimizer.zero_grad(set_to_none=True)
            for row in chunk:
                batch = prepare_training_example(
                    processor, row, prompt, augmentation, apply_probability, epoch, seed
                )
                loss = forward_loss(model, batch)
                losses.append(float(loss.detach().cpu()))
                # Backpropagate each micro-example immediately. Keeping every
                # loss graph alive until the end of the chunk multiplies peak
                # activation memory by `accumulation` and can exhaust an L40S.
                # Parameters are not updated until the complete chunk has been
                # accumulated, so this is the same effective mean-loss batch.
                (loss / len(chunk)).backward()
                del loss, batch
            grad = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config["maximum_gradient_norm"]),
            )
            if not torch.isfinite(grad):
                raise FloatingPointError("Non-finite command-v3 gradient")
            optimizer.step()
            scheduler.step()
            state["global_step"] += 1
            state["next_micro_index"] = min(chunk_start + len(chunk), len(schedule))
            if state["global_step"] % 10 == 0:
                print(
                    f"step {state['global_step']}/{total_steps} loss={sum(losses[-len(chunk):]) / len(chunk):.4f} "
                    f"grad={float(grad):.4f}",
                    flush=True,
                )
            if state["global_step"] % int(config["checkpoint_every_optimizer_steps"]) == 0:
                save_checkpoint(latest_path, model, optimizer, scheduler, identity, state)
                print(f"saved     {latest_path}", flush=True)
        return {"training_mean_loss": sum(losses) / len(losses), "training_examples": len(schedule)}

    stop_reason = "maximum_epochs"
    for epoch in range(int(state["next_epoch"]), int(config["maximum_epochs"]) + 1):
        start_index = int(state["next_micro_index"]) if epoch == int(state["next_epoch"]) else 0
        metrics = run_epoch(epoch, start_index)
        metrics.update(
            {
                "epoch": epoch,
                "torgo_dev": evaluate(model, processor, manifests["torgo_dev"], predictions / f"epoch_{epoch:02d}_torgo_dev.tsv", prompt, max_tokens, f"epoch-{epoch}/torgo-dev"),
                "command_dev": evaluate(model, processor, manifests["command_dev"], predictions / f"epoch_{epoch:02d}_command_dev.tsv", prompt, max_tokens, f"epoch-{epoch}/command-dev"),
                "normal_dev": evaluate(model, processor, manifests["normal_dev"], predictions / f"epoch_{epoch:02d}_normal_dev.tsv", prompt, max_tokens, f"epoch-{epoch}/normal-dev"),
            }
        )
        metrics["objective"] = objective(metrics)
        eligible, failures = guard(metrics, baseline["v1_no_prompt"], config)
        metrics["deployable_guard_passed"] = eligible
        metrics["deployable_guard_failures"] = failures
        state["history"].append(metrics)

        improved_candidate = state["best_candidate"] is None or metrics["objective"] < state["best_candidate"]["objective"]
        if improved_candidate:
            state["best_candidate"] = metrics
            state["epochs_without_improvement"] = 0
            save_checkpoint(candidate_path, model, optimizer, scheduler, identity, state)
            print(f"selected  candidate epoch {epoch}", flush=True)
        else:
            state["epochs_without_improvement"] += 1
        if eligible and (
            state["best_deployable"] is None or metrics["objective"] < state["best_deployable"]["objective"]
        ):
            state["best_deployable"] = metrics
            save_checkpoint(deployable_path, model, optimizer, scheduler, identity, state)
            print(f"selected  deployable epoch {epoch}", flush=True)
        else:
            print(f"guard     epoch {epoch} passed={eligible} failures={failures}", flush=True)

        state["next_epoch"] = epoch + 1
        state["next_micro_index"] = 0
        save_checkpoint(latest_path, model, optimizer, scheduler, identity, state)
        (output / "training_history.json").write_text(
            json.dumps(state["history"], indent=2) + "\n", encoding="utf-8"
        )
        if (
            epoch >= int(config["minimum_epochs"])
            and state["epochs_without_improvement"] >= int(config["early_stopping_patience"])
        ):
            stop_reason = "development_early_stopping"
            break

    selected_kind = "deployable" if state["best_deployable"] is not None else "candidate_only"
    selected_path = deployable_path if selected_kind == "deployable" else candidate_path
    selected_saved = torch.load(selected_path, map_location="cpu", weights_only=False)
    selected_metrics = (
        state["best_deployable"] if selected_kind == "deployable" else state["best_candidate"]
    )

    # Protected test evaluation starts only after development selection.
    load_partial_state(model, foundation_state, exact=True)
    load_safetensors_adapter(model, v1_adapter, exact=False)
    v1_test = {
        "torgo_test": evaluate(model, processor, manifests["torgo_test"], predictions / "v1_torgo_test.tsv", None, max_tokens, "v1/torgo-test"),
        "command_test": evaluate(model, processor, manifests["command_test"], predictions / "v1_command_test.tsv", None, max_tokens, "v1/command-test"),
        "normal_test": evaluate(model, processor, manifests["normal_test"], predictions / "v1_normal_test.tsv", None, max_tokens, "v1/normal-test"),
        "personal": evaluate(model, processor, manifests["personal"], predictions / "v1_personal.tsv", None, max_tokens, "v1/personal"),
    }
    beams = int(config["diagnostic_beams"])
    v1_diagnostic_path = diagnostics / "v1_top5.tsv"
    v1_diagnostic = evaluate_nbest(
        model, processor, manifests["command_diagnostic"], v1_diagnostic_path,
        None, beams, max_tokens, "v1",
    )

    load_partial_state(model, selected_saved["trainable_state"], exact=True)
    v3_test = {
        "torgo_test": evaluate(model, processor, manifests["torgo_test"], predictions / "v3_torgo_test.tsv", prompt, max_tokens, "v3/torgo-test"),
        "command_test": evaluate(model, processor, manifests["command_test"], predictions / "v3_command_test.tsv", prompt, max_tokens, "v3/command-test"),
        "normal_test": evaluate(model, processor, manifests["normal_test"], predictions / "v3_normal_test.tsv", prompt, max_tokens, "v3/normal-test"),
        "personal": evaluate(model, processor, manifests["personal"], predictions / "v3_personal.tsv", prompt, max_tokens, "v3/personal"),
    }
    v3_diagnostic_path = diagnostics / "v3_top5.tsv"
    v3_diagnostic = evaluate_nbest(
        model, processor, manifests["command_diagnostic"], v3_diagnostic_path,
        prompt, beams, max_tokens, "v3",
    )

    torgo_guard = v3_test["torgo_test"]["micro_wer"] <= v1_test["torgo_test"]["micro_wer"] + float(
        config["maximum_torgo_test_wer_degradation_absolute"]
    )
    normal_guard = v3_test["normal_test"]["micro_wer"] <= v1_test["normal_test"]["micro_wer"] + float(
        config["maximum_normal_test_wer_degradation_absolute"]
    )
    command_relative = (
        v1_test["command_test"]["micro_wer"] - v3_test["command_test"]["micro_wer"]
    ) / max(v1_test["command_test"]["micro_wer"], 1e-9)
    top5_gain = (
        v3_diagnostic["categories"]["command_all"]["exact_in_topk_rate"]
        - v1_diagnostic["categories"]["command_all"]["exact_in_topk_rate"]
    )
    command_pass = command_relative >= float(config["required_relative_command_test_wer_improvement"]) or top5_gain >= float(
        config["required_command_top5_exact_gain_absolute"]
    )
    decision = (
        "deploy_v3_literal_multi_hypothesis"
        if selected_kind == "deployable" and torgo_guard and normal_guard and command_pass
        else "keep_v1"
    )

    adapter_path = output / "adapter.safetensors"
    descriptor = save_adapter(
        adapter_path,
        model,
        {
            "format": "echora-qwen3-asr-command-v3",
            **identity,
            "selected_kind": selected_kind,
            "selected_epoch": selected_metrics["epoch"],
            "literal_prompt": prompt,
            "semantic_repair": False,
            "candidate_ranker": False,
        },
    )
    descriptor["adapter_sha256"] = sha256(adapter_path)
    (output / "adapter_config.json").write_text(
        json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
    )
    i_water = {
        "v1": personal_i_water(v1_diagnostic_path),
        "v3": personal_i_water(v3_diagnostic_path),
    }
    decision_summary = {
        "decision": decision,
        "selected_kind": selected_kind,
        "selected_epoch": selected_metrics["epoch"],
        "test_guards": {"torgo": torgo_guard, "normal": normal_guard, "command": command_pass},
        "command_test_relative_wer_improvement": command_relative,
        "command_top5_exact_gain_absolute": top5_gain,
        "v1_test": {name: values["micro_wer"] for name, values in v1_test.items()},
        "v3_test": {name: values["micro_wer"] for name, values in v3_test.items()},
        "v1_command_top5": v1_diagnostic["categories"]["command_all"],
        "v3_command_top5": v3_diagnostic["categories"]["command_all"],
        "personal_i_water": i_water,
        "limitations": [
            "TORGO has only eight dysarthric speakers in these train/dev/test folds.",
            "Command compositions are controlled augmentations, not naturally spoken commands.",
            "Beam probabilities are relative search scores, not calibrated confidence.",
            "No semantic repair, candidate ranker, or personal context was used.",
        ],
    }
    (output / "next_decision.json").write_text(
        json.dumps(decision_summary, indent=2) + "\n", encoding="utf-8"
    )
    share = "COMMAND_V3_DONE\n" + json.dumps(decision_summary, indent=2) + "\n"
    (output / "SHARE_THIS_WITH_CODEX.txt").write_text(share, encoding="utf-8")

    result = {
        "status": "completed",
        **identity,
        "gpu": torch.cuda.get_device_name(0),
        "stop_reason": stop_reason,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "baseline": baseline,
        "history": state["history"],
        "selected_kind": selected_kind,
        "selected_development": selected_metrics,
        "v1_test": v1_test,
        "v3_test": v3_test,
        "v1_diagnostic": v1_diagnostic,
        "v3_diagnostic": v3_diagnostic,
        "decision": decision_summary,
        "artifact": descriptor,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.monotonic() - started,
        "literal_transcripts_only": True,
        "semantic_repair": False,
        "candidate_ranker": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    del optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"COMMAND_V3_TRAINING_DONE: {output / 'result.json'}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--v1-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(
        arguments.config,
        arguments.data,
        arguments.model,
        arguments.v1_adapter,
        arguments.output,
        arguments.resume,
    )
