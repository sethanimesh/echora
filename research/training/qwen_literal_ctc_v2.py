"""Joint Qwen audio adaptation plus a literal character-CTC branch.

The autoregressive Qwen transcript and monotonic CTC transcript are evaluated
separately.  This module never repairs grammar, infers intent, or ranks either
output using personal context.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "research" / "benchmarks"))

try:
    from .augment import AugmentConfig
    from .literal_ctc import CTC_SYMBOLS, LiteralCTCHead, ctc_loss, greedy_decode
    from .qwen_torgo_pilot import (
        MANIFEST_NAMES,
        audit_data,
        combined_manifest_sha,
        cpu_copy,
        epoch_schedule,
        evaluate,
        load_audio,
        load_model,
        make_scheduler,
        optimizer_to_cuda,
        partial_state,
        prepare_training_example,
        read_json,
        read_jsonl,
        score_predictions,
        set_partial_training_mode,
        sha256,
        to_cuda,
        validate_foundation,
    )
except ImportError:
    from augment import AugmentConfig
    from literal_ctc import CTC_SYMBOLS, LiteralCTCHead, ctc_loss, greedy_decode
    from qwen_torgo_pilot import (
        MANIFEST_NAMES,
        audit_data,
        combined_manifest_sha,
        cpu_copy,
        epoch_schedule,
        evaluate,
        load_audio,
        load_model,
        make_scheduler,
        optimizer_to_cuda,
        partial_state,
        prepare_training_example,
        read_json,
        read_jsonl,
        score_predictions,
        set_partial_training_mode,
        sha256,
        to_cuda,
        validate_foundation,
    )


CHECKPOINT_FORMAT = "echora-qwen-literal-ctc-v2"


def load_initial_adapter(model, path: Path, expected_sha: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required v1 adapter is missing: {path}")
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"v1 adapter SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    state = load_file(path, device="cpu")
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(state) != set(trainable):
        missing = sorted(set(trainable) - set(state))
        unexpected = sorted(set(state) - set(trainable))
        raise RuntimeError(
            f"v1 adapter parameter mismatch; missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameter = trainable[name]
            if parameter.shape != value.shape:
                raise RuntimeError(f"v1 adapter shape mismatch: {name}")
            parameter.copy_(value.to(parameter.device, parameter.dtype))


def make_optimizer(model, head, config: dict):
    return torch.optim.AdamW(
        [
            {
                "params": [item for item in model.parameters() if item.requires_grad],
                "lr": float(config["encoder_learning_rate"]),
            },
            {
                "params": list(head.parameters()),
                "lr": float(config["ctc_head_learning_rate"]),
            },
        ],
        weight_decay=float(config["weight_decay"]),
        fused=True,
    )


def save_checkpoint(path: Path, model, head, optimizer, scheduler, metadata: dict) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        **metadata,
        "trainable_state": partial_state(model),
        "ctc_state": {
            name: value.detach().cpu().clone() for name, value in head.state_dict().items()
        },
        "optimizer_state": cpu_copy(optimizer.state_dict()),
        "scheduler_state": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path: Path, model, head, optimizer, scheduler, identity: dict) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Unknown checkpoint format in {path}")
    for key, value in identity.items():
        if saved.get(key) != value:
            raise RuntimeError(f"Checkpoint {key} does not match this run")
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != set(saved["trainable_state"]):
        raise RuntimeError("Checkpoint Qwen parameter names changed")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(saved["trainable_state"][name].to(parameter.device, parameter.dtype))
    head.load_state_dict(saved["ctc_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
    optimizer_to_cuda(optimizer)
    scheduler.load_state_dict(saved["scheduler_state"])
    torch.set_rng_state(saved["torch_rng_state"])
    torch.cuda.set_rng_state(saved["cuda_rng_state"])
    return saved


def ctc_features(model, batch: dict[str, torch.Tensor], require_grad: bool):
    inputs = to_cuda(batch)
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.model.get_audio_features(
            inputs["input_features"], inputs["input_features_mask"], return_dict=True
        )
    return output.pooler_output


def train_head_example(model, head, batch, transcript, loss_function):
    features = ctc_features(model, batch, require_grad=False)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = head(features)
    return ctc_loss(logits.float(), transcript, loss_function)


def train_joint_example(model, head, batch, transcript, loss_function, weight: float):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**to_cuda(batch))
        qwen_loss = output.loss
        logits = head(output.audio_hidden_states)
    if qwen_loss is None or not torch.isfinite(qwen_loss):
        raise FloatingPointError("Qwen loss is non-finite")
    acoustic_loss, input_length, target_length = ctc_loss(
        logits.float(), transcript, loss_function
    )
    total = qwen_loss if acoustic_loss is None else qwen_loss + weight * acoustic_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Joint loss is non-finite")
    return total, qwen_loss, acoustic_loss, input_length, target_length


def evaluate_ctc(model, processor, head, rows: list[dict], output: Path, label: str) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions, elapsed_values = [], []
    temporary = output.with_suffix(output.suffix + ".tmp")
    model.eval()
    head.eval()
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
            inputs = processor.apply_transcription_request(
                audio=[audio], language=["English"]
            )
            features = inputs["input_features"].to("cuda")
            mask = inputs["input_features_mask"].to("cuda")
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                encoded = model.model.get_audio_features(
                    features, mask, return_dict=True
                ).pooler_output
                logits = head(encoded)
            transcript = greedy_decode(logits)
            elapsed = time.perf_counter() - started
            predictions.append(transcript)
            elapsed_values.append(elapsed)
            writer.writerow(
                {
                    "id": row["id"],
                    "speaker": row["speaker"],
                    "condition": row["condition"],
                    "audio_filepath": row["audio_filepath"],
                    "audio_seconds": f"{len(audio) / 16000:.4f}",
                    "decode_seconds": f"{elapsed:.4f}",
                    "reference": row["text"],
                    "transcript": transcript,
                }
            )
            handle.flush()
            if index % 50 == 0 or index == len(rows):
                print(f"eval {label}: {index}/{len(rows)}", flush=True)
    temporary.replace(output)
    metrics = score_predictions(rows, predictions)
    metrics["decode_seconds"] = sum(elapsed_values)
    metrics["audio_seconds"] = sum(float(row["duration"]) for row in rows)
    metrics["rtf"] = metrics["decode_seconds"] / metrics["audio_seconds"]
    metrics["predictions_sha256"] = sha256(output)
    print(
        f"eval {label}: WER={metrics['micro_wer']:.2%} CER={metrics['micro_cer']:.2%}",
        flush=True,
    )
    return metrics


def qwen_guard(candidate: dict, baseline: dict, config: dict) -> tuple[bool, list[str]]:
    reasons = []
    if candidate["qwen_torgo_dev"]["micro_wer"] > (
        baseline["qwen_torgo_dev"]["micro_wer"]
        + float(config["maximum_qwen_dev_wer_degradation_absolute"])
    ):
        reasons.append("Qwen TORGO-dev WER")
    if candidate["qwen_normal_dev"]["micro_wer"] > (
        baseline["qwen_normal_dev"]["micro_wer"]
        + float(config["maximum_normal_dev_wer_degradation_absolute"])
    ):
        reasons.append("Qwen normal-dev WER")
    if candidate["qwen_torgo_dev"]["deletion_rate"] > (
        baseline["qwen_torgo_dev"]["deletion_rate"]
        + float(config["maximum_dev_deletion_rate_increase_absolute"])
    ):
        reasons.append("Qwen TORGO-dev deletion rate")
    if candidate["qwen_torgo_dev"]["empty_rate"] > (
        baseline["qwen_torgo_dev"]["empty_rate"]
        + float(config["maximum_dev_empty_rate_increase_absolute"])
    ):
        reasons.append("Qwen TORGO-dev empty rate")
    return not reasons, reasons


def export_artifacts(output: Path, model, head, config: dict, identity: dict, best: dict):
    qwen_path = output / "adapter.safetensors"
    qwen_state = {name: value.contiguous() for name, value in partial_state(model).items()}
    metadata = {
        "format": "echora-qwen3-asr-literal-ctc-v2",
        "run_id": config["run_id"],
        "base_model_sha256": config["model_weight_sha256"],
        "initial_adapter_sha256": config["initial_adapter_sha256"],
        "selected_phase": str(best["phase"]),
        "selected_epoch": str(best["epoch"]),
    }
    qwen_temporary = qwen_path.with_suffix(qwen_path.suffix + ".tmp")
    save_file(qwen_state, qwen_temporary, metadata=metadata)
    qwen_temporary.replace(qwen_path)
    ctc_path = output / "ctc_head.safetensors"
    ctc_temporary = ctc_path.with_suffix(ctc_path.suffix + ".tmp")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()},
        ctc_temporary,
        metadata={**metadata, "symbols": json.dumps(CTC_SYMBOLS)},
    )
    ctc_temporary.replace(ctc_path)
    descriptor = {
        **metadata,
        **identity,
        "qwen_adapter": str(qwen_path),
        "qwen_adapter_sha256": sha256(qwen_path),
        "ctc_head": str(ctc_path),
        "ctc_head_sha256": sha256(ctc_path),
        "ctc_symbols": CTC_SYMBOLS,
        "semantic_repair": False,
        "candidate_ranker": False,
    }
    (output / "artifact_config.json").write_text(
        json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
    )
    return descriptor


def train(
    config_path: Path,
    data_dir: Path,
    model_dir: Path,
    initial_adapter: Path,
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
    if sha256(initial_adapter) != config["initial_adapter_sha256"]:
        raise RuntimeError("Initial v1 adapter is not the frozen, verified artifact")

    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions"
    predictions.mkdir(exist_ok=True)
    identity = {
        "run_id": config["run_id"],
        "base_model_sha256": config["model_weight_sha256"],
        "initial_adapter_sha256": config["initial_adapter_sha256"],
        "config_sha256": sha256(config_path),
        "manifest_sha256": combined_manifest_sha(manifest_paths),
    }
    (output / "run_manifest.json").write_text(
        json.dumps({**identity, "audit": audit, "config": config}, indent=2) + "\n",
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
    print(f"run       {config['run_id']}", flush=True)
    print(f"gpu       {torch.cuda.get_device_name(0)}", flush=True)
    print(f"init      verified {initial_adapter}", flush=True)

    processor, model = load_model(config, model_dir)
    load_initial_adapter(model, initial_adapter, config["initial_adapter_sha256"])
    head = LiteralCTCHead(int(config["ctc_input_size"])).to("cuda")

    accumulation = int(config["gradient_accumulation_steps"])
    epoch_size = len(epoch_schedule(manifests, config, 1))
    total_epochs = int(config["head_warmup_epochs"]) + int(config["maximum_joint_epochs"])
    total_steps = math.ceil(epoch_size / accumulation) * total_epochs
    optimizer = make_optimizer(model, head, config)
    scheduler = make_scheduler(optimizer, total_steps, float(config["warmup_ratio"]))
    loss_function = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)

    augmentation_values = dict(config["augmentation"])
    apply_probability = float(augmentation_values.pop("apply_probability"))
    augmentation = AugmentConfig(**augmentation_values)
    state: dict[str, Any] = {
        "phase": "head_warmup",
        "next_epoch": 1,
        "next_micro_index": 0,
        "global_step": 0,
        "best": None,
        "epochs_without_improvement": 0,
        "history": [],
    }
    if resume is not None:
        saved = restore_checkpoint(resume, model, head, optimizer, scheduler, identity)
        for key in state:
            state[key] = saved[key]
        print(
            f"resume    phase={state['phase']} epoch={state['next_epoch']} "
            f"index={state['next_micro_index']}",
            flush=True,
        )

    baseline_path = output / "baseline_v1.json"
    max_tokens = int(config["generation_max_new_tokens"])
    if baseline_path.is_file():
        baseline = read_json(baseline_path)
        if any(baseline.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Existing v1 baseline belongs to a different v2 run")
    else:
        print("baseline  evaluating immutable v1 Qwen adapter", flush=True)
        baseline = {
            **identity,
            "qwen_torgo_dev": evaluate(
                model, processor, manifests["torgo_dev"],
                predictions / "v1_qwen_torgo_dev.tsv", max_tokens, "v1/qwen/torgo-dev"
            ),
            "qwen_normal_dev": evaluate(
                model, processor, manifests["normal_dev"],
                predictions / "v1_qwen_normal_dev.tsv", max_tokens, "v1/qwen/normal-dev"
            ),
            "qwen_torgo_test": evaluate(
                model, processor, manifests["torgo_test"],
                predictions / "v1_qwen_torgo_test.tsv", max_tokens, "v1/qwen/torgo-test"
            ),
            "qwen_normal_test": evaluate(
                model, processor, manifests["normal_test"],
                predictions / "v1_qwen_normal_test.tsv", max_tokens, "v1/qwen/normal-test"
            ),
            "qwen_personal": evaluate(
                model, processor, manifests["personal"],
                predictions / "v1_qwen_personal.tsv", max_tokens, "v1/qwen/personal"
            ),
        }
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    latest_path = output / "checkpoint-latest.pt"
    best_path = output / "checkpoint-best.pt"

    def run_training_epoch(phase: str, epoch: int, start_index: int) -> dict:
        schedule = epoch_schedule(manifests, config, epoch)
        qwen_values, ctc_values = [], []
        ctc_used = ctc_skipped = 0
        print(
            f"{phase} epoch={epoch} examples={len(schedule)} resume_index={start_index}",
            flush=True,
        )
        for chunk_start in range(start_index, len(schedule), accumulation):
            chunk = schedule[chunk_start : chunk_start + accumulation]
            model.eval() if phase == "head_warmup" else set_partial_training_mode(model, config)
            head.train()
            optimizer.zero_grad(set_to_none=True)
            differentiable = []
            for row in chunk:
                batch = prepare_training_example(
                    processor, row, augmentation, apply_probability, epoch, seed
                )
                if phase == "head_warmup":
                    acoustic, _, _ = train_head_example(
                        model, head, batch, row["text"], loss_function
                    )
                    if acoustic is not None:
                        differentiable.append(acoustic)
                        ctc_values.append(float(acoustic.detach().cpu()))
                        ctc_used += 1
                    else:
                        ctc_skipped += 1
                else:
                    total, qwen, acoustic, _, _ = train_joint_example(
                        model, head, batch, row["text"], loss_function,
                        float(config["ctc_loss_weight"]),
                    )
                    differentiable.append(total)
                    qwen_values.append(float(qwen.detach().cpu()))
                    if acoustic is None:
                        ctc_skipped += 1
                    else:
                        ctc_values.append(float(acoustic.detach().cpu()))
                        ctc_used += 1
                del batch
            if not differentiable:
                raise RuntimeError("A whole optimizer chunk had no valid CTC alignment")
            torch.stack(differentiable).mean().backward()
            parameters = list(head.parameters())
            if phase == "joint":
                parameters += [item for item in model.parameters() if item.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(config["maximum_gradient_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
            optimizer.step()
            scheduler.step()
            state["global_step"] += 1
            state["next_micro_index"] = min(chunk_start + len(chunk), len(schedule))
            if state["global_step"] % 10 == 0:
                print(
                    f"step {state['global_step']}/{total_steps} phase={phase} "
                    f"ctc={sum(ctc_values[-len(chunk):]) / max(min(len(ctc_values), len(chunk)), 1):.4f} "
                    f"grad={float(grad_norm):.4f}",
                    flush=True,
                )
            if state["global_step"] % int(config["checkpoint_every_optimizer_steps"]) == 0:
                save_checkpoint(latest_path, model, head, optimizer, scheduler, {**identity, **state})
                print(f"saved     {latest_path}", flush=True)
            del differentiable
        return {
            "qwen_training_mean_loss": None if not qwen_values else sum(qwen_values) / len(qwen_values),
            "ctc_training_mean_loss": sum(ctc_values) / len(ctc_values),
            "ctc_examples_used": ctc_used,
            "ctc_examples_skipped_unalignable": ctc_skipped,
        }

    if state["phase"] == "head_warmup":
        for epoch in range(int(state["next_epoch"]), int(config["head_warmup_epochs"]) + 1):
            start_index = int(state["next_micro_index"]) if epoch == int(state["next_epoch"]) else 0
            metrics = run_training_epoch("head_warmup", epoch, start_index)
            metrics.update(
                {
                    "phase": "head_warmup",
                    "epoch": epoch,
                    "ctc_torgo_dev": evaluate_ctc(
                        model, processor, head, manifests["torgo_dev"],
                        predictions / f"warmup_{epoch:02d}_ctc_torgo_dev.tsv",
                        f"warmup-{epoch}/ctc/torgo-dev",
                    ),
                    "ctc_normal_dev": evaluate_ctc(
                        model, processor, head, manifests["normal_dev"],
                        predictions / f"warmup_{epoch:02d}_ctc_normal_dev.tsv",
                        f"warmup-{epoch}/ctc/normal-dev",
                    ),
                    "selection_guard_passed": True,
                    "selection_guard_failures": [],
                }
            )
            state["history"].append(metrics)
            state["best"] = metrics
            state["next_epoch"] = epoch + 1
            state["next_micro_index"] = 0
            save_checkpoint(best_path, model, head, optimizer, scheduler, {**identity, **state})
            save_checkpoint(latest_path, model, head, optimizer, scheduler, {**identity, **state})
        state["phase"] = "joint"
        state["next_epoch"] = 1
        state["next_micro_index"] = 0
        save_checkpoint(latest_path, model, head, optimizer, scheduler, {**identity, **state})

    stop_reason = "maximum_joint_epochs"
    for epoch in range(int(state["next_epoch"]), int(config["maximum_joint_epochs"]) + 1):
        start_index = int(state["next_micro_index"]) if epoch == int(state["next_epoch"]) else 0
        metrics = run_training_epoch("joint", epoch, start_index)
        metrics.update(
            {
                "phase": "joint",
                "epoch": epoch,
                "qwen_torgo_dev": evaluate(
                    model, processor, manifests["torgo_dev"],
                    predictions / f"joint_{epoch:02d}_qwen_torgo_dev.tsv",
                    max_tokens, f"joint-{epoch}/qwen/torgo-dev",
                ),
                "qwen_normal_dev": evaluate(
                    model, processor, manifests["normal_dev"],
                    predictions / f"joint_{epoch:02d}_qwen_normal_dev.tsv",
                    max_tokens, f"joint-{epoch}/qwen/normal-dev",
                ),
                "ctc_torgo_dev": evaluate_ctc(
                    model, processor, head, manifests["torgo_dev"],
                    predictions / f"joint_{epoch:02d}_ctc_torgo_dev.tsv",
                    f"joint-{epoch}/ctc/torgo-dev",
                ),
                "ctc_normal_dev": evaluate_ctc(
                    model, processor, head, manifests["normal_dev"],
                    predictions / f"joint_{epoch:02d}_ctc_normal_dev.tsv",
                    f"joint-{epoch}/ctc/normal-dev",
                ),
            }
        )
        eligible, reasons = qwen_guard(metrics, baseline, config)
        metrics["selection_guard_passed"] = eligible
        metrics["selection_guard_failures"] = reasons
        state["history"].append(metrics)
        current_key = (
            metrics["ctc_torgo_dev"]["micro_wer"],
            metrics["ctc_torgo_dev"]["micro_cer"],
        )
        best_key = (
            state["best"]["ctc_torgo_dev"]["micro_wer"],
            state["best"]["ctc_torgo_dev"]["micro_cer"],
        )
        if eligible and current_key < best_key:
            state["best"] = metrics
            state["epochs_without_improvement"] = 0
            save_checkpoint(best_path, model, head, optimizer, scheduler, {**identity, **state})
            print(f"selected  joint epoch {epoch}", flush=True)
        else:
            state["epochs_without_improvement"] += 1
            print(f"not best  joint epoch {epoch}; guard={eligible} reasons={reasons}", flush=True)
        state["next_epoch"] = epoch + 1
        state["next_micro_index"] = 0
        save_checkpoint(latest_path, model, head, optimizer, scheduler, {**identity, **state})
        (output / "training_history.json").write_text(
            json.dumps(state["history"], indent=2) + "\n", encoding="utf-8"
        )
        if (
            epoch >= int(config["minimum_joint_epochs"])
            and state["epochs_without_improvement"] >= int(config["early_stopping_patience"])
        ):
            stop_reason = "development_early_stopping"
            break

    best_saved = torch.load(best_path, map_location="cpu", weights_only=False)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(best_saved["trainable_state"][name].to(parameter.device, parameter.dtype))
    head.load_state_dict(best_saved["ctc_state"])
    best = state["best"]
    final = {
        "qwen_torgo_test": evaluate(
            model, processor, manifests["torgo_test"],
            predictions / "v2_qwen_torgo_test.tsv", max_tokens, "v2/qwen/torgo-test"
        ),
        "qwen_normal_test": evaluate(
            model, processor, manifests["normal_test"],
            predictions / "v2_qwen_normal_test.tsv", max_tokens, "v2/qwen/normal-test"
        ),
        "qwen_personal": evaluate(
            model, processor, manifests["personal"],
            predictions / "v2_qwen_personal.tsv", max_tokens, "v2/qwen/personal"
        ),
        "ctc_torgo_test": evaluate_ctc(
            model, processor, head, manifests["torgo_test"],
            predictions / "v2_ctc_torgo_test.tsv", "v2/ctc/torgo-test"
        ),
        "ctc_normal_test": evaluate_ctc(
            model, processor, head, manifests["normal_test"],
            predictions / "v2_ctc_normal_test.tsv", "v2/ctc/normal-test"
        ),
        "ctc_personal": evaluate_ctc(
            model, processor, head, manifests["personal"],
            predictions / "v2_ctc_personal.tsv", "v2/ctc/personal"
        ),
    }
    artifact = export_artifacts(output, model, head, config, identity, best)
    result = {
        "status": "completed",
        **identity,
        "gpu": torch.cuda.get_device_name(0),
        "stop_reason": stop_reason,
        "baseline_v1": baseline,
        "selected_development": best,
        "final_evaluation": final,
        "artifact": artifact,
        "history": state["history"],
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.monotonic() - started,
        "literal_transcripts_only": True,
        "semantic_repair": False,
        "candidate_ranker": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    del optimizer, scheduler, model, head
    gc.collect()
    torch.cuda.empty_cache()
    print(f"LITERAL_V2_TRAINING_DONE: {output / 'result.json'}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(
        arguments.config,
        arguments.data,
        arguments.model,
        arguments.initial_adapter,
        arguments.output,
        arguments.resume,
    )
