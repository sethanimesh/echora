"""Bounded Qwen3-ASR partial-training, checkpoint, reload and resume probe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(config: dict, manifest: list[dict], model_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 2 requires CUDA; refusing a CPU or MPS fallback")
    gpu_name = torch.cuda.get_device_name(0)
    required_gpu = config["required_gpu_name_fragment"]
    if required_gpu.lower() not in gpu_name.lower():
        raise RuntimeError(f"Gate 2 requires {required_gpu}; detected {gpu_name}")

    weights = model_dir / config["model_weight_file"]
    if not weights.is_file():
        raise FileNotFoundError(f"Foundation weights not found at {weights}")
    actual_bytes = weights.stat().st_size
    if actual_bytes != int(config["model_weight_bytes"]):
        raise RuntimeError(
            f"Foundation size mismatch: expected {config['model_weight_bytes']}, got {actual_bytes}"
        )
    actual_sha = sha256(weights)
    if actual_sha != config["model_weight_sha256"]:
        raise RuntimeError(
            f"Foundation SHA-256 mismatch: expected {config['model_weight_sha256']}, got {actual_sha}"
        )

    purposes = [row.get("purpose") for row in manifest]
    if purposes.count("longest") != 1:
        raise ValueError("Probe manifest must contain exactly one longest row")
    if purposes.count("tiny") != int(config["tiny_subset_size"]):
        raise ValueError("Probe manifest tiny-subset count does not match the frozen config")
    excluded = {config["outer_test_speaker"], config["development_speaker"]}
    for row in manifest:
        if row["speaker"] in excluded:
            raise ValueError(f"Held-out speaker leaked into the probe: {row['speaker']}")
        if row["condition"] != config["training_condition"]:
            raise ValueError(f"Unexpected condition in probe: {row['condition']}")
        if float(row["duration"]) > float(config["maximum_probe_audio_seconds"]):
            raise ValueError(f"Over-length audio in probe: {row['id']}")
        if not Path(row["audio_filepath"]).is_file():
            raise FileNotFoundError(f"Staged audio is missing: {row['audio_filepath']}")


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


def configure_trainable_parameters(model, config: dict) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    layers = model.model.audio_tower.layers
    selected_layers = [int(index) for index in config["trainable_audio_layer_indices"]]
    if len(layers) != 24:
        raise RuntimeError(f"Expected 24 Qwen audio layers, found {len(layers)}")
    if selected_layers != [20, 21, 22, 23]:
        raise RuntimeError(f"Unexpected trainable audio layers: {selected_layers}")
    for index in selected_layers:
        for parameter in layers[index].parameters():
            parameter.requires_grad = True

    if config["train_multimodal_projector"] is not True:
        raise RuntimeError("The Gate 2 projector must be trainable")
    for parameter in model.model.multi_modal_projector.parameters():
        parameter.requires_grad = True

    trainable_names = [name for name, item in model.named_parameters() if item.requires_grad]
    allowed_prefixes = tuple(
        [f"model.audio_tower.layers.{index}." for index in selected_layers]
        + ["model.multi_modal_projector."]
    )
    unexpected = [name for name in trainable_names if not name.startswith(allowed_prefixes)]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:5]}")
    if not any(name.startswith("model.multi_modal_projector.") for name in trainable_names):
        raise RuntimeError("No multimodal projector parameters were made trainable")
    if not trainable_names:
        raise RuntimeError("No parameters were made trainable")
    return trainable_names


def set_partial_training_mode(model, config: dict) -> None:
    model.eval()
    for index in config["trainable_audio_layer_indices"]:
        model.model.audio_tower.layers[int(index)].train()
    model.model.multi_modal_projector.train()


def prepare_example(processor, row: dict) -> dict[str, torch.Tensor]:
    transcript = row["text"].strip()
    conversation = [
        [
            {
                "role": "user",
                "content": [{"type": "audio", "path": row["audio_filepath"]}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"language English<asr_text>{transcript}",
                    }
                ],
            },
        ]
    ]
    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        processor_kwargs={"output_labels": True},
    )
    return {key: value.cpu() for key, value in batch.items() if torch.is_tensor(value)}


def to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}


def forward_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(**to_cuda(batch)).loss
    if loss is None or not torch.isfinite(loss):
        value = None if loss is None else float(loss.detach().cpu())
        raise FloatingPointError(f"Non-finite loss: {value}")
    return loss


def mean_evaluation_loss(model, batches: list[dict[str, torch.Tensor]]) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in batches:
            values.append(float(forward_loss(model, batch).detach().cpu()))
    return sum(values) / len(values)


def gradient_is_finite(model) -> bool:
    found = False
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            found = True
            if not torch.isfinite(parameter.grad).all():
                return False
    return found


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


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


def save_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    config: dict,
    config_sha256: str,
    manifest_sha256: str,
    step: int,
    loss_history: list[float],
) -> None:
    trainable_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "format": "echora-qwen-gate2-partial-v1",
        "run_id": config["run_id"],
        "step": step,
        "base_model_id": config["model_id"],
        "base_model_revision": config["model_revision"],
        "base_model_sha256": config["model_weight_sha256"],
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "trainable_audio_layer_indices": config["trainable_audio_layer_indices"],
        "trainable_state": trainable_state,
        "optimizer_state": cpu_copy(optimizer.state_dict()),
        "loss_history": loss_history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path: Path, model, optimizer, config: dict) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "echora-qwen-gate2-partial-v1":
        raise RuntimeError("Unknown checkpoint format")
    if checkpoint.get("run_id") != config["run_id"]:
        raise RuntimeError("Checkpoint run ID does not match the frozen config")
    if checkpoint.get("base_model_sha256") != config["model_weight_sha256"]:
        raise RuntimeError("Checkpoint foundation SHA-256 does not match")

    trainable = {name: item for name, item in model.named_parameters() if item.requires_grad}
    saved = checkpoint["trainable_state"]
    if set(trainable) != set(saved):
        raise RuntimeError("Checkpoint trainable-parameter names do not match this model")
    with torch.no_grad():
        for name, parameter in trainable.items():
            if parameter.shape != saved[name].shape:
                raise RuntimeError(f"Checkpoint shape mismatch for {name}")
            parameter.copy_(saved[name].to(parameter.device, parameter.dtype))
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    optimizer_to_device(optimizer, "cuda")
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    return checkpoint


def make_optimizer(model, config: dict) -> torch.optim.Optimizer:
    parameters = [item for item in model.parameters() if item.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        fused=True,
    )


def training_step(model, optimizer, batch, config: dict) -> tuple[float, float]:
    set_partial_training_mode(model, config)
    optimizer.zero_grad(set_to_none=True)
    loss = forward_loss(model, batch)
    loss.backward()
    if not gradient_is_finite(model):
        raise FloatingPointError("Missing or non-finite gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(config["maximum_gradient_norm"]),
    )
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
    optimizer.step()
    return float(loss.detach().cpu()), float(grad_norm.detach().cpu())


def run(config_path: Path, manifest_path: Path, model_dir: Path, output_dir: Path) -> None:
    config = read_json(config_path)
    manifest = read_jsonl(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_inputs(config, manifest, model_dir)

    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    started = time.monotonic()
    print(f"run      {config['run_id']}", flush=True)
    print(f"model    {config['model_id']} @ {config['model_revision']}", flush=True)
    print(f"gpu      {torch.cuda.get_device_name(0)}", flush=True)
    print("loading  foundation in FP32; forward/backward uses BF16 autocast", flush=True)
    processor, model = load_model(config, model_dir)
    trainable_names = [name for name, item in model.named_parameters() if item.requires_grad]
    trainable_count = sum(item.numel() for item in model.parameters() if item.requires_grad)
    total_count = sum(item.numel() for item in model.parameters())
    print(
        f"trainable {trainable_count:,}/{total_count:,} parameters "
        f"({100 * trainable_count / total_count:.2f}%)",
        flush=True,
    )
    print("modules  audio layers 20-23 + multimodal projector; decoder frozen", flush=True)

    longest_row = next(row for row in manifest if row["purpose"] == "longest")
    tiny_rows = [row for row in manifest if row["purpose"] == "tiny"]
    longest_batch = prepare_example(processor, longest_row)
    tiny_batches = [prepare_example(processor, row) for row in tiny_rows]
    optimizer = make_optimizer(model, config)

    print(
        f"longest {longest_row['id']} ({float(longest_row['duration']):.2f}s): "
        "forward/backward only",
        flush=True,
    )
    set_partial_training_mode(model, config)
    optimizer.zero_grad(set_to_none=True)
    longest_loss = forward_loss(model, longest_batch)
    longest_loss.backward()
    if not gradient_is_finite(model):
        raise FloatingPointError("Longest-example backward produced invalid gradients")
    print(f"longest loss {float(longest_loss.detach().cpu()):.6f}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    del longest_batch, longest_loss
    torch.cuda.empty_cache()

    initial_loss = mean_evaluation_loss(model, tiny_batches)
    print(f"initial tiny-set mean loss {initial_loss:.6f}", flush=True)
    loss_history = []
    steps = int(config["overfit_steps_before_checkpoint"])
    for step in range(1, steps + 1):
        index = (step - 1) % len(tiny_batches)
        loss, grad_norm = training_step(model, optimizer, tiny_batches[index], config)
        loss_history.append(loss)
        print(
            f"step {step:02d}/{steps} speaker={tiny_rows[index]['speaker']} "
            f"loss={loss:.6f} grad_norm={grad_norm:.4f}",
            flush=True,
        )

    final_loss = mean_evaluation_loss(model, tiny_batches)
    relative_drop = (initial_loss - final_loss) / initial_loss
    print(
        f"final tiny-set mean loss {final_loss:.6f}; relative drop {relative_drop:.2%}",
        flush=True,
    )
    minimum_drop = float(config["minimum_relative_loss_drop"])
    if not math.isfinite(relative_drop) or relative_drop < minimum_drop:
        raise RuntimeError(
            f"Loss-drop gate failed: required {minimum_drop:.2%}, observed {relative_drop:.2%}"
        )

    config_digest = sha256(config_path)
    manifest_digest = sha256(manifest_path)
    first_checkpoint = output_dir / f"checkpoint-step-{steps}.pt"
    save_checkpoint(
        first_checkpoint,
        model,
        optimizer,
        config,
        config_digest,
        manifest_digest,
        steps,
        loss_history,
    )
    print(f"saved    {first_checkpoint}", flush=True)

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    print("reload   fresh foundation + partial checkpoint", flush=True)
    processor, model = load_model(config, model_dir)
    optimizer = make_optimizer(model, config)
    checkpoint = restore_checkpoint(first_checkpoint, model, optimizer, config)
    if int(checkpoint["step"]) != steps:
        raise RuntimeError("Reloaded checkpoint has the wrong step")

    resume_steps = int(config["resume_steps"])
    resume_losses = []
    for offset in range(1, resume_steps + 1):
        index = (steps + offset - 1) % len(tiny_batches)
        loss, grad_norm = training_step(model, optimizer, tiny_batches[index], config)
        resume_losses.append(loss)
        print(
            f"resume step {steps + offset} speaker={tiny_rows[index]['speaker']} "
            f"loss={loss:.6f} grad_norm={grad_norm:.4f}",
            flush=True,
        )

    final_step = steps + resume_steps
    resumed_checkpoint = output_dir / f"checkpoint-resumed-step-{final_step}.pt"
    save_checkpoint(
        resumed_checkpoint,
        model,
        optimizer,
        config,
        config_digest,
        manifest_digest,
        final_step,
        loss_history + resume_losses,
    )
    checkpoint_sha = sha256(resumed_checkpoint)
    elapsed = time.monotonic() - started
    result = {
        "status": "passed",
        "run_id": config["run_id"],
        "gpu": torch.cuda.get_device_name(0),
        "base_model_sha256": config["model_weight_sha256"],
        "config_sha256": config_digest,
        "manifest_sha256": manifest_digest,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "trainable_parameter_names": trainable_names,
        "initial_tiny_mean_loss": initial_loss,
        "final_tiny_mean_loss": final_loss,
        "relative_loss_drop": relative_drop,
        "resume_loss": resume_losses,
        "completed_step": final_step,
        "checkpoint": str(resumed_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "elapsed_seconds": elapsed,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"result   {result_path}", flush=True)
    print(f"elapsed  {elapsed / 60:.1f} minutes", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config, arguments.manifest, arguments.model, arguments.output)
