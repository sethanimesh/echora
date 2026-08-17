"""Qwen3-ASR command-v3 model wiring and small manual LoRA implementation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration


class LoRALinear(torch.nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self,
        base: torch.nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not isinstance(base, torch.nn.Linear):
            raise TypeError("LoRALinear can wrap only torch.nn.Linear")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / rank
        self.dropout = torch.nn.Dropout(float(dropout))
        self.lora_A = torch.nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + residual * self.scaling


def install_lora(model, config: dict) -> list[str]:
    layers = model.model.language_model.layers
    selected = [int(value) for value in config["decoder_lora_layer_indices"]]
    if len(layers) != 28:
        raise RuntimeError(f"Expected 28 Qwen decoder layers, found {len(layers)}")
    if selected != list(range(20, 28)):
        raise RuntimeError(f"Command-v3 decoder LoRA layers changed unexpectedly: {selected}")
    targets = list(config["decoder_lora_targets"])
    if targets != ["q_proj", "v_proj"]:
        raise RuntimeError(f"Command-v3 LoRA targets changed unexpectedly: {targets}")
    names = []
    for index in selected:
        attention = layers[index].self_attn
        for target in targets:
            current = getattr(attention, target)
            if isinstance(current, LoRALinear):
                raise RuntimeError(f"LoRA already installed at decoder layer {index} {target}")
            wrapped = LoRALinear(
                current,
                int(config["decoder_lora_rank"]),
                float(config["decoder_lora_alpha"]),
                float(config["decoder_lora_dropout"]),
            )
            setattr(attention, target, wrapped)
            names.extend(
                [
                    f"model.language_model.layers.{index}.self_attn.{target}.lora_A",
                    f"model.language_model.layers.{index}.self_attn.{target}.lora_B",
                ]
            )
    return names


def configure_trainables(model, config: dict) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad = False
    audio_layers = model.model.audio_tower.layers
    selected_audio = [int(value) for value in config["trainable_audio_layer_indices"]]
    if len(audio_layers) != 24 or selected_audio != [20, 21, 22, 23]:
        raise RuntimeError("Command-v3 requires Qwen audio layers 20-23")
    for index in selected_audio:
        for parameter in audio_layers[index].parameters():
            parameter.requires_grad = True
    if config.get("train_multimodal_projector") is not True:
        raise RuntimeError("Command-v3 requires the multimodal projector")
    for parameter in model.model.multi_modal_projector.parameters():
        parameter.requires_grad = True
    lora_names = install_lora(model, config)
    parameters = dict(model.named_parameters())
    for name in lora_names:
        parameters[name].requires_grad = True

    audio_prefixes = tuple(
        [f"model.audio_tower.layers.{index}." for index in selected_audio]
        + ["model.multi_modal_projector."]
    )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [
        name
        for name in trainable
        if not name.startswith(audio_prefixes) and name not in set(lora_names)
    ]
    if unexpected or not trainable:
        raise RuntimeError(f"Unexpected command-v3 trainables: {unexpected[:5]}")
    return {
        "all": trainable,
        "audio": [name for name in trainable if name.startswith(audio_prefixes)],
        "lora": lora_names,
    }


def set_training_mode(model, config: dict) -> None:
    model.eval()
    for index in config["trainable_audio_layer_indices"]:
        model.model.audio_tower.layers[int(index)].train()
    model.model.multi_modal_projector.train()
    for index in config["decoder_lora_layer_indices"]:
        attention = model.model.language_model.layers[int(index)].self_attn
        for target in config["decoder_lora_targets"]:
            getattr(attention, target).train()


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
    groups = configure_trainables(model, config)
    return processor, model, groups


def partial_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_partial_state(model, state: dict[str, torch.Tensor], exact: bool = True) -> None:
    parameters = dict(model.named_parameters())
    trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
    unknown = set(state) - set(parameters)
    if unknown:
        raise RuntimeError(f"Adapter has unknown parameters: {sorted(unknown)[:5]}")
    if exact and set(state) != trainable:
        missing = sorted(trainable - set(state))
        extra = sorted(set(state) - trainable)
        raise RuntimeError(f"Adapter parameter mismatch; missing={missing[:3]} extra={extra[:3]}")
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = parameters[name]
            if parameter.shape != tensor.shape:
                raise RuntimeError(f"Adapter shape mismatch for {name}")
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))


def load_safetensors_adapter(model, path: Path, exact: bool = True) -> None:
    load_partial_state(model, load_file(path, device="cpu"), exact=exact)


def save_adapter(path: Path, model, metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {name: tensor.contiguous() for name, tensor in partial_state(model).items()}
    string_metadata = {name: str(value) for name, value in metadata.items()}
    save_file(state, temporary, metadata=string_metadata)
    temporary.replace(path)
    descriptor = {**metadata, "adapter_file": path.name}
    path.with_name("adapter_config.json").write_text(
        json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
    )
    return descriptor
