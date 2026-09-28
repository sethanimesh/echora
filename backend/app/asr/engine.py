"""Self-contained command-v3 Qwen inference engine."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from ..audio import audio_quality
from ..schemas import Hypothesis, RawAsrResult


EXPECTED_FOUNDATION_SHA256 = "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1"
EXPECTED_ADAPTER_SHA256 = "7cd203cc0cbf479e6afa198cc3895fedc71f1907b870fcb7f1dece0a1e6b2021"
ADAPTER_FORMAT = "echora-qwen3-asr-command-v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def build_hypotheses(texts: list[str], scores: list[float]) -> list[Hypothesis]:
    """Deduplicate decoded beams while preserving decoder order and evidence."""
    unique: list[tuple[str, float]] = []
    seen: set[str] = set()
    for text, score in zip(texts, scores, strict=True):
        cleaned = text.strip()
        key = normalize(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((cleaned, float(score)))
    if not unique:
        raise RuntimeError("The ASR model returned no transcript hypotheses")
    probabilities = torch.softmax(torch.tensor([score for _, score in unique]), dim=0).tolist()
    return [
        Hypothesis(
            id=f"h{index}",
            literal_text=text,
            sequence_score=round(score, 6),
            search_weight=round(float(probability), 6),
        )
        for index, ((text, score), probability) in enumerate(zip(unique, probabilities, strict=True), 1)
    ]


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / rank
        self.dropout = torch.nn.Dropout(float(dropout))
        self.lora_A = torch.nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = functional.linear(functional.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + residual * self.scaling


def select_device(requested: str) -> tuple[str, torch.dtype]:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            raise RuntimeError(
                "No supported accelerator is available. Set ECHORA_DEVICE=cpu explicitly for slow CPU inference."
            )
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("ECHORA_DEVICE=cuda was requested, but CUDA is unavailable")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("ECHORA_DEVICE=mps was requested, but Apple MPS is unavailable")
        return "mps", torch.bfloat16
    if requested == "cpu":
        return "cpu", torch.float32
    raise ValueError("ECHORA_DEVICE must be auto, mps, cuda, or cpu")


class QwenCommandEngine:
    def __init__(
        self,
        foundation: Path,
        adapter: Path,
        config_path: Path,
        device_request: str = "auto",
        verify_hashes: bool = True,
    ) -> None:
        for path in (foundation, adapter, config_path):
            if not path.exists():
                raise FileNotFoundError(f"Required model artifact is missing: {path}")
        weight = foundation / "model.safetensors"
        if verify_hashes:
            actual_foundation = sha256(weight)
            actual_adapter = sha256(adapter)
            if actual_foundation != EXPECTED_FOUNDATION_SHA256:
                raise RuntimeError(f"Foundation hash mismatch: {actual_foundation}")
            if actual_adapter != EXPECTED_ADAPTER_SHA256:
                raise RuntimeError(f"Adapter hash mismatch: {actual_adapter}")
        with safe_open(adapter, framework="pt") as handle:
            metadata = handle.metadata() or {}
        if metadata.get("format") != ADAPTER_FORMAT:
            raise RuntimeError("Adapter is not an Echora command-v3 artifact")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.prompt = str(self.config["literal_prompt"])
        self.max_tokens = int(self.config.get("generation_max_new_tokens", 128))
        self.device, self.dtype = select_device(device_request)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(foundation, local_files_only=True)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            foundation,
            dtype=self.dtype,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.config.use_cache = False
        self._install_lora()
        self._load_adapter(adapter)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.load_seconds = round(time.perf_counter() - started, 3)

    def _install_lora(self) -> None:
        layers = self.model.model.language_model.layers
        selected = [int(value) for value in self.config["decoder_lora_layer_indices"]]
        targets = list(self.config["decoder_lora_targets"])
        if len(layers) != 28 or selected != list(range(20, 28)) or targets != ["q_proj", "v_proj"]:
            raise RuntimeError("The foundation architecture does not match command-v3")
        for index in selected:
            attention = layers[index].self_attn
            for target in targets:
                setattr(
                    attention,
                    target,
                    LoRALinear(
                        getattr(attention, target),
                        int(self.config["decoder_lora_rank"]),
                        float(self.config["decoder_lora_alpha"]),
                        float(self.config["decoder_lora_dropout"]),
                    ),
                )

    def _load_adapter(self, adapter: Path) -> None:
        state = load_file(adapter, device="cpu")
        parameters = dict(self.model.named_parameters())
        missing_names = sorted(set(state) - set(parameters))
        if missing_names:
            raise RuntimeError(f"Adapter contains unknown tensors: {missing_names[:3]}")
        expected = {
            name
            for name in parameters
            if name.startswith(tuple(f"model.audio_tower.layers.{i}." for i in range(20, 24)))
            or name.startswith("model.multi_modal_projector.")
            or (".self_attn.q_proj.lora_" in name or ".self_attn.v_proj.lora_" in name)
            and any(f"layers.{i}." in name for i in range(20, 28))
        }
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise RuntimeError(f"Adapter tensor mismatch; missing={missing[:3]} extra={extra[:3]}")
        with torch.no_grad():
            for name, tensor in state.items():
                parameter = parameters[name]
                if parameter.shape != tensor.shape:
                    raise RuntimeError(f"Adapter shape mismatch for {name}")
                parameter.copy_(tensor.to(parameter.device, parameter.dtype))

    def transcribe(self, audio: np.ndarray, beams: int, backend: str = "local") -> RawAsrResult:
        started = time.perf_counter()
        inputs = self.processor.apply_transcription_request(
            audio=[audio], language=["English"], prompt=self.prompt
        ).to(self.model.device, self.model.dtype)
        previous_cache = self.model.config.use_cache
        self.model.config.use_cache = True
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    num_beams=beams,
                    num_return_sequences=beams,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        finally:
            self.model.config.use_cache = previous_cache
        ids = generated.sequences[:, inputs["input_ids"].shape[1] :]
        texts = self.processor.decode(ids, return_format="transcription_only")
        scores = generated.sequences_scores.detach().float().cpu().tolist()
        hypotheses = build_hypotheses(texts, scores)
        return RawAsrResult(
            backend=backend,
            device=f"{self.device}/{str(self.dtype).replace('torch.', '')}",
            hypotheses=hypotheses,
            audio_quality=audio_quality(audio),
            decode_seconds=round(time.perf_counter() - started, 3),
        )

    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Return ordered frozen acoustic frames, independently of decoder text.

        Callers use the backend's existing inference lock. No beam, generated
        token, decoder score, model parameter, or generation setting is changed.
        These features are ephemeral at runtime; research caching is explicit.
        """
        inputs = self.processor.apply_transcription_request(
            audio=[audio], language=["English"], prompt=self.prompt
        ).to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            features = self.model.model.get_audio_features(
                inputs["input_features"], inputs["input_features_mask"], return_dict=True
            ).pooler_output
        if features.ndim != 2 or features.shape[1] != 2048 or not torch.isfinite(features).all():
            raise RuntimeError("Unexpected frozen Qwen audio feature shape or values")
        return features.detach().float().cpu().contiguous()
