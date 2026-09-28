"""Fail-closed artifact loader and runtime adapter for acoustic verification."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from .fusion import rank_scores, value


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = ROOT / "data" / "derived" / "verification" / "run-v1" / "artifact"
FOUNDATION_SHA = "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1"
ADAPTER_SHA = "7cd203cc0cbf479e6afa198cc3895fedc71f1907b870fcb7f1dece0a1e6b2021"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VerificationRuntime:
    def __init__(self, artifact_dir: Path | str | None = None, device="cpu"):
        self.path = Path(artifact_dir or os.getenv("ECHORA_VERIFICATION_ARTIFACT", str(DEFAULT_ARTIFACT)))
        self.device = device
        self.available = False
        self.status, self.reason = "unavailable", "verification_artifact_missing"
        self.manifest, self.fusion, self.calibration, self.model = {}, {}, {}, None
        if not (self.path / "manifest.json").is_file():
            return
        try:
            manifest = json.loads((self.path / "manifest.json").read_text())
            if (manifest.get("format") != "echora-acoustic-verifier-v1"
                    or manifest.get("feature_version") != "command-v3-projected-audio-2048-v1"
                    or manifest.get("foundation_sha256") != FOUNDATION_SHA
                    or manifest.get("adapter_sha256") != ADAPTER_SHA
                    or manifest.get("asr_device") not in {"mps/bfloat16", "cpu/float32"}
                    or manifest.get("beams") != 5 or not manifest.get("trained")):
                raise ValueError("Verification artifact does not match deployed ASR")
            inference_path = Path(os.getenv("ECHORA_INFERENCE_CONFIG", str(
                ROOT / "models/echora-qwen3-asr-command-v3/evaluation/qwen_command_v3.json")))
            inference = json.loads(inference_path.read_text())
            if (manifest.get("literal_prompt") != inference["literal_prompt"]
                    or manifest.get("max_new_tokens") != inference["generation_max_new_tokens"]):
                raise ValueError("Verification decoding configuration differs from deployed ASR")
            for name in ("scorer.safetensors", "fusion.json", "calibration.json"):
                if sha256(self.path / name) != manifest["files"][name]:
                    raise ValueError(f"Verification artifact hash mismatch: {name}")
            from safetensors.torch import load_file
            from .scorer import AcousticTextScorer
            self.model = AcousticTextScorer(manifest["scorer_config"])
            self.model.load_state_dict(load_file(self.path / "scorer.safetensors"), strict=True)
            self.model.to(device).eval()
            self.model.requires_grad_(False)
            self.fusion = json.loads((self.path / "fusion.json").read_text())
            self.calibration = json.loads((self.path / "calibration.json").read_text())
            for name in ("asr_mean", "asr_scale", "audio_mean", "audio_scale", "asr_weight",
                         "context_weight", "tie_margin", "audio_support_floor", "maximum_audio_seconds"):
                if not isinstance(self.fusion.get(name), (int, float)) or not math.isfinite(self.fusion[name]):
                    raise ValueError("Fusion parameters must be finite")
            if (min(self.fusion["asr_scale"], self.fusion["audio_scale"]) <= 0
                    or not 0 <= self.fusion["asr_weight"] <= 1
                    or not 0 <= self.fusion["context_weight"] <= 0.2
                    or not 0 <= self.fusion["tie_margin"] <= 0.2
                    or not 0.15 <= self.fusion["maximum_audio_seconds"] <= 60):
                raise ValueError("Fusion parameters exceed their validated bounds")
            for name, dimension in (("acoustic_gate", 6), ("context_gate", 8)):
                gate = self.fusion.get(name) or {}
                if any(len(gate.get(field, [])) != dimension for field in ("coefficients", "centers", "scales")):
                    raise ValueError("Learned adaptive gate is missing or has invalid dimensions")
                if not all(math.isfinite(float(number)) for field in ("coefficients", "centers", "scales") for number in gate[field]):
                    raise ValueError("Adaptive gate coefficients must be finite")
                if min(gate["scales"]) <= 0:
                    raise ValueError("Adaptive gate scales must be positive")
            if self.calibration.get("threshold") != 0.95:
                raise ValueError("Acceptance threshold differs from the frozen experiment")
            if self.calibration.get("valid"):
                if (not all(math.isfinite(float(self.calibration.get(name, math.nan))) for name in ("slope", "margin_slope", "intercept"))
                        or min(self.calibration["slope"], self.calibration["margin_slope"]) < 0
                        or min(self.calibration.get("correct_groups", 0), self.calibration.get("incorrect_groups", 0)) < 10
                        or self.calibration.get("all_beams_wrong_groups", 0) < 1):
                    raise ValueError("Calibration lacks enough independent groups or finite coefficients")
            if self.calibration.get("audit_passed") and (
                    not self.calibration.get("valid") or self.calibration.get("accepted_groups", 0) < 20
                    or self.calibration.get("incorrect_groups_audit", -1) != 0
                    or self.calibration.get("harmful_context_flips", -1) != 0):
                raise ValueError("Acceptance audit does not meet the frozen empirical gate")
            self.manifest, self.available = manifest, True
            self.status = "ready" if self.calibration.get("audit_passed") else "advisory"
            self.reason = "verified_artifact" if self.status == "ready" else "acceptance_audit_not_passed"
        except Exception as exc:
            self.model = None
            self.reason = f"verification_artifact_invalid:{type(exc).__name__}"

    def score(self, features, hypotheses) -> dict[str, float]:
        if not self.available or features is None:
            return {}
        import torch
        with torch.inference_mode():
            output = self.model(torch.as_tensor(features), [value(item, "literal_text") for item in hypotheses])
        return {value(item, "id"): float(logit) for item, logit in
                zip(hypotheses, output.detach().float().cpu().tolist(), strict=True)}

    def rank(self, raw, acoustic_scores=None, retrieval_by_id=None):
        hypotheses = value(raw, "hypotheses", [])
        details = {"artifact_id": self.manifest.get("artifact_id"),
                   "feature_version": self.manifest.get("feature_version"),
                   "calibration_valid": bool(self.calibration.get("valid")),
                   "validated_asr_device": self.manifest.get("asr_device"),
                   "acceptance_audit_passed": bool(self.calibration.get("audit_passed"))}
        if not self.available or not acoustic_scores:
            return {"status": "unavailable" if not self.available else "advisory",
                    "decision": "ambiguous", "selected_hypothesis_id": None,
                    "ordered_hypothesis_ids": [value(item, "id") for item in hypotheses],
                    "scores": [], "artifacts": details,
                    "reason": self.reason if not self.available else "acoustic_features_unavailable"}
        try:
            matched_device = value(raw, "device") == self.manifest.get("asr_device")
            result = rank_scores(hypotheses, acoustic_scores, retrieval_by_id, self.fusion,
                                 self.calibration if matched_device else None,
                                 permit_selection=self.status == "ready" and matched_device,
                                 audio_quality=value(raw, "audio_quality"))
            if not matched_device: result["reason"] = "asr_device_or_precision_not_validated"
            return {"status": self.status if matched_device else "advisory", **result, "artifacts": details}
        except (ValueError, TypeError, KeyError, OverflowError):
            return {"status": "advisory", "decision": "ambiguous", "selected_hypothesis_id": None,
                    "ordered_hypothesis_ids": [value(item, "id") for item in hypotheses],
                    "scores": [], "artifacts": details, "reason": "verification_evidence_invalid"}
