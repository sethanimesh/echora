import copy
import json

import pytest
import torch
from safetensors.torch import save_file

from app.verification.fusion import rank_scores, sigmoid
from app.verification.runtime import ADAPTER_SHA, FOUNDATION_SHA, ROOT, VerificationRuntime, sha256
from app.verification.scorer import AcousticTextScorer, DEFAULT_CONFIG, FEATURE_VERSION, FORMAT, contrastive_loss, text_ids


QUALITY = {"seconds": 2, "clipped_samples": 0, "low_level_warning": False}
HYPOTHESES = [{"id": "h1", "literal_text": "I want tea", "sequence_score": -0.1, "search_weight": 0.6},
              {"id": "h2", "literal_text": "I want coffee", "sequence_score": -0.2, "search_weight": 0.4}]
FUSION = {"asr_mean": 0, "asr_scale": 1, "audio_mean": 0, "audio_scale": 1,
          "asr_weight": 0.5, "context_weight": 0.2, "tie_margin": 0.1, "audio_support_floor": 0,
          "maximum_audio_seconds": 45}
CALIBRATION = {"valid": True, "slope": 1, "margin_slope": 2, "intercept": 10, "threshold": 0.95}


def test_context_cannot_promote_acoustically_unsupported_candidate_or_change_evidence():
    original = copy.deepcopy(HYPOTHESES)
    result = rank_scores(HYPOTHESES, {"h1": 2, "h2": -4}, {"h2": {"relevance": 1, "eligible": True}},
                         FUSION, CALIBRATION, permit_selection=True, audio_quality=QUALITY)
    assert result["selected_hypothesis_id"] == "h1"
    assert result["scores"][1]["context_contribution"] == 0
    assert HYPOTHESES == original


def test_single_candidate_and_large_context_are_not_proof_of_correctness():
    result = rank_scores(HYPOTHESES[:1], {"h1": -4}, {"h1": 100}, FUSION, CALIBRATION,
                         permit_selection=True, audio_quality=QUALITY)
    assert result["decision"] == "ambiguous"
    assert result["scores"][0]["selection_margin"] == 0


@pytest.mark.parametrize("flags", [{"stale": True}, {"contradictory": True}, {"out_of_scope": True},
                                   {"deleted": True}, {"reliability": float("nan")}, {"reliability": 0}])
def test_invalid_or_unreliable_context_is_zero(flags):
    result = rank_scores(HYPOTHESES, {"h1": 1, "h2": 1.02},
                         {"h2": {"eligible": True, "relevance": 1, **flags}}, FUSION)
    assert next(row for row in result["scores"] if row["hypothesis_id"] == "h2")["context_contribution"] == 0


def test_uncalibrated_or_bad_quality_never_selects():
    for calibration, quality in ((None, QUALITY), (CALIBRATION, None),
                                  (CALIBRATION, {**QUALITY, "low_level_warning": True}),
                                  (CALIBRATION, {**QUALITY, "seconds": 46}),
                                  (CALIBRATION, {**QUALITY, "clipped_samples": 1000})):
        result = rank_scores(HYPOTHESES, {"h1": 8, "h2": 1}, {}, FUSION, calibration,
                             permit_selection=True, audio_quality=quality)
        assert result["selected_hypothesis_id"] is None


def test_context_cannot_increase_calibrated_correctness():
    scores = {"h1": 1, "h2": 1.02}
    plain = rank_scores(HYPOTHESES, scores, {}, FUSION, CALIBRATION, audio_quality=QUALITY)
    context = rank_scores(HYPOTHESES, scores, {"h2": 1}, FUSION, CALIBRATION, audio_quality=QUALITY)
    assert context["scores"][0]["calibrated_correctness"] <= plain["scores"][0]["calibrated_correctness"]
    assert all(row["calibrated_correctness"] is None for row in context["scores"][1:])
    winner = context["scores"][0]
    assert winner["calibrated_correctness"] == sigmoid(CALIBRATION["slope"] * winner["acoustic_score"]
        + CALIBRATION["margin_slope"] * winner["selection_margin"] + CALIBRATION["intercept"])


def test_acoustic_gate_adapts_to_margin_and_preserves_hard_context_cap():
    config = {**FUSION, "acoustic_gate": {"coefficients": [0, 2, 0, 0, 0, 0],
                                          "centers": [0] * 6, "scales": [1] * 6}}
    small = rank_scores(HYPOTHESES, {"h1": 1, "h2": 1}, {}, config)
    wide = rank_scores([{**HYPOTHESES[0], "sequence_score": 3}, HYPOTHESES[1]], {"h1": 1, "h2": 1}, {}, config)
    assert wide["scores"][0]["adaptive_asr_weight"] > small["scores"][0]["adaptive_asr_weight"]


def test_scorer_uses_order_preserves_utf8_and_backpropagates():
    torch.manual_seed(1)
    model = AcousticTextScorer({"input_size": 4, "dimension": 8, "heads": 2, "layers": 1, "feedforward": 16, "dropout": 0})
    frames = torch.randn(8, 4)
    forward = model(frames, ["I need water", "I do not need water"])
    reverse = model(frames.flip(0), ["I need water", "I do not need water"])
    assert not torch.allclose(forward, reverse, atol=1e-7)
    contrastive_loss(forward, torch.tensor([True, False])).backward()
    assert model.audio[1].weight.grad.abs().sum() > 0
    assert len(text_ids("José 12")) > 0
    assert len(text_ids("word " * 300)) > 128


def artifact(tmp_path):
    inference = json.loads((ROOT / "models/echora-qwen3-asr-command-v3/evaluation/qwen_command_v3.json").read_text())
    model = AcousticTextScorer()
    save_file(model.state_dict(), tmp_path / "scorer.safetensors")
    fusion = {**FUSION, "acoustic_gate": {"coefficients": [0] * 6, "centers": [0] * 6, "scales": [1] * 6},
              "context_gate": {"coefficients": [0] * 8, "centers": [0] * 8, "scales": [1] * 8}}
    (tmp_path / "fusion.json").write_text(json.dumps(fusion))
    (tmp_path / "calibration.json").write_text(json.dumps({"valid": False, "audit_passed": False, "threshold": 0.95}))
    manifest = {"format": FORMAT, "feature_version": FEATURE_VERSION, "foundation_sha256": FOUNDATION_SHA,
                "asr_device": "mps/bfloat16",
                "adapter_sha256": ADAPTER_SHA, "beams": 5, "trained": True,
                "literal_prompt": inference["literal_prompt"], "max_new_tokens": inference["generation_max_new_tokens"],
                "scorer_config": DEFAULT_CONFIG, "files": {name: sha256(tmp_path / name)
                    for name in ("scorer.safetensors", "fusion.json", "calibration.json")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_missing_and_advisory_artifacts_never_fabricate_confidence(tmp_path):
    assert VerificationRuntime(tmp_path).rank({"hypotheses": HYPOTHESES})["decision"] == "ambiguous"
    artifact(tmp_path)
    runtime = VerificationRuntime(tmp_path)
    assert runtime.available and runtime.status == "advisory"
    result = runtime.rank({"hypotheses": HYPOTHESES, "audio_quality": QUALITY}, {"h1": 1, "h2": 0})
    assert result["decision"] == "ambiguous"
    assert all(row["calibrated_correctness"] is None for row in result["scores"])


def test_different_live_asr_device_keeps_ranking_but_never_automatic_selection(tmp_path):
    artifact(tmp_path)
    runtime = VerificationRuntime(tmp_path)
    runtime.status = "ready"
    runtime.calibration = CALIBRATION
    result = runtime.rank({"hypotheses": HYPOTHESES, "device": "cpu/float32", "audio_quality": QUALITY}, {"h1": 2, "h2": 0})
    assert result["status"] == "advisory" and result["decision"] == "ambiguous"
    assert result["reason"] == "asr_device_or_precision_not_validated"
    assert len(result["scores"]) == 2
    assert all(row["calibrated_correctness"] is None for row in result["scores"])


@pytest.mark.parametrize("corruption", ["hash", "prompt", "beam", "nan", "audit"])
def test_artifact_corruption_or_false_audit_is_rejected(tmp_path, corruption):
    manifest = artifact(tmp_path)
    if corruption == "hash": (tmp_path / "fusion.json").write_text("{}")
    elif corruption == "prompt": manifest["literal_prompt"] = "repair grammar"
    elif corruption == "beam": manifest["beams"] = 4
    elif corruption == "nan":
        fusion = json.loads((tmp_path / "fusion.json").read_text()); fusion["asr_weight"] = float("nan")
        (tmp_path / "fusion.json").write_text(json.dumps(fusion)); manifest["files"]["fusion.json"] = sha256(tmp_path / "fusion.json")
    elif corruption == "audit":
        (tmp_path / "calibration.json").write_text(json.dumps({**CALIBRATION, "audit_passed": True,
            "correct_groups": 20, "incorrect_groups": 20, "accepted_groups": 19,
            "incorrect_groups_audit": 0, "harmful_context_flips": 0}))
        manifest["files"]["calibration.json"] = sha256(tmp_path / "calibration.json")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    runtime = VerificationRuntime(tmp_path)
    assert not runtime.available
    assert runtime.rank({"hypotheses": HYPOTHESES}, {"h1": 2, "h2": 1})["decision"] == "ambiguous"
