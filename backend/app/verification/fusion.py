"""Pure, bounded score fusion. Original hypotheses are never changed."""
from __future__ import annotations

import math


def value(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def sigmoid(number: float) -> float:
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, number))))


def relevance(item) -> float:
    if isinstance(item, (int, float)):
        number = float(item)
    elif isinstance(item, dict) and item.get("eligible", True) and not any(
            item.get(flag) for flag in ("stale", "contradictory", "out_of_scope", "deleted")):
        number = float(item.get("score", item.get("relevance", 0)))
        reliability = float(item.get("reliability", 1))
        if not math.isfinite(reliability): return 0.0
        number *= max(0.0, min(1.0, reliability))
    else:
        return 0.0
    return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0


def quality_features(audio_quality):
    seconds = float(value(audio_quality, "seconds", 0) or 0)
    clipped = float(value(audio_quality, "clipped_samples", 0) or 0)
    if not math.isfinite(seconds) or not math.isfinite(clipped):
        return 1.0, 1.0
    return (max(0.0, min(1.0, clipped / max(1.0, seconds * 16000))),
            float(bool(value(audio_quality, "low_level_warning", True))))


def gate_features(asr_values, audio_values, audio_quality):
    asr_sorted, audio_sorted = sorted(asr_values, reverse=True), sorted(audio_values, reverse=True)
    return [1.0, (asr_sorted[0] - asr_sorted[1]) if len(asr_sorted) > 1 else 0.0,
            (audio_sorted[0] - audio_sorted[1]) if len(audio_sorted) > 1 else 0.0,
            audio_sorted[0], *quality_features(audio_quality)]


def context_gate_features(rows, retrieval_by_id, audio_quality):
    relevance_values = sorted([row["context_relevance"] for row in rows], reverse=True)
    core = sorted([row["acoustic_score"] for row in rows], reverse=True)
    reliability = max([max(0.0, min(1.0, float(item.get("reliability", 1))))
                       if isinstance(item, dict) and item.get("eligible", True) else 0.0
                       for item in (retrieval_by_id or {}).values()] or [0.0])
    return [1.0, relevance_values[0],
            relevance_values[0] - relevance_values[1] if len(relevance_values) > 1 else 0.0,
            reliability, core[0] - core[1] if len(core) > 1 else 0.0,
            max(row["audio_standardized"] for row in rows), *quality_features(audio_quality)]


def adaptive_gate(features, specification):
    coefficients = specification["coefficients"]
    centers = specification["centers"]
    scales = specification["scales"]
    if not len(features) == len(coefficients) == len(centers) == len(scales):
        raise ValueError("Adaptive gate dimensions differ")
    normalized = [max(-10.0, min(10.0, (feature - center) / scale))
                  for feature, center, scale in zip(features, centers, scales, strict=True)]
    return sigmoid(sum(weight * feature for weight, feature in zip(coefficients, normalized, strict=True)))


def rank_scores(hypotheses, acoustic_scores: dict, retrieval_by_id: dict | None,
                fusion: dict, calibration: dict | None = None, *, permit_selection=False,
                audio_quality=None):
    hypotheses = list(hypotheses)
    if not hypotheses:
        return {"decision": "ambiguous", "selected_hypothesis_id": None,
                "ordered_hypothesis_ids": [], "scores": [], "reason": "no_candidates"}
    if len({value(item, "id") for item in hypotheses}) != len(hypotheses):
        raise ValueError("Hypothesis IDs must be unique")
    asr_mean, asr_scale = float(fusion["asr_mean"]), max(float(fusion["asr_scale"]), 1e-6)
    audio_mean, audio_scale = float(fusion["audio_mean"]), max(float(fusion["audio_scale"]), 1e-6)
    asr_values = [(float(value(item, "sequence_score")) - asr_mean) / asr_scale for item in hypotheses]
    audio_values = [(float(acoustic_scores[value(item, "id")]) - audio_mean) / audio_scale for item in hypotheses]
    alpha = (adaptive_gate(gate_features(asr_values, audio_values, audio_quality), fusion["acoustic_gate"])
             if fusion.get("acoustic_gate") else max(0.0, min(1.0, float(fusion["asr_weight"]))))
    context_weight = max(0.0, min(0.2, float(fusion.get("context_weight", 0))))
    tie_margin = max(0.0, min(0.2, float(fusion.get("tie_margin", 0.1))))
    floor = float(fusion.get("audio_support_floor", -math.inf))
    rows = []
    for index, hypothesis in enumerate(hypotheses):
        identifier = value(hypothesis, "id")
        audio = acoustic_scores.get(identifier)
        asr = float(value(hypothesis, "sequence_score"))
        if audio is None or not math.isfinite(float(audio)) or not math.isfinite(asr):
            raise ValueError("Every hypothesis needs finite acoustic and ASR evidence")
        audio = float(audio)
        core = alpha * ((asr - asr_mean) / asr_scale) + (1 - alpha) * ((audio - audio_mean) / audio_scale)
        rows.append({"hypothesis_id": identifier, "asr_sequence_score": asr,
                     "audio_text_logit": audio, "acoustic_score": core,
                     "audio_standardized": (audio - audio_mean) / audio_scale,
                     "adaptive_asr_weight": alpha,
                     "context_relevance": relevance((retrieval_by_id or {}).get(identifier)),
                     "original_rank": index + 1})
    best = max(row["acoustic_score"] for row in rows)
    if fusion.get("context_gate"):
        context_weight *= adaptive_gate(context_gate_features(rows, retrieval_by_id, audio_quality), fusion["context_gate"])
    for row in rows:
        row["acoustically_supported"] = (row["audio_text_logit"] >= floor
                                           and best - row["acoustic_score"] <= tie_margin + 1e-12)
        row["context_contribution"] = (context_weight * row["context_relevance"]
                                         if row["acoustically_supported"] else 0.0)
        row["fused_score"] = row["acoustic_score"] + row["context_contribution"]
    rows.sort(key=lambda row: (-row["fused_score"], row["original_rank"]))
    winner = rows[0]
    calibrated = bool(calibration and calibration.get("valid"))
    for row in rows:
        # Context never raises the confidence used to authorize selection.
        competitors = [other["acoustic_score"] for other in rows if other["hypothesis_id"] != row["hypothesis_id"]]
        row["selection_margin"] = row["acoustic_score"] - max(competitors) if competitors else 0.0
        # Calibration examples are selected winners, not arbitrary alternatives.
        row["calibrated_correctness"] = None
    if calibrated:
        winner["calibrated_correctness"] = sigmoid(float(calibration["slope"]) * winner["acoustic_score"]
            + float(calibration.get("margin_slope", 0)) * winner["selection_margin"] + float(calibration["intercept"]))
    threshold = float((calibration or {}).get("threshold", 0.95))
    seconds = float(value(audio_quality, "seconds", 0) or 0)
    clipped = float(value(audio_quality, "clipped_samples", 0) or 0)
    quality_ok = (math.isfinite(seconds) and 0.15 <= seconds <= float(fusion.get("maximum_audio_seconds", 45))
                  and math.isfinite(clipped) and clipped >= 0 and clipped <= seconds * 16000 * 0.01
                  and not value(audio_quality, "low_level_warning", True))
    selected = (permit_selection and calibrated and quality_ok and winner["acoustically_supported"]
                and winner["calibrated_correctness"] >= threshold)
    return {"decision": "selected" if selected else "ambiguous",
            "selected_hypothesis_id": winner["hypothesis_id"] if selected else None,
            "ordered_hypothesis_ids": [row["hypothesis_id"] for row in rows],
            "scores": rows, "audio_quality_eligible": quality_ok,
            "calibrated_probability_scope": "ranking_winner_only",
            "reason": "supported_selection" if selected else
            ("uncalibrated" if not calibrated else "clarification_required")}
