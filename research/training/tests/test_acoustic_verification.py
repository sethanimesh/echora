import copy
import hashlib
import json

import numpy as np
import pytest

from research.training.acoustic_verification import (audit_acceptance, digest, fit_calibration,
    fit_gate, grouped_split, homophone_key, load_cache, training_pairs, training_name_candidates,
    phonetic_key, paired_group_intervals, binomial_interval, without_acoustic, cache_storage_preflight)
from research.training.acoustic_verification import verify_frozen_fit, evaluate, sha256
from research.training.acoustic_verification import release_device_cache
from research.training.acoustic_verification import cache_rows


CONFIG = {"minimum_audit_accepted_groups": 20, "minimum_calibration_groups_per_class": 10, "acceptance_threshold": 0.95}
FUSION = {"asr_mean": 0, "asr_scale": 1, "audio_mean": 0, "audio_scale": 1,
          "asr_weight": 0, "context_weight": 0.2, "tie_margin": 0.1, "audio_support_floor": 0}
CAL = {"valid": True, "slope": 1, "margin_slope": 0, "intercept": 10, "threshold": 0.95}


def records(count):
    return [{"row": {"id": str(index), "group": str(index), "text": "tea"},
             "raw": {"hypotheses": [{"id": "h1", "literal_text": "tea", "sequence_score": -1},
                                      {"id": "h2", "literal_text": "coffee", "sequence_score": -2}],
                     "audio_quality": {"seconds": 2, "clipped_samples": 0, "low_level_warning": False}},
             "audio_scores": {"h1": 2, "h2": 0}} for index in range(count)]


def contexts(items):
    return {item["row"]["id"]: {name: {} for name in ("empty", "relevant", "wrong_recipient", "stale", "contradictory", "misleading")} for item in items}


def test_four_partitions_never_split_repeated_prompts():
    rows = [{"id": str(index), "group": str(index // 3), "text": "one word"} for index in range(90)]
    a, b, c = grouped_split(rows, 3, 12)
    c1, c2 = grouped_split(c, 2, 13)
    groups = [{row["group"] for row in partition} for partition in (a, b, c1, c2)]
    assert all(not left & right for index, left in enumerate(groups) for right in groups[index + 1:])
    assert a == grouped_split(rows, 3, 12)[0]


def test_natural_beams_and_meaning_changes_train_without_homophone_false_negatives():
    row = {"id": "x", "text": "I need two cups"}
    cached = {"raw": {"hypotheses": [{"literal_text": "I need too cups"}, {"literal_text": "I need five cups"}]}}
    pairs = training_pairs(row, cached, ["cat", "coffee", "tea"])
    assert pairs[0] == "i need two cups"
    assert "i need five cups" in pairs
    assert "i need too cups" not in pairs
    assert homophone_key("pain") == homophone_key("pane")


def test_cache_reuse_verifies_feature_and_audio_hashes(tmp_path):
    (tmp_path / "cache").mkdir()
    audio = tmp_path / "audio.wav"; audio.write_bytes(b"original")
    row = {"id": "x", "audio_filepath": str(audio)}
    path = tmp_path / "cache/x.npy"; np.save(path, np.zeros((2, 4)))
    cached = {"row_sha256": digest(row), "feature_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest()}
    (tmp_path / "cache/x.json").write_text(json.dumps(cached))
    assert load_cache(tmp_path, row) == cached
    audio.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"): load_cache(tmp_path, row)


def test_small_empirical_gate_reports_honest_uncertainty_and_does_not_lower_threshold():
    items = records(20)
    result = audit_acceptance(items, contexts(items), FUSION, CAL, CONFIG)
    assert result["audit_passed"] and result["accepted_groups"] == 20
    assert result["one_sided_95_percent_upper_false_accept_bound"] == pytest.approx(0.13910834)
    assert not audit_acceptance(items[:19], contexts(items[:19]), FUSION, CAL, CONFIG)["audit_passed"]
    items[0]["row"]["text"] = "coffee"
    assert not audit_acceptance(items, contexts(items), FUSION, CAL, CONFIG)["audit_passed"]


def test_context_duplicates_do_not_inflate_independent_audit_count():
    items = records(20)
    for item in items: item["row"]["group"] = "same-prompt"
    audit = audit_acceptance(items, contexts(items), FUSION, CAL, CONFIG)
    assert audit["accepted_groups"] == 1 and not audit["audit_passed"]


def test_calibration_needs_both_classes_and_uses_margin():
    samples = [{"group": str(index), "logit": 0.0, "margin": index / 20,
                "correct": index >= 10, "all_beams_wrong": index < 10, "weight": 1} for index in range(20)]
    calibrated = fit_calibration(samples, CONFIG)
    assert calibrated["valid"] and calibrated["margin_slope"] > 0
    assert not fit_calibration(samples[:10], CONFIG)["valid"]
    assert not fit_calibration([{**sample, "all_beams_wrong": False} for sample in samples], CONFIG)["valid"]


def test_gate_learns_from_bounded_quality_features():
    examples = [{"features": [1.0, float(index)], "base": np.array([0.0, 1.0]),
                 "delta": np.array([2.0, -1.0]), "positive": np.array([index > 4, index <= 4]), "weight": 1}
                for index in range(10)]
    fitted = fit_gate(examples, 0)
    assert fitted["coefficients"][1] > 0
    assert all(np.isfinite(fitted["coefficients"]))


def test_negative_categories_survive_full_beam_budget_and_use_training_names_only():
    source = [{"text": "Will Robin bring two cups?"}, {"text": "Will Robin bring tea?"},
              {"text": "Ask Nancy for cups."}, {"text": "Tell Nancy about tea."}]
    names = training_name_candidates(source)
    assert names == ["nancy", "robin"]
    cached = {"raw": {"hypotheses": [{"literal_text": f"will someone bring {count} cups"} for count in range(8)]}}
    pairs, categories = training_pairs({"id": "x", "text": source[0]["text"]}, cached,
                                      ["tea", "coffee", "will robin bring tea"], 8, names, with_sources=True)
    assert {"incorrect_beam", "negation", "quantity", "name"} <= set(categories)
    assert any("nancy" in text for text in pairs)
    assert any("five cups" in text for text in pairs)
    assert len(pairs) == 9


def test_quantity_digits_and_known_soundalikes_have_honest_contrasts():
    pairs, categories = training_pairs({"id": "x", "text": "bring 12 cups"},
                                      {"raw": {"hypotheses": []}}, ["tea", "coffee"], with_sources=True)
    assert "bring 13 cups" in pairs and "quantity" in categories
    assert phonetic_key("Robert") == phonetic_key("Rupert")
    assert homophone_key("sail here at night") == homophone_key("sale hear at knight")
    assert homophone_key("two cups on the thirteenth") == homophone_key("2 cups on the 13th")
    assert homophone_key("twenty one") == homophone_key("21")
    assert homophone_key("it's your piece John") == homophone_key("its you're peace Jon")
    pair = training_pairs({"id": "x", "text": "I need two cups"},
                          {"raw": {"hypotheses": [{"literal_text": "I need 2 cups"}]}}, ["tea", "coffee"])
    assert "i need 2 cups" not in pair


def test_paired_intervals_keep_repeated_prompts_and_exact_identical_outputs():
    rows = [{"reference": "tea", "group": str(index // 2),
             "outputs": {mode: "tea" if index < 6 else "coffee"
                         for mode in ("acoustic_only", "reranked", "context_assisted", "integrated")}}
            for index in range(10)]
    measured = paired_group_intervals(rows, seed=3, replicates=100)
    assert len(measured) == 12
    assert all(item["prompt_groups"] == 5 for item in measured)
    assert all(item["paired_difference_interval_95"] == [0.0, 0.0] for item in measured)
    assert binomial_interval(0, 0) == [0.0, 1.0]
    assert binomial_interval(0, 20)[1] > 0.05


def test_context_only_cannot_use_acoustic_logits_and_does_not_modify_record():
    original = records(1)[0]
    ablation = without_acoustic(original)
    assert set(ablation["audio_scores"].values()) == {0}
    assert original["audio_scores"]["h1"] == 2


def test_low_disk_preflight_preserves_whole_recordings(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("research.training.acoustic_verification.shutil.disk_usage", lambda path: SimpleNamespace(free=100))
    with pytest.raises(OSError, match="Completed cache is preserved"):
        cache_storage_preflight(tmp_path, {}, [{"duration": 40}])
    assert not list(tmp_path.iterdir())


def test_accelerator_housekeeping_only_releases_unused_buffers(monkeypatch):
    calls = []
    monkeypatch.setattr("research.training.acoustic_verification.torch.mps.synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr("research.training.acoustic_verification.torch.mps.empty_cache", lambda: calls.append("empty_cache"))
    release_device_cache("cpu")
    assert calls == []
    release_device_cache("mps")
    assert calls == ["synchronize", "empty_cache"]


def test_cached_asr_device_cannot_be_mixed_on_resume(tmp_path, monkeypatch):
    (tmp_path / "cache").mkdir(); (tmp_path / "cache/x.json").write_text("{}")
    monkeypatch.setattr("research.training.acoustic_verification.load_cache", lambda *args: {"raw": {"device": "cpu/float32"}})
    monkeypatch.setattr("research.training.acoustic_verification.make_engine", lambda *args: pytest.fail("Mismatched cache must fail before loading Qwen"))
    with pytest.raises(ValueError, match="device or precision differs"):
        cache_rows(tmp_path, {"identity": {}}, [{"id": "x"}], "mps")


def test_test_decode_device_must_match_frozen_fitting_route(tmp_path, monkeypatch):
    monkeypatch.setattr("research.training.acoustic_verification.freeze_training_protocol", lambda *args: {"asr_device": "mps/bfloat16"})
    monkeypatch.setattr("research.training.acoustic_verification.verify_frozen_fit", lambda *args: {})
    monkeypatch.setattr("research.training.acoustic_verification.cache_rows", lambda *args: pytest.fail("Mismatched test decoding must never start"))
    with pytest.raises(ValueError, match="Held-out decoding device"):
        evaluate(tmp_path, {}, "cpu")


@pytest.mark.parametrize("changed", ["fusion", "calibration", "context_only_fusion"])
def test_mutated_fitting_copy_is_rejected_before_any_test_decode(tmp_path, monkeypatch, changed):
    artifact = tmp_path / "artifact"; artifact.mkdir()
    (artifact / "scorer.safetensors").write_bytes(b"frozen-weights")
    (tmp_path / "scorer-best.safetensors").write_bytes(b"frozen-weights")
    (tmp_path / "context_bank.json").write_text("{}")
    (tmp_path / "protocol.json").write_text("{}")
    fitted = {"fusion": {"weight": 0.1}, "calibration": {"threshold": 0.95}, "context_only_fusion": {"weight": 0.1}}
    files = {"scorer.safetensors": sha256(artifact / "scorer.safetensors")}
    for key, filename in (("fusion", "fusion.json"), ("calibration", "calibration.json"), ("context_only_fusion", "context-only-fusion.json")):
        (artifact / filename).write_text(json.dumps(fitted[key])); files[filename] = sha256(artifact / filename)
    experiment = {"identity": {"beams": 5}}
    manifest = {"files": files, "artifact_id": digest([experiment["identity"], files])[:24],
                "context_bank_sha256": sha256(tmp_path / "context_bank.json"), "protocol_sha256": sha256(tmp_path / "protocol.json")}
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    fitted["artifact"] = manifest
    (tmp_path / "fit.json").write_text(json.dumps(fitted))
    assert verify_frozen_fit(tmp_path, experiment) == fitted
    fitted[changed]["changed"] = True
    (tmp_path / "fit.json").write_text(json.dumps(fitted))
    monkeypatch.setattr("research.training.acoustic_verification.freeze_training_protocol", lambda *args: {})
    monkeypatch.setattr("research.training.acoustic_verification.cache_rows", lambda *args: pytest.fail("Test decoding must not begin"))
    with pytest.raises(ValueError, match="parameters changed after audit"):
        evaluate(tmp_path, experiment, "mps")
