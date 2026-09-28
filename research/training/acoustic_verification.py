"""Resumable Mac-only frozen-Qwen verification experiment.

Stages: prepare, benchmark, cache, train, fit, evaluate, all. M04 is decoded
only by evaluate, after the scorer, fusion, calibration and acceptance audit
are frozen. Cache records never replace literal ASR scores with learned ones.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import json
import math
import platform
import importlib.metadata
from pathlib import Path
import random
import re
import resource
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from backend.app.verification.scorer import AcousticTextScorer, contrastive_loss, normalize, FORMAT, FEATURE_VERSION
from backend.app.verification.fusion import rank_scores, gate_features, context_gate_features
from backend.app.verification.runtime import FOUNDATION_SHA, ADAPTER_SHA, sha256
from research.benchmarks.metrics import aggregate, score

DEFAULT_CONFIG = ROOT / "research/training/configs/acoustic_verification_v1.json"
DEFAULT_RUN = ROOT / "data/derived/verification/run-v1"
SPLITS_TO_FIT = ("train", "selection", "fusion", "calibration", "audit")
SPLITS_TO_TEST = ("test", "normal_test")
SCENARIOS = ("empty", "relevant", "wrong_recipient", "stale", "contradictory", "misleading")
HOMOPHONES = [{"to", "too", "two"}, {"one", "won"}, {"four", "for"}, {"right", "write"},
              {"see", "sea"}, {"tea", "tee"}, {"no", "know"}, {"pain", "pane"},
              {"flower", "flour"}, {"there", "their", "they're"}, {"sale", "sail"},
              {"hear", "here"}, {"knight", "night"}, {"eight", "ate"}, {"pair", "pear"},
              {"anna", "ana"}, {"ann", "anne"}, {"its", "it's"}, {"your", "you're"},
              {"whose", "who's"}, {"by", "buy", "bye"}, {"be", "bee"}, {"which", "witch"},
              {"peace", "piece"}, {"meet", "meat"}, {"week", "weak"}, {"wait", "weight"},
              {"allowed", "aloud"}, {"break", "brake"}, {"whole", "hole"}, {"knew", "new"},
              {"john", "jon"}, {"sara", "sarah"}, {"mark", "marc"}, {"carl", "karl"}]
_VERIFIED_FILES = {}
_CONTEXT_ENCODER = None
SOURCE_FILES = ("research/training/acoustic_verification.py", "backend/app/asr/engine.py",
                "backend/app/audio.py", "backend/app/verification/scorer.py",
                "backend/app/verification/fusion.py", "backend/app/verification/runtime.py",
                "backend/app/messaging/fidelity.py",
                "communication/backend/retrieval.py", "research/benchmarks/metrics.py")


def memory_report():
    report = {"process_rss_highwater_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
              * (1 if platform.system() == "Darwin" else 1024)}
    if torch.backends.mps.is_available():
        report.update(mps_allocated_bytes_snapshot=torch.mps.current_allocated_memory(),
                      mps_driver_bytes_snapshot=torch.mps.driver_allocated_memory())
    report["mps_values_are_snapshots_not_peak"] = True
    return report


def release_device_cache(device):
    """Drop unused accelerator buffers between completed operations only."""
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def verify_file(path, expected):
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, expected)
    if key not in _VERIFIED_FILES:
        if sha256(path) != expected: raise ValueError(f"Cached file checksum changed: {path.name}")
        _VERIFIED_FILES[key] = True


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def source_identity():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def decoding_device(device):
    return {"mps": "mps/bfloat16", "cpu": "cpu/float32"}[device]


def freeze_integrity(run, experiment):
    """Freeze all input audio bytes without decoding or mining test references."""
    identity = experiment["identity"]
    if digest(experiment["splits"]) != identity["split_sha256"] or digest(experiment["config"]) != identity["config_sha256"]:
        raise ValueError("Frozen experiment membership or configuration was modified")
    path = run / "corpus-integrity.json"
    if path.exists():
        frozen = read_json(path)
        if frozen["split_sha256"] != identity["split_sha256"]:
            raise ValueError("Corpus hash manifest does not match frozen membership")
        for rows in experiment["splits"].values():
            for row in rows: verify_file(Path(row["audio_filepath"]), frozen["audio_sha256"][row["id"]])
        return frozen
    frozen = {"split_sha256": identity["split_sha256"], "frozen_at": int(time.time()),
              "audio_sha256": {row["id"]: sha256(Path(row["audio_filepath"]))
                               for rows in experiment["splits"].values() for row in rows}}
    atomic_json(path, frozen)
    return frozen


def freeze_training_protocol(run, experiment):
    freeze_integrity(run, experiment)
    source_manifests = ("data/derived/torgo/folds/M04/train.jsonl", "data/derived/torgo/folds/M04/dev.jsonl",
                        "data/derived/torgo/folds/M04/test.jsonl", "data/derived/common_voice/pilot-v1/train.jsonl",
                        "data/derived/common_voice/pilot-v1/test.jsonl")
    selected_ids = {row["id"] for rows in experiment["splits"].values() for row in rows}
    originals, eligibility = {}, {}
    for name in source_manifests:
        rows = manifest_rows(ROOT / name)
        originals.update({row["id"]: row for row in rows})
        excluded = [row for row in rows if not 0.15 <= float(row["duration"]) <= experiment["config"]["maximum_audio_seconds"]]
        eligibility[name] = {"original_recordings": len(rows), "duration_excluded_whole_recordings": len(excluded),
                             "duration_excluded_ids": [row["id"] for row in excluded],
                             "included_recordings": sum(row["id"] in selected_ids for row in rows),
                             "duration_eligible_but_not_sampled": sum(row["id"] not in selected_ids and row not in excluded for row in rows)}
    if any(originals.get(row["id"]) != row for rows in experiment["splits"].values() for row in rows):
        raise ValueError("Original source manifests differ from frozen experiment rows")
    cache_identity = {row["id"]: sha256(cache_path(run, row)) for name in SPLITS_TO_FIT
                      for row in experiment["splits"][name]}
    devices = {read_json(cache_path(run, row))["raw"]["device"] for name in SPLITS_TO_FIT
               for row in experiment["splits"][name]}
    if len(devices) != 1 or not devices <= {"mps/bfloat16", "cpu/float32"}:
        raise ValueError("Fitting cache mixes unvalidated ASR devices or precision")
    protocol = {"experiment_sha256": sha256(run / "experiment.json"),
                "asr_device": next(iter(devices)),
                "corpus_integrity_sha256": sha256(run / "corpus-integrity.json"),
                "context_bank_sha256": sha256(run / "context_bank.json"),
                "source_sha256": source_identity(),
                "source_manifests_sha256": {name: sha256(ROOT / name) for name in source_manifests},
                "corpus_eligibility": eligibility,
                "minilm_bundle_sha256": {str(path.relative_to(ROOT)): sha256(path)
                    for path in sorted((ROOT / "models/echora-minilm-l6-v2").rglob("*")) if path.is_file()},
                "qwen_processor_sha256": {str(path.relative_to(ROOT)): sha256(path)
                    for path in sorted((ROOT / "models/echora-qwen3-asr-command-v3/foundation").glob("*"))
                    if path.is_file() and path.suffix in {".json", ".txt", ".model"}},
                "fitting_cache_metadata_sha256": cache_identity,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers", "scipy", "safetensors")},
                "python": platform.python_version(), "platform": platform.platform(),
                "evaluation": {"bootstrap_replicates": 2000, "bootstrap_unit": "normalized_prompt_group",
                               "bootstrap_seed": experiment["config"]["seed"], "confidence_level": 0.95}}
    path = run / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Frozen training/evaluation code or inputs changed; do not resume this run with changed methods")
    if not path.exists(): atomic_json(path, protocol)
    return protocol


def manifest_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        old = str(row["audio_filepath"])
        # Resolve only corpus-relative paths into the canonical repository.
        if "/data/raw/" not in old:
            raise ValueError(f"Noncanonical corpus path: {old}")
        row["audio_filepath"] = str(ROOT / "data/raw" / old.split("/data/raw/", 1)[1])
        if not Path(row["audio_filepath"]).is_file():
            raise FileNotFoundError(row["audio_filepath"])
        row["group"] = digest(normalize(row["text"]))[:20]
    return rows


def grouped_split(rows, parts: int, seed: int):
    """Group repeated prompts across sessions; balance word-length strata."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    strata = defaultdict(list)
    for identifier, items in groups.items():
        strata[min(5, len(normalize(items[0]["text"]).split()))].append(identifier)
    result = [[] for _ in range(parts)]
    for _, identifiers in sorted(strata.items()):
        identifiers.sort(key=lambda identifier: digest([seed, identifier]))
        for identifier in identifiers:
            slot = min(range(parts), key=lambda index: (len(result[index]), index))
            result[slot].extend(groups[identifier])
    return [sorted(part, key=lambda row: row["id"]) for part in result]


def balanced_subset(rows, count, seed):
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    for speaker, values in by_speaker.items():
        values.sort(key=lambda row: digest([seed, speaker, row["id"]]))
    selected = []
    while len(selected) < min(count, len(rows)):
        for speaker in sorted(by_speaker):
            if by_speaker[speaker] and len(selected) < count:
                selected.append(by_speaker[speaker].pop())
    return selected


def prepare(run: Path, config: dict):
    path = run / "experiment.json"
    if path.exists():
        experiment = read_json(path)
        if experiment["config"] != config:
            raise ValueError("Existing run configuration differs; use a separate run directory")
        return experiment
    maximum = config["maximum_audio_seconds"]
    clean = lambda rows: [row for row in rows if 0.15 <= float(row["duration"]) <= maximum]
    base = ROOT / "data/derived/torgo/folds/M04"
    train = clean(manifest_rows(base / "train.jsonl"))
    dysarthric = [row for row in train if row["condition"] == "dysarthric"]
    controls = balanced_subset([row for row in train if row["condition"] == "control"],
                               config["control_train_limit"], config["seed"])
    normal = ROOT / "data/derived/common_voice/pilot-v1"
    ordinary = balanced_subset(clean(manifest_rows(normal / "train.jsonl")),
                                config["normal_train_limit"], config["seed"])
    selection, fusion, remaining = grouped_split(clean(manifest_rows(base / "dev.jsonl")), 3, config["seed"])
    calibration, audit = grouped_split(remaining, 2, config["seed"] + 1)
    splits = {"train": dysarthric + controls + ordinary, "selection": selection,
              "fusion": fusion, "calibration": calibration, "audit": audit,
              "test": clean(manifest_rows(base / "test.jsonl")),
              "normal_test": clean(manifest_rows(normal / "test.jsonl"))}
    train_speakers = {row["speaker"] for row in splits["train"]}
    if train_speakers & {"F03", "M04"}:
        raise ValueError("Training speaker leakage")
    dev_groups = [{row["group"] for row in splits[name]} for name in SPLITS_TO_FIT[1:]]
    if any(left & right for index, left in enumerate(dev_groups) for right in dev_groups[index + 1:]):
        raise ValueError("Development prompt-group leakage")
    all_ids = [row["id"] for values in splits.values() for row in values]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Utterance appears in multiple partitions")
    inference = read_json(ROOT / "models/echora-qwen3-asr-command-v3/evaluation/qwen_command_v3.json")
    identity = {"foundation_sha256": FOUNDATION_SHA, "adapter_sha256": ADAPTER_SHA,
                "feature_version": FEATURE_VERSION, "beams": config["beams"],
                "literal_prompt": inference["literal_prompt"],
                "max_new_tokens": inference["generation_max_new_tokens"],
                "cache_precision": "float32", "split_sha256": digest(splits), "config_sha256": digest(config)}
    experiment = {"config": config, "identity": identity, "splits": splits,
                  "limitations": ["Fixed deployed adapter; historical M04 holdout, not a new untouched test.",
                                  "F03 calibration is one speaker already used in ASR model selection.",
                                  "Original TORGO prompt overlap matches deployed model training.",
                                  "Context fixtures are artificial approved entries, not patient history."]}
    atomic_json(path, experiment)
    # Every fixture originates from the training partition before any test decoding.
    counts = Counter(normalize(row["text"]) for row in splits["train"])
    phrases = sorted(counts, key=lambda text: (-counts[text], text))
    phrases = [text for text in phrases if text and len(text.split()) <= 8][:160]
    now = int(time.time())
    entries = [{"source_id": f"synthetic-train-{index}", "kind": "remembered", "language": "en",
                "text": text, "wording": text, "scope": {"recipient": "Alex"},
                "created_at": now, "source_revision": 1, "content_hash": digest(text),
                "provenance": "synthetic-approved-training-only"} for index, text in enumerate(phrases)]
    atomic_json(run / "context_bank.json", {"frozen_at": now, "source_split_sha256": digest(splits["train"]),
                                           "entries": entries, "sha256": digest(entries)})
    print(json.dumps({"stage": "prepared", "splits": {name: {"utterances": len(rows),
                       "prompt_groups": len({row['group'] for row in rows})} for name, rows in splits.items()}}), flush=True)
    return experiment


def make_engine(device):
    from backend.app.asr.engine import QwenCommandEngine
    model = ROOT / "models/echora-qwen3-asr-command-v3"
    if device not in {"mps", "cpu"}:
        raise ValueError("This experiment is Mac-only; no remote or CUDA backend is allowed")
    return QwenCommandEngine(model / "foundation", model / "adapter/adapter.safetensors",
                             model / "evaluation/qwen_command_v3.json", device)


def cache_path(run, row):
    return run / "cache" / (row["id"] + ".json")


def load_cache(run, row, identity=None):
    path = cache_path(run, row)
    cached = read_json(path)
    if cached["row_sha256"] != digest(row) or (identity and cached["identity"] != identity):
        raise ValueError(f"Stale cached decoding: {row['id']}")
    if "raw_sha256" in cached and digest(cached["raw"]) != cached["raw_sha256"]:
        raise ValueError(f"Cached literal evidence changed: {row['id']}")
    feature_path = path.with_suffix(".npy")
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    verify_file(feature_path, cached["feature_sha256"])
    verify_file(Path(row["audio_filepath"]), cached["audio_sha256"])
    return cached


def features_for(run, row):
    features = np.load(cache_path(run, row).with_suffix(".npy"), allow_pickle=False)
    return torch.from_numpy(features)


def cache_storage_preflight(run, experiment, remaining):
    """Check whole-recording storage before loading Qwen; never shrink the corpus."""
    measured = []
    for path in (run / "cache").glob("*.json"):
        entry = read_json(path)
        feature = path.with_suffix(".npy")
        if feature.is_file() and entry.get("audio_seconds", 0) > 0:
            measured.append((feature.stat().st_size + path.stat().st_size) / entry["audio_seconds"])
    # The ordered Qwen feature sequence is ~13 frames/sec x2048 float32.
    # Include both observed overhead and a 25% margin, plus checkpoint/reserve.
    bytes_per_second = max([13 * 2048 * 4, *measured])
    estimated = int(sum(float(row["duration"]) for row in remaining) * bytes_per_second)
    required = int(estimated * 1.25) + (64 + 512) * 1024 ** 2
    free = shutil.disk_usage(run).free
    report = {"remaining_recordings": len(remaining), "estimated_cache_bytes": estimated,
              "required_free_bytes": required, "available_bytes": free,
              "margin": 1.25, "checkpoint_and_reserve_bytes": (64 + 512) * 1024 ** 2}
    if free < required:
        raise OSError(28, f"Whole-recording cache requires {required / 1024 ** 3:.2f} GiB free; "
                      f"only {free / 1024 ** 3:.2f} GiB is available. Free local storage or use "
                      "a writable attached-drive run directory. Completed cache is preserved.")
    return report


def cache_rows(run, experiment, rows, device="mps", engine=None):
    from backend.app.audio import decode_audio
    remaining = []
    for row in rows:
        if cache_path(run, row).exists():
            cached = load_cache(run, row, experiment["identity"])
            if cached["raw"]["device"] != decoding_device(device):
                raise ValueError("Cached ASR device or precision differs from the requested decoding route")
        else:
            remaining.append(row)
    if not remaining:
        return []
    storage = cache_storage_preflight(run, experiment, remaining)
    print(json.dumps({"stage": "storage_preflight", **storage}), flush=True)
    owned_engine = engine is None
    engine = engine or make_engine(device)
    if (engine.prompt != experiment["identity"]["literal_prompt"]
            or engine.max_tokens != experiment["identity"]["max_new_tokens"]
            or experiment["config"]["beams"] != experiment["identity"]["beams"]):
        raise ValueError("Live Qwen decoding settings differ from the frozen experiment")
    timings = []
    for index, row in enumerate(remaining, 1):
        started = time.perf_counter()
        source = Path(row["audio_filepath"])
        audio_bytes = source.read_bytes()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        waveform = decode_audio(audio_bytes, source.name, experiment["config"]["maximum_audio_seconds"])
        # Commit genuine literal evidence before feature extraction. A feature
        # failure or interrupted test resume must not decode a recording twice.
        decoded_path = run / "decodings" / (row["id"] + ".json")
        reused_decoding = decoded_path.exists()
        if decoded_path.exists():
            decoded = read_json(decoded_path)
            if (decoded["identity"] != experiment["identity"] or decoded["row_sha256"] != digest(row)
                    or decoded["audio_sha256"] != audio_hash or digest(decoded["raw"]) != decoded["raw_sha256"]
                    or decoded["raw"]["device"] != decoding_device(device)):
                raise ValueError(f"Frozen literal evidence changed: {row['id']}")
            raw = decoded["raw"]
        else:
            raw = engine.transcribe(waveform, experiment["config"]["beams"]).model_dump()
            atomic_json(decoded_path, {"identity": experiment["identity"], "row_sha256": digest(row),
                                      "audio_sha256": audio_hash, "raw": raw, "raw_sha256": digest(raw)})
        before_features = time.perf_counter()
        features = engine.extract_features(waveform).numpy()
        feature_seconds = time.perf_counter() - before_features
        path = cache_path(run, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, features, allow_pickle=False)
        temporary.replace(path.with_suffix(".npy"))
        release_device_cache(device)
        elapsed = time.perf_counter() - started
        if reused_decoding: elapsed += raw["decode_seconds"]
        cached = {"id": row["id"], "identity": experiment["identity"], "row_sha256": digest(row),
                  "audio_sha256": audio_hash, "raw": raw, "raw_sha256": digest(raw),
                  "feature_sha256": sha256(path.with_suffix(".npy")), "feature_shape": list(features.shape),
                  "elapsed_seconds": elapsed, "feature_seconds": feature_seconds,
                  "decoded_evidence_reused": reused_decoding,
                  "memory": memory_report(),
                  "audio_seconds": len(waveform) / 16000}
        atomic_json(path, cached)
        timings.append(cached)
        print(json.dumps({"stage": "cache", "complete": index, "remaining": len(remaining) - index,
                          "id": row["id"], "seconds": round(elapsed, 3),
                          "audio_seconds": cached["audio_seconds"]}), flush=True)
    if owned_engine:
        del engine
        if device == "mps": torch.mps.empty_cache()
    return timings


def benchmark(run, experiment, device):
    rows = sorted(experiment["splits"]["train"], key=lambda row: (row["duration"], row["id"]))
    # Twelve evenly-spaced duration quantiles include long examples without opening M04.
    selected = [rows[min(len(rows) - 1, int((i + 0.5) * len(rows) / 12))] for i in range(12)]
    cache_rows(run, experiment, selected, device)
    measured = [load_cache(run, row) for row in selected]
    total_rows = [row for name in SPLITS_TO_FIT for row in experiment["splits"][name]]
    rates = [(entry["audio_seconds"], entry["elapsed_seconds"]) for entry in measured]
    estimate = sum(min(rates, key=lambda pair: abs(pair[0] - row["duration"]))[1] for row in total_rows)
    report = {"device": device, "samples": len(selected), "train_cache_utterances": len(total_rows),
              "estimated_train_cache_hours": estimate / 3600, "estimate_is_not_a_guarantee": True,
              "memory": memory_report(),
              "rows": [{"id": item["id"], "audio_seconds": item["audio_seconds"],
                        "elapsed_seconds": item["elapsed_seconds"], "feature_seconds": item["feature_seconds"],
                        "feature_shape": item["feature_shape"]} for item in measured]}
    atomic_json(run / "benchmark.json", report)
    print(json.dumps(report), flush=True)
    return report


def homophone_key(text):
    small = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
    tens = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
    ordinals = ("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth").split()
    words = []
    for word in normalize(text).split():
        if word.isdigit() and int(word) < 100:
            number = int(word)
            words.extend([small[number]] if number < 20 else [tens[number // 10]] + ([small[number % 10]] if number % 10 else []))
        elif re.fullmatch(r"\d+(?:st|nd|rd|th)", word) and int(word[:-2]) < 20:
            words.append(ordinals[int(word[:-2])])
        else:
            words.append(word)
    return tuple(next((min(group) for group in HOMOPHONES if word in group), word) for word in words)


def training_name_candidates(rows):
    """Capitalized interior tokens from training only; heuristic, not an entity label."""
    counts = Counter()
    for row in rows:
        words = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", row["text"])
        for word in words[1:]:
            base = word.removesuffix("'s").removesuffix("’s")
            if len(base) >= 3 and base.istitle(): counts[base.casefold()] += 1
    excluded = {"the", "and", "but", "will", "are", "could", "would", "without", "after", "before",
                "new", "north", "south", "east", "west", "city", "school", "district", "county",
                "river", "lake", "united", "states", "university", "library", "club", "department"}
    return sorted(word for word, count in counts.items() if count >= 2 and word not in excluded)


def phonetic_key(text):
    """A transparent English Soundex-like mining heuristic, not phonetic truth."""
    groups = {letter: str(index) for index, letters in enumerate(("bfpv", "cgjkqsxz", "dt", "l", "mn", "r"), 1) for letter in letters}
    output = []
    for word in normalize(text).split():
        codes = [groups.get(letter, "0") for letter in word]
        collapsed = [code for index, code in enumerate(codes) if index == 0 or code != codes[index - 1]]
        output.append((word[:1] + "".join(code for code in collapsed[1:] if code != "0") + "000")[:4])
    return " ".join(output)


def training_pairs(row, cached, vocabulary, maximum=8, names=(), phonetic=None, *, with_sources=False):
    reference = normalize(row["text"])
    pools = {"incorrect_beam": [item["literal_text"] for item in cached["raw"]["hypotheses"]],
             "similar_phrase": difflib.get_close_matches(reference, vocabulary, n=8, cutoff=0.35),
             "meaning_change": [], "negation": [], "quantity": [], "name": [],
             "contextual_distractor": [item for item in vocabulary if item.split()[:1] == reference.split()[:1]][:8]}
    if phonetic is not None:
        matches = difflib.get_close_matches(phonetic_key(reference), phonetic, n=5, cutoff=0.5)
        pools["similar_sounding_phrase"] = [text for key in matches for text in phonetic[key]][:12]
    # Controlled meaning changes supplement natural beam confusions.
    replacements = {"water": "coffee", "tea": "coffee", "coffee": "tea", "mother": "father",
                    "brother": "sister", "dog": "cat", "man": "woman", "yes": "no",
                    "left": "right", "hot": "cold"}
    quantities = {"one": "three", "two": "five", "three": "two", "four": "six", "five": "two",
                  "six": "four", "seven": "three", "eight": "two", "nine": "five", "ten": "two"}
    words = reference.split()
    for index, word in enumerate(words):
        if word in replacements:
            changed = words[:]; changed[index] = replacements[word]; pools["meaning_change"].append(" ".join(changed))
        if word in quantities or re.fullmatch(r"\d+(?:st|nd|rd|th)?", word):
            changed = words[:]
            changed[index] = quantities.get(word) or re.sub(r"\d+", lambda match: str(int(match[0]) + 1), word)
            pools["quantity"].append(" ".join(changed))
        base = word.removesuffix("'s")
        if base in names:
            alternative = next((name for name in names if name != base and homophone_key(name) != homophone_key(base)), None)
            if alternative:
                changed = words[:]; changed[index] = alternative + ("'s" if word.endswith("'s") else "")
                pools["name"].append(" ".join(changed))
    if "not" in words:
        pools["negation"].append(" ".join(word for word in words if word != "not"))
    elif len(words) > 1:
        pools["negation"].append(" ".join(words[:1] + ["not"] + words[1:]))
        pools["meaning_change"].append(" ".join(reversed(words)))
    # The deterministic unrelated fallback is training-only and avoids empty contrasts.
    pools["contextual_distractor"] += vocabulary[:5]
    negatives, sources, seen = [], [], {reference}
    # Round-robin categories ensure natural beams cannot consume the entire budget.
    while any(pools.values()) and len(negatives) < maximum:
        for category, pool in pools.items():
            while pool:
                item = normalize(pool.pop(0))
                if item and item not in seen and homophone_key(item) != homophone_key(reference):
                    seen.add(item); negatives.append(item); sources.append(category); break
            if len(negatives) == maximum: break
    if not negatives: raise ValueError(f"No safe training contrast for {row['id']}")
    return ([reference] + negatives, sources) if with_sources else [reference] + negatives


def infer_records(model, run, rows, *, keep_logits=True):
    model.eval()
    records = []
    with torch.inference_mode():
        for row in rows:
            cached = load_cache(run, row)
            hypotheses = cached["raw"]["hypotheses"]
            started = time.perf_counter()
            logits = model(features_for(run, row), [item["literal_text"] for item in hypotheses]).float().cpu().tolist()
            records.append({"row": row, "raw": cached["raw"],
                            "scorer_seconds": time.perf_counter() - started,
                            "decode_and_feature_seconds": cached["elapsed_seconds"],
                            "feature_seconds": cached["feature_seconds"],
                            "audio_scores": {item["id"]: logit for item, logit in zip(hypotheses, logits, strict=True)}})
    return records


def selection_metrics(model, run, rows):
    records = infer_records(model, run, rows)
    scores = []
    for item in records:
        best = max(item["raw"]["hypotheses"], key=lambda hyp: item["audio_scores"][hyp["id"]])
        scores.append(score(item["row"]["text"], best["literal_text"]))
    total = aggregate(scores)
    return {"wer": total.wer, "cer": total.cer, "utterances": len(rows)}


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list): return [cpu_tree(item) for item in value]
    if isinstance(value, tuple): return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, model, optimizer, payload):
    temporary = path.with_suffix(".tmp")
    state = {**payload, "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
             "torch_rng": torch.get_rng_state()}
    if torch.backends.mps.is_available(): state["mps_rng"] = torch.mps.get_rng_state()
    torch.save(state, temporary)
    temporary.replace(path)


def save_scorer(path, model):
    from safetensors.torch import save_file
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, temporary)
    temporary.replace(path)


def train(run, experiment, device):
    freeze_training_protocol(run, experiment)
    config = experiment["config"]
    torch.manual_seed(config["seed"])
    model = AcousticTextScorer(config["scorer"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rows = experiment["splits"]["train"]
    vocabulary = sorted({normalize(row["text"]) for row in rows})
    names = training_name_candidates(rows)
    phonetic = defaultdict(list)
    for text in vocabulary: phonetic[phonetic_key(text)].append(text)
    for name in SPLITS_TO_FIT:
        for row in experiment["splits"][name]: load_cache(run, row, experiment["identity"])
    pair_details = {row["id"]: training_pairs(row, load_cache(run, row), vocabulary, config["maximum_negatives"],
                                             names, phonetic, with_sources=True) for row in rows}
    pairs = {identifier: item[0] for identifier, item in pair_details.items()}
    negative_manifest = {"source_partition": "train", "profile_data_used": False,
                         "names": {"method": "repeated_capitalized_interior_training_tokens", "candidates": names},
                         "category_counts": dict(Counter(category for _, sources in pair_details.values() for category in sources)),
                         "pairs_sha256": digest(pair_details), "pairs": pair_details}
    path = run / "training-pairs.json"
    if path.exists() and read_json(path) != json.loads(json.dumps(negative_manifest)):
        raise ValueError("Frozen acoustic contrast pairs changed")
    if not path.exists(): atomic_json(path, negative_manifest)
    history, epoch_start, cursor, best_wer, stale = [], 1, 0, math.inf, 0
    checkpoint = run / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"] != experiment["identity"]: raise ValueError("Checkpoint identity differs")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        for values in optimizer.state.values():
            for key, val in values.items():
                if torch.is_tensor(val): values[key] = val.to(device)
        torch.set_rng_state(saved["torch_rng"])
        if device == "mps" and "mps_rng" in saved: torch.mps.set_rng_state(saved["mps_rng"])
        history, epoch_start, cursor = saved["history"], saved["epoch"], saved["cursor"]
        best_wer, stale = saved["best_wer"], saved["stale"]
        if saved.get("complete"): return read_json(run / "training.json")
    started = time.perf_counter()
    accumulation = int(config["gradient_accumulation"])
    for epoch in range(epoch_start, int(config["maximum_epochs"]) + 1):
        order = list(rows); random.Random(config["seed"] + epoch).shuffle(order)
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses, count = [], 0
        for index in range(cursor, len(order)):
            row = order[index]; texts = pairs[row["id"]]
            logits = model(features_for(run, row), texts)
            target = torch.tensor([True] + [False] * (len(texts) - 1), device=device)
            loss = contrastive_loss(logits, target)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite contrastive training loss")
            (loss / accumulation).backward(); losses.append(float(loss.detach().cpu())); count += 1
            if count == accumulation or index + 1 == len(order):
                if count != accumulation:
                    for parameter in model.parameters():
                        if parameter.grad is not None: parameter.grad.mul_(accumulation / count)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); count = 0
                release_device_cache(device)
                if ((index + 1) // accumulation) % config["checkpoint_every_steps"] == 0:
                    save_checkpoint(checkpoint, model, optimizer, {"identity": experiment["identity"],
                        "history": history, "epoch": epoch, "cursor": index + 1, "best_wer": best_wer, "stale": stale})
                    print(json.dumps({"stage": "train", "epoch": epoch, "cursor": index + 1,
                                      "utterances": len(order), "mean_loss": sum(losses) / len(losses)}), flush=True)
        metrics = selection_metrics(model, run, experiment["splits"]["selection"])
        metrics.update(epoch=epoch, mean_loss=sum(losses) / len(losses) if losses else None,
                       elapsed_seconds=time.perf_counter() - started, memory=memory_report())
        history.append(metrics)
        if metrics["wer"] < best_wer - 1e-12:
            best_wer, stale = metrics["wer"], 0
            save_scorer(run / "scorer-best.safetensors", model)
            atomic_json(run / "selected.json", metrics)
        else: stale += 1
        complete = epoch == config["maximum_epochs"] or (epoch >= config["minimum_epochs"] and stale >= config["patience"])
        atomic_json(run / "training.json", {"history": history, "selected": read_json(run / "selected.json"), "device": device,
                                            "complete": complete, "parameters": sum(p.numel() for p in model.parameters())})
        save_checkpoint(checkpoint, model, optimizer, {"identity": experiment["identity"], "history": history,
                        "epoch": epoch + 1, "cursor": 0, "best_wer": best_wer, "stale": stale, "complete": complete})
        print(json.dumps({"stage": "epoch", **metrics, "complete": complete}), flush=True)
        cursor = 0
        if complete: break
    return read_json(run / "training.json")


def load_scorer(run, experiment, device):
    from safetensors.torch import load_file
    model = AcousticTextScorer(experiment["config"]["scorer"])
    model.load_state_dict(load_file(run / "scorer-best.safetensors"))
    return model.to(device).eval()


def context_scenarios(run, records):
    """Use the application's real local semantic retrieval on frozen synthetic entries."""
    from communication.backend.retrieval import rank_entries, MiniLMEmbedder
    bank = read_json(run / "context_bank.json")
    global _CONTEXT_ENCODER
    if _CONTEXT_ENCODER is None:
        underlying = MiniLMEmbedder()
        class CachedEncoder:
            model_revision = underlying.model_revision
            dimension = underlying.dimension
            def __init__(self): self.cache = {}
            def encode_chunks(self, text):
                if text not in self.cache: self.cache[text] = underlying.encode_chunks(text)
                return self.cache[text]
        _CONTEXT_ENCODER = CachedEncoder()
    embedder = _CONTEXT_ENCODER
    now = bank["frozen_at"]
    entries = bank["entries"]
    selection = {"recipient": "Alex", "setting": "home", "scenario": "general", "listener": "familiar"}
    variants = {
        "relevant": entries[:80],
        "wrong_recipient": [{**entry, "scope": {"recipient": "Pat"}} for entry in entries[:80]],
        "stale": [{**entry, "created_at": now - 365 * 86400} for entry in entries[:80]],
        "contradictory": entries[:80] + [{**entry, "source_id": entry["source_id"] + "-contradiction",
                           "text": "not " + entry["text"], "wording": "not " + entry["wording"],
                           "content_hash": digest("not " + entry["text"])} for entry in entries[:80]],
        "misleading": entries[80:],
    }
    output = {}
    for record in records:
        maps = {"empty": {}}
        for name, items in variants.items():
            result = rank_entries(record["raw"]["hypotheses"], items, selection=selection,
                                  language="en", embedder=embedder, now=now,
                                  vector_loader=lambda entry: embedder.encode_chunks(entry["text"]))
            maps[name] = result.get("candidates", result)
        output[record["row"]["id"]] = maps
    return output


def winner(result, record):
    identifier = result["ordered_hypothesis_ids"][0]
    return next(item["literal_text"] for item in record["raw"]["hypotheses"] if item["id"] == identifier)


def ranked(record, fusion, context=None, calibration=None, permit=False):
    return rank_scores(record["raw"]["hypotheses"], record["audio_scores"], context,
                       fusion, calibration, permit_selection=permit,
                       audio_quality=record["raw"].get("audio_quality"))


def correctness(text, record):
    return normalize(text) == normalize(record["row"]["text"])


def fit_gate(examples, prior_intercept):
    """Regularized logistic mixing gate; target sets are minimum-WER beams."""
    from scipy.optimize import minimize
    features = np.asarray([item["features"] for item in examples], dtype=float)
    centers, scales = features.mean(0), np.maximum(features.std(0), 0.01)
    centers[0], scales[0] = 0, 1
    normalized = np.clip((features - centers) / scales, -10, 10)
    prior = np.zeros(features.shape[1]); prior[0] = prior_intercept
    weights = np.asarray([item["weight"] for item in examples], dtype=float); weights /= weights.sum()
    def objective(coefficients):
        gates = 1 / (1 + np.exp(-np.clip(normalized @ coefficients, -60, 60)))
        loss, gradient = 0.0, np.zeros_like(coefficients)
        for index, (item, gate) in enumerate(zip(examples, gates, strict=True)):
            logits = item["base"] + gate * item["delta"]
            exponential = np.exp(logits - logits.max()); probability = exponential / exponential.sum()
            positive = probability * item["positive"]
            positive /= positive.sum()
            probability_mass = probability[item["positive"]].sum()
            loss -= weights[index] * math.log(max(float(probability_mass), 1e-30))
            derivative = float(np.sum((probability - positive) * item["delta"]))
            gradient += weights[index] * derivative * gate * (1 - gate) * normalized[index]
        difference = coefficients - prior
        return loss + 0.01 * float(difference @ difference), gradient + 0.02 * difference
    fitted = minimize(objective, prior, jac=True, method="L-BFGS-B", bounds=[(-10, 10)] * len(prior))
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError("Adaptive fusion gate did not converge")
    return {"coefficients": fitted.x.tolist(), "centers": centers.tolist(), "scales": scales.tolist(),
            "regularization": 0.01, "fit_partition": "F03-fusion"}


def oracle_mask(record):
    errors = [score(record["row"]["text"], item["literal_text"]).word_errors for item in record["raw"]["hypotheses"]]
    return np.asarray(errors) == min(errors)


def fit_fusion(records, scenarios, maximum_audio_seconds=45):
    asr = np.array([item["sequence_score"] for record in records for item in record["raw"]["hypotheses"]])
    audio = np.array([value for record in records for value in record["audio_scores"].values()])
    positives = [record["audio_scores"][item["id"]] for record in records for item in record["raw"]["hypotheses"]
                 if correctness(item["literal_text"], record)]
    if not positives: raise ValueError("No exact positive beam in fusion development")
    base = {"asr_mean": float(asr.mean()), "asr_scale": max(float(asr.std()), 1e-3),
            "audio_mean": float(audio.mean()), "audio_scale": max(float(audio.std()), 1e-3),
            "audio_support_floor": float(np.quantile(positives, 0.05)), "tie_margin": 0.1, "context_weight": 0.0,
            "maximum_audio_seconds": maximum_audio_seconds}
    candidates = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        config = {**base, "asr_weight": alpha}
        loss = aggregate([score(record["row"]["text"], winner(ranked(record, config), record)) for record in records]).wer
        candidates.append((loss, -alpha, config))
    _, _, selected = min(candidates, key=lambda item: item[:2])
    groups = Counter(record["row"]["group"] for record in records)
    examples = []
    for record in records:
        asr_values = np.asarray([(item["sequence_score"] - base["asr_mean"]) / base["asr_scale"]
                                 for item in record["raw"]["hypotheses"]])
        audio_values = np.asarray([(record["audio_scores"][item["id"]] - base["audio_mean"]) / base["audio_scale"]
                                   for item in record["raw"]["hypotheses"]])
        examples.append({"features": gate_features(asr_values.tolist(), audio_values.tolist(), record["raw"].get("audio_quality")),
                         "base": audio_values, "delta": asr_values - audio_values,
                         "positive": oracle_mask(record), "weight": 1 / groups[record["row"]["group"]]})
    initial = min(0.95, max(0.05, selected["asr_weight"]))
    selected["acoustic_gate"] = fit_gate(examples, math.log(initial / (1 - initial)))
    context_examples = []
    for record in records:
        for scenario in SCENARIOS[1:]:
            retrieval = scenarios[record["row"]["id"]][scenario]
            preliminary = ranked(record, selected, retrieval)
            by_id = {item["hypothesis_id"]: item for item in preliminary["scores"]}
            ordered = [by_id[item["id"]] for item in record["raw"]["hypotheses"]]
            context_examples.append({"features": context_gate_features(ordered, retrieval, record["raw"].get("audio_quality")),
                "base": np.asarray([item["acoustic_score"] for item in ordered]),
                "delta": np.asarray([0.2 * item["context_relevance"] if item["acoustically_supported"] else 0 for item in ordered]),
                "positive": oracle_mask(record), "weight": 1 / (groups[record["row"]["group"]] * (len(SCENARIOS) - 1))})
    selected["context_gate"] = fit_gate(context_examples, -1.0)
    context_trials = []
    for margin in (0.05, 0.1, 0.2):
        for weight in (0.0, 0.05, 0.1, 0.2):
            config = {**selected, "tie_margin": margin, "context_weight": weight}
            losses, harmful = [], 0
            for record in records:
                plain = winner(ranked(record, config), record)
                for scenario in SCENARIOS[1:]:
                    contextual = winner(ranked(record, config, scenarios[record["row"]["id"]][scenario]), record)
                    harmful += int(correctness(plain, record) and not correctness(contextual, record))
                    losses.append(score(record["row"]["text"], contextual))
            if harmful == 0:
                context_trials.append((aggregate(losses).wer, weight, margin, config))
    selected = min(context_trials, key=lambda item: item[:3])[3]
    return {**selected, "fit_partition": "F03-fusion", "harmful_context_flips_on_fit": 0}


def without_acoustic(record):
    """The context-only ablation cannot inspect the learned scorer's outputs."""
    return {**record, "audio_scores": {item["id"]: 0.0 for item in record["raw"]["hypotheses"]}}


def fit_context_only(records, scenarios, fusion):
    records = [without_acoustic(record) for record in records]
    base = {"asr_mean": fusion["asr_mean"], "asr_scale": fusion["asr_scale"],
            "maximum_audio_seconds": fusion["maximum_audio_seconds"],
            "audio_mean": 0.0, "audio_scale": 1.0, "asr_weight": 1.0,
            "audio_support_floor": 0.0, "tie_margin": 0.1, "context_weight": 0.0}
    groups = Counter(record["row"]["group"] for record in records)
    examples = []
    for record in records:
        for scenario in SCENARIOS[1:]:
            retrieval = scenarios[record["row"]["id"]][scenario]
            scores = ranked(record, base, retrieval)["scores"]
            by_id = {item["hypothesis_id"]: item for item in scores}
            ordered = [by_id[item["id"]] for item in record["raw"]["hypotheses"]]
            examples.append({"features": context_gate_features(ordered, retrieval, record["raw"].get("audio_quality")),
                "base": np.asarray([item["acoustic_score"] for item in ordered]),
                "delta": np.asarray([0.2 * item["context_relevance"] if item["acoustically_supported"] else 0 for item in ordered]),
                "positive": oracle_mask(record), "weight": 1 / (groups[record["row"]["group"]] * (len(SCENARIOS) - 1))})
    base["context_gate"] = fit_gate(examples, -1.0)
    trials = []
    for margin in (0.05, 0.1, 0.2):
        for weight in (0.0, 0.05, 0.1, 0.2):
            config = {**base, "tie_margin": margin, "context_weight": weight}
            losses, harmful = [], 0
            for record in records:
                plain = winner(ranked(record, config), record)
                for scenario in SCENARIOS[1:]:
                    text = winner(ranked(record, config, scenarios[record["row"]["id"]][scenario]), record)
                    harmful += int(correctness(plain, record) and not correctness(text, record))
                    losses.append(score(record["row"]["text"], text))
            if not harmful: trials.append((aggregate(losses).wer, weight, margin, config))
    return {**min(trials, key=lambda item: item[:3])[3], "fit_partition": "F03-fusion",
            "scorer_input": "absent", "harmful_context_flips_on_fit": 0}


def calibration_examples(records, scenarios, fusion):
    groups = Counter(record["row"]["group"] for record in records)
    examples = []
    for record in records:
        for scenario in SCENARIOS:
            result = ranked(record, fusion, scenarios[record["row"]["id"]][scenario])
            examples.append({"logit": result["scores"][0]["acoustic_score"],
                             "margin": result["scores"][0]["selection_margin"],
                             "correct": correctness(winner(result, record), record),
                             "all_beams_wrong": not any(correctness(item["literal_text"], record)
                                                        for item in record["raw"]["hypotheses"]),
                             "group": record["row"]["group"],
                             "weight": 1 / (groups[record["row"]["group"]] * len(SCENARIOS))})
    return examples


def fit_calibration(examples, config):
    from scipy.optimize import minimize
    correct_groups = {item["group"] for item in examples if item["correct"]}
    incorrect_groups = {item["group"] for item in examples if not item["correct"]}
    all_wrong_groups = {item["group"] for item in examples if item.get("all_beams_wrong")}
    report = {"valid": False, "audit_passed": False, "threshold": config["acceptance_threshold"],
              "correct_groups": len(correct_groups), "incorrect_groups": len(incorrect_groups),
              "all_beams_wrong_groups": len(all_wrong_groups),
              "fit_partition": "F03-calibration", "probability_scope": "single-development-speaker"}
    minimum = config["minimum_calibration_groups_per_class"]
    if min(len(correct_groups), len(incorrect_groups)) < minimum:
        return {**report, "reason": "insufficient_calibration_groups"}
    if not all_wrong_groups:
        return {**report, "reason": "all_beams_wrong_calibration_coverage_missing"}
    x = np.array([item["logit"] for item in examples]); margins = np.array([item["margin"] for item in examples])
    y = np.array([item["correct"] for item in examples], dtype=float)
    weights = np.array([item["weight"] for item in examples], dtype=float); weights /= weights.sum()
    def objective(params):
        logits = params[0] * x + params[1] * margins + params[2]
        return float(np.sum(weights * (np.logaddexp(0, logits) - y * logits)) + 0.001 * np.sum(params ** 2))
    fitted = minimize(objective, np.array([1.0, 0.0, 0.0]), method="L-BFGS-B", bounds=((0, 100), (0, 100), (-100, 100)))
    if not fitted.success or not np.isfinite(fitted.x).all():
        return {**report, "reason": "calibration_fit_failed"}
    return {**report, "valid": True, "slope": float(fitted.x[0]), "margin_slope": float(fitted.x[1]),
            "intercept": float(fitted.x[2]),
            "reason": "awaiting_untouched_acceptance_audit"}


def audit_acceptance(records, scenarios, fusion, calibration, config):
    from scipy.stats import beta
    accepted, failed = set(), set()
    harmful = 0
    for record in records:
        plain = winner(ranked(record, fusion), record)
        for scenario in SCENARIOS:
            result = ranked(record, fusion, scenarios[record["row"]["id"]][scenario], calibration, True)
            contextual = winner(result, record)
            harmful += int(correctness(plain, record) and not correctness(contextual, record))
            if result["decision"] == "selected":
                accepted.add(record["row"]["group"])
                if not correctness(contextual, record): failed.add(record["row"]["group"])
    count, errors = len(accepted), len(failed)
    upper = float(beta.ppf(0.95, errors + 1, count - errors)) if count > errors else 1.0
    passed = bool(calibration.get("valid") and count >= config["minimum_audit_accepted_groups"]
                  and errors == 0 and harmful == 0)
    return {"audit_passed": passed, "audit_partition": "F03-audit", "accepted_groups": count,
            "incorrect_groups_audit": errors, "harmful_context_flips": harmful,
            "observed_false_accept_rate": errors / count if count else None,
            "one_sided_95_percent_upper_false_accept_bound": upper,
            "not_a_five_percent_risk_guarantee": True,
            "reason": "empirical_acceptance_audit_passed" if passed else "acceptance_audit_not_passed"}


def verify_frozen_fit(run, experiment):
    fitted = read_json(run / "fit.json")
    manifest = read_json(run / "artifact/manifest.json")
    if fitted["artifact"] != manifest or manifest["artifact_id"] != digest([experiment["identity"], manifest["files"]])[:24]:
        raise ValueError("Frozen fitting artifact identity changed")
    for key, filename in (("fusion", "fusion.json"), ("calibration", "calibration.json"),
                          ("context_only_fusion", "context-only-fusion.json")):
        path = run / "artifact" / filename
        if sha256(path) != manifest["files"][filename] or fitted[key] != read_json(path):
            raise ValueError(f"Frozen {key} parameters changed after audit")
    if (sha256(run / "artifact/scorer.safetensors") != manifest["files"]["scorer.safetensors"]
            or sha256(run / "scorer-best.safetensors") != manifest["files"]["scorer.safetensors"]
            or sha256(run / "context_bank.json") != manifest["context_bank_sha256"]
            or sha256(run / "protocol.json") != manifest["protocol_sha256"]):
        raise ValueError("Frozen scorer, context fixture or protocol changed after audit")
    return fitted


def fit(run, experiment, device):
    protocol = freeze_training_protocol(run, experiment)
    if (run / "fit.json").exists():
        frozen = verify_frozen_fit(run, experiment)
        if frozen["artifact"]["config_sha256"] != experiment["identity"]["config_sha256"]:
            raise ValueError("Frozen fit belongs to a different experiment")
        print(json.dumps({"stage": "fit", "already_frozen": True}), flush=True)
        return frozen
    if not read_json(run / "training.json").get("complete"):
        raise ValueError("Training must finish before fitting fusion or calibration")
    # Production serves the small scorer on CPU; fitting must use that route.
    model = load_scorer(run, experiment, "cpu")
    sets = {name: infer_records(model, run, experiment["splits"][name]) for name in ("fusion", "calibration", "audit")}
    contexts = {name: context_scenarios(run, records) for name, records in sets.items()}
    fusion = fit_fusion(sets["fusion"], contexts["fusion"], experiment["config"]["maximum_audio_seconds"])
    context_only = fit_context_only(sets["fusion"], contexts["fusion"], fusion)
    calibration = fit_calibration(calibration_examples(sets["calibration"], contexts["calibration"], fusion), experiment["config"])
    calibration.update(audit_acceptance(sets["audit"], contexts["audit"], fusion, calibration, experiment["config"]))
    artifact = run / "artifact"; artifact.mkdir(parents=True, exist_ok=True)
    save_scorer(artifact / "scorer.safetensors", model)
    atomic_json(artifact / "fusion.json", fusion); atomic_json(artifact / "calibration.json", calibration)
    atomic_json(artifact / "context-only-fusion.json", context_only)
    files = {name: sha256(artifact / name) for name in ("scorer.safetensors", "fusion.json", "calibration.json", "context-only-fusion.json")}
    manifest = {"format": FORMAT, "trained": True, **experiment["identity"], "files": files,
                "scoring_device": "cpu", "training_device": read_json(run / "training.json")["device"],
                "asr_device": protocol["asr_device"],
                "scorer_config": experiment["config"]["scorer"], "selected": read_json(run / "selected.json"),
                "context_bank_sha256": sha256(run / "context_bank.json"),
                "protocol_sha256": sha256(run / "protocol.json"),
                "artifact_id": digest([experiment["identity"], files])[:24]}
    atomic_json(artifact / "manifest.json", manifest)
    atomic_json(run / "fit.json", {"fusion": fusion, "context_only_fusion": context_only,
                                  "calibration": calibration, "artifact": manifest})
    print(json.dumps({"stage": "fit", "calibration": calibration, "fusion": fusion}), flush=True)


def binomial_interval(errors, count):
    from scipy.stats import beta
    return [float(beta.ppf(0.025, errors, count - errors + 1)) if errors else 0.0,
            float(beta.ppf(0.975, errors + 1, count - errors)) if errors < count else 1.0]


def metric_vector(rows, modes):
    values = np.zeros(3 + 3 * len(modes), dtype=float)
    for row in rows:
        baseline = score(row["reference"], row["outputs"][modes[0]])
        values[:3] += [baseline.reference_words, baseline.reference_characters, 1]
        for index, mode in enumerate(modes):
            measured = score(row["reference"], row["outputs"][mode])
            values[3 + 3 * index:6 + 3 * index] += [measured.word_errors, measured.character_errors,
                                                  normalize(row["reference"]) == normalize(row["outputs"][mode])]
    return values


def paired_group_intervals(rows, *, seed, replicates=2000):
    """Paired prompt-cluster bootstrap; repeated recordings stay together."""
    modes = ("acoustic_only", "reranked", "context_assisted", "integrated")
    groups = defaultdict(list)
    for row in rows: groups[row["group"]].append(row)
    matrices = np.asarray([metric_vector(items, modes) for _, items in sorted(groups.items())])
    if not len(matrices): return []
    randomizer = np.random.default_rng(seed)
    samples = randomizer.integers(0, len(matrices), size=(replicates, len(matrices)))
    totals = matrices[samples].sum(axis=1)
    point = matrices.sum(axis=0)
    ratios, points = {}, {}
    for index, mode in enumerate(modes):
        start = 3 + 3 * index
        ratios[mode] = totals[:, start:start + 3] / np.maximum(totals[:, :3], 1)
        points[mode] = point[start:start + 3] / np.maximum(point[:3], 1)
    report = []
    for mode in modes:
        for index, name in enumerate(("wer", "cer", "exact_match")):
            values = ratios[mode][:, index]
            delta = values - ratios["acoustic_only"][:, index]
            report.append({"mode": mode, "metric": name, "estimate": float(points[mode][index]),
                           "interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
                           "paired_difference_from_asr": float(points[mode][index] - points["acoustic_only"][index]),
                           "paired_difference_interval_95": np.quantile(delta, [0.025, 0.975]).tolist(),
                           "prompt_groups": len(groups), "replicates": replicates,
                           "method": "paired_normalized_prompt_cluster_percentile_bootstrap"})
    return report


def timing_summary(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)), "p95_seconds": float(np.quantile(values, 0.95)),
            "total_seconds": float(values.sum())} if len(values) else {"count": 0}


def evaluate(run, experiment, device):
    protocol = freeze_training_protocol(run, experiment)
    fit_report = verify_frozen_fit(run, experiment)
    if protocol["asr_device"] != decoding_device(device):
        raise ValueError("Held-out decoding device or precision differs from the frozen fitting cache")
    if (run / "evaluation.json").exists():
        report = read_json(run / "evaluation.json")
        if report["artifact_id"] != fit_report["artifact"]["artifact_id"]:
            raise ValueError("Published evaluation uses a different frozen artifact")
        print(json.dumps({"stage": "evaluated", "already_frozen": True}), flush=True)
        return report
    # First and only test decoding follows the frozen fit/audit artifact.
    test_rows = [row for name in SPLITS_TO_TEST for row in experiment["splits"][name]]
    cache_rows(run, experiment, test_rows, device)
    model = load_scorer(run, experiment, "cpu")
    records = infer_records(model, run, test_rows)
    started = time.perf_counter()
    contexts = context_scenarios(run, records)
    retrieval_seconds = time.perf_counter() - started
    fusion, calibration = fit_report["fusion"], fit_report["calibration"]
    context_only = fit_report["context_only_fusion"]
    fusion_timings = []
    rows, buckets = [], defaultdict(list)
    for record in records:
        source = record["row"]
        baseline = record["raw"]["hypotheses"][0]["literal_text"]
        acoustic_result = ranked(record, fusion)
        acoustic = winner(acoustic_result, record)
        for scenario in SCENARIOS:
            retrieval = contexts[source["id"]][scenario]
            assisted = ranked(without_acoustic(record), context_only, retrieval)
            started = time.perf_counter()
            integrated = ranked(record, fusion, retrieval, calibration, calibration.get("audit_passed", False))
            fusion_timings.append(time.perf_counter() - started)
            outputs = {"acoustic_only": baseline, "reranked": acoustic,
                       "context_assisted": winner(assisted, record), "integrated": winner(integrated, record)}
            for mode, text in outputs.items(): buckets[(source["condition"], scenario, mode)].append(score(source["text"], text))
            rows.append({"id": source["id"], "speaker": source["speaker"], "condition": source["condition"],
                         "group": source["group"], "scenario": scenario, "reference": source["text"],
                         "outputs": outputs, "raw": record["raw"], "ranking": integrated,
                         "context_hits": retrieval,
                         "harmful_context_flip": correctness(acoustic, record) and not correctness(outputs["integrated"], record),
                         "harmful_context_only_flip": correctness(baseline, record) and not correctness(outputs["context_assisted"], record),
                         "all_beams_wrong": not any(correctness(hyp["literal_text"], record) for hyp in record["raw"]["hypotheses"])})
    metrics = []
    for (condition, scenario, mode), items in sorted(buckets.items()):
        total = aggregate(items)
        metrics.append({"condition": condition, "scenario": scenario, "mode": mode, "utterances": len(items),
                        "wer": total.wer, "cer": total.cer, "substitutions": total.word_substitutions,
                        "deletions": total.word_deletions, "insertions": total.word_insertions,
                        "exact_match": sum(item.word_errors == 0 for item in items) / len(items)})
    audit, intervals, oracle = [], [], []
    for condition in sorted({row["condition"] for row in rows}):
        for scenario in SCENARIOS:
            subset = [row for row in rows if row["condition"] == condition and row["scenario"] == scenario]
            accepted = [row for row in subset if row["ranking"]["decision"] == "selected"]
            errors = sum(normalize(row["outputs"]["integrated"]) != normalize(row["reference"]) for row in accepted)
            accepted_groups = {row["group"] for row in accepted}
            failed_groups = {row["group"] for row in accepted if normalize(row["outputs"]["integrated"]) != normalize(row["reference"])}
            wrong_groups = {row["group"] for row in subset if row["all_beams_wrong"]}
            false_wrong_groups = {row["group"] for row in accepted if row["all_beams_wrong"]}
            audit.append({"condition": condition, "scenario": scenario, "utterances": len(subset),
                          "accepted": len(accepted), "incorrect_accepts": errors,
                          "coverage": len(accepted) / len(subset) if subset else 0,
                          "clarification_coverage": 1 - len(accepted) / len(subset) if subset else 0,
                          "accepted_error_rate": errors / len(accepted) if accepted else None,
                          "accepted_prompt_groups": len(accepted_groups), "incorrect_accepted_prompt_groups": len(failed_groups),
                          "accepted_group_error_interval_95": binomial_interval(len(failed_groups), len(accepted_groups)),
                          "harmful_context_flips": sum(row["harmful_context_flip"] for row in subset),
                          "harmful_context_only_flips": sum(row["harmful_context_only_flip"] for row in subset),
                          "all_beams_wrong": sum(row["all_beams_wrong"] for row in subset),
                          "all_beams_wrong_accepts": sum(row["all_beams_wrong"] for row in accepted),
                          "all_beams_wrong_groups": len(wrong_groups), "all_beams_wrong_accepted_groups": len(false_wrong_groups),
                          "all_beams_wrong_false_accept_interval_95": binomial_interval(len(false_wrong_groups), len(wrong_groups))})
            intervals.extend({"condition": condition, "scenario": scenario, **item} for item in
                             paired_group_intervals(subset, seed=protocol["evaluation"]["bootstrap_seed"],
                                                    replicates=protocol["evaluation"]["bootstrap_replicates"]))
        unique = [row for row in rows if row["condition"] == condition and row["scenario"] == "empty"]
        best = aggregate([min((score(row["reference"], hyp["literal_text"]) for hyp in row["raw"]["hypotheses"]),
                              key=lambda measured: (measured.word_errors, measured.character_errors)) for row in unique])
        oracle.append({"condition": condition, "utterances": len(unique), "wer": best.wer, "cer": best.cer,
                       "exact_beam_coverage": sum(not row["all_beams_wrong"] for row in unique) / len(unique)})
    output = run / "evaluation.jsonl"
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    report = {"artifact_id": fit_report["artifact"]["artifact_id"], "matched_decoding": experiment["identity"],
              "metrics": metrics, "paired_uncertainty": intervals, "oracle": oracle,
              "corpus_eligibility": protocol["corpus_eligibility"],
              "historical_test_exposure": True,
              "acceptance": audit, "limitations": experiment["limitations"] + [
                  "Relevant/misleading labels identify fixed training-only phrase pools; relevance is not guaranteed for each test utterance.",
                  "Uncertainty clusters repeated prompts; this single-speaker TORGO test cannot establish population or clinical performance.",
                  "Reported latency is offline sequential execution; it does not measure microphone-to-speech latency."],
              "memory": memory_report(),
              "latency": {"decoding_and_features": timing_summary([record["decode_and_feature_seconds"] for record in records]),
                          "feature_extraction": timing_summary([record["feature_seconds"] for record in records]),
                          "acoustic_scorer": timing_summary([record["scorer_seconds"] for record in records]),
                          "fusion": timing_summary(fusion_timings), "context_scenarios_total_seconds": retrieval_seconds,
                          "context_timing_note": "Includes encoder loading and cached fixture vectors across all six controlled conditions."},
              "repeated_prompt_overlap": {name: {other: {"shared_prompt_groups": len(
                  {row["group"] for row in experiment["splits"][name]} & {row["group"] for row in experiment["splits"][other]}),
                  "test_utterances_with_shared_prompt": sum(row["group"] in {item["group"] for item in experiment["splits"][other]}
                                                           for row in experiment["splits"][name])}
                  for other in SPLITS_TO_FIT} for name in SPLITS_TO_TEST},
              "protocol_sha256": sha256(run / "protocol.json"),
              "context_bank_sha256": sha256(run / "context_bank.json"), "test_used_for_fitting": False}
    atomic_json(run / "evaluation.json", report)
    print(json.dumps({"stage": "evaluated", **report}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "benchmark", "cache", "train", "fit", "evaluate", "all"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    experiment = prepare(args.run_dir, read_json(args.config))
    if args.stage == "prepare": return
    if args.stage in ("benchmark", "all"): benchmark(args.run_dir, experiment, args.device)
    if args.stage in ("cache", "all"):
        cache_rows(args.run_dir, experiment, [row for name in SPLITS_TO_FIT for row in experiment["splits"][name]], args.device)
    if args.stage in ("train", "all"): train(args.run_dir, experiment, args.device)
    if args.stage in ("fit", "all"): fit(args.run_dir, experiment, args.device)
    if args.stage in ("evaluate", "all"): evaluate(args.run_dir, experiment, args.device)


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        if exc.errno != 28: raise
        print(json.dumps({"stage": "blocked", "reason": "local_disk_full", "detail": str(exc),
                          "resume": "Free local disk space or move this complete run folder to an attached drive. Resume the same stage; completed evidence and checkpoints are preserved."}), flush=True)
        raise SystemExit(2) from None
