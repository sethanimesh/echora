"""Build deterministic literal-command compositions from TORGO isolated words.

The generated audio never crosses speakers.  Training, development and test
phrases are disjoint, while F03 and M04 remain completely speaker held out.
These compositions are an augmentation/robustness set, not new clinical data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


WORD = re.compile(r"^[a-z]+(?:'[a-z]+)?$")
MINIMUM_COMPONENT_SECONDS = 0.25
GAP_RANGES = {
    "short": (0.12, 0.35),
    "medium": (0.55, 0.95),
    "long": (1.20, 1.90),
    "very_long": (2.20, 3.20),
}
STRETCH_RANGES = {
    "none": (1.0, 1.0),
    "mild": (1.15, 1.35),
    "strong": (1.40, 1.70),
    "very_strong": (1.75, 2.0),
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def localize_staged_audio(rows: list[dict], pilot_data: Path, cloud_root: Path) -> list[dict]:
    localized = []
    for row in rows:
        cloud_path = Path(row["audio_filepath"])
        try:
            relative = cloud_path.relative_to(cloud_root)
        except ValueError as error:
            raise ValueError(f"Staged path is outside cloud root: {cloud_path}") from error
        local_path = pilot_data / relative
        if not local_path.is_file():
            raise FileNotFoundError(f"Staged source audio is missing: {local_path}")
        item = dict(row)
        item["audio_filepath"] = str(local_path)
        localized.append(item)
    return localized


def normalized_word(text: str) -> str | None:
    value = text.strip().lower().replace("’", "'")
    if value in {"x", "xx", "xxx"} or not WORD.fullmatch(value):
        return None
    return value


def eligible_words(rows: list[dict], allowed_vocabulary: set[str] | None = None) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("condition") != "dysarthric":
            continue
        if float(row.get("duration", 0.0)) < MINIMUM_COMPONENT_SECONDS:
            continue
        word = normalized_word(str(row["text"]))
        if word is None or (allowed_vocabulary is not None and word not in allowed_vocabulary):
            continue
        grouped[str(row["speaker"])][word].append(row)
    if not grouped:
        raise ValueError("No eligible isolated dysarthric words were found")
    return {speaker: dict(words) for speaker, words in grouped.items()}


def trim_word(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if not len(audio):
        raise ValueError("Cannot trim empty audio")
    _, bounds = librosa.effects.trim(
        audio, top_db=38, frame_length=400, hop_length=160
    )
    padding = round(sample_rate * 0.12)
    start = max(0, int(bounds[0]) - padding)
    end = min(len(audio), int(bounds[1]) + padding)
    trimmed = audio[start:end]
    # Avoid aggressive trimming on unusually quiet dysarthric recordings.
    if len(trimmed) < round(sample_rate * 0.18):
        return audio
    return trimmed.astype(np.float32)


def load_word(row: dict, stretch: float, sample_rate: int = 16_000) -> np.ndarray:
    audio, _ = librosa.load(
        row["audio_filepath"], sr=sample_rate, mono=True, dtype=np.float32
    )
    audio = trim_word(np.asarray(audio, dtype=np.float32), sample_rate)
    if stretch > 1.0001:
        audio = librosa.effects.time_stretch(audio, rate=1.0 / stretch)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid stretched word audio for {row['id']}")
    fade = min(round(sample_rate * 0.008), len(audio) // 3)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return np.asarray(audio, dtype=np.float32)


def edge_noise_rms(parts: list[np.ndarray], sample_rate: int) -> float:
    edges = []
    width = max(1, round(sample_rate * 0.05))
    for part in parts:
        edges.extend((part[:width], part[-width:]))
    values = np.concatenate(edges)
    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    return min(max(rms * 0.30, 1e-7), 0.003)


def compose(
    component_rows: list[dict],
    gaps: list[float],
    stretches: list[float],
    rng: np.random.Generator,
    sample_rate: int = 16_000,
) -> np.ndarray:
    parts = [
        load_word(row, stretch, sample_rate)
        for row, stretch in zip(component_rows, stretches)
    ]
    noise_rms = edge_noise_rms(parts, sample_rate)
    output = [parts[0]]
    for gap, part in zip(gaps, parts[1:]):
        count = max(1, round(gap * sample_rate))
        output.append(rng.normal(0.0, noise_rms, count).astype(np.float32))
        output.append(part)
    audio = np.concatenate(output)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32)


def stable_seed(seed: int, *values: object) -> int:
    payload = ":".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def build_split(
    split: str,
    grouped: dict[str, dict[str, list[dict]]],
    count: int,
    destination: Path,
    cloud_audio: Path,
    seed: int,
    forbidden_phrases: set[str],
    maximum_seconds: float,
) -> tuple[list[dict], set[str]]:
    destination.mkdir(parents=True, exist_ok=False)
    speakers = sorted(grouped)
    for speaker in speakers:
        if len(grouped[speaker]) < 20:
            raise ValueError(f"{speaker} has too few isolated words for command composition")
    rows: list[dict] = []
    phrases = set(forbidden_phrases)
    attempts = 0
    limit = max(count * 500, 10_000)
    while len(rows) < count and attempts < limit:
        index = len(rows)
        speaker = speakers[index % len(speakers)]
        rng_seed = stable_seed(seed, split, attempts, speaker)
        py_rng = random.Random(rng_seed)
        np_rng = np.random.default_rng(rng_seed)
        words = sorted(grouped[speaker])
        word_count = 2 if index % 2 == 0 else 3
        selected_words = py_rng.sample(words, word_count)
        phrase = " ".join(selected_words)
        attempts += 1
        if phrase in phrases:
            continue

        gap_name = list(GAP_RANGES)[index % len(GAP_RANGES)]
        gap_min, gap_max = GAP_RANGES[gap_name]
        gaps = [py_rng.uniform(gap_min, gap_max) for _ in range(word_count - 1)]
        stretch_name = list(STRETCH_RANGES)[(index // len(GAP_RANGES)) % len(STRETCH_RANGES)]
        stretch_min, stretch_max = STRETCH_RANGES[stretch_name]
        stretches = [1.0] * word_count
        stretch_index = py_rng.randrange(word_count)
        stretches[stretch_index] = py_rng.uniform(stretch_min, stretch_max)
        component_rows = [py_rng.choice(grouped[speaker][word]) for word in selected_words]
        audio = compose(component_rows, gaps, stretches, np_rng)
        duration = len(audio) / 16_000
        if duration > maximum_seconds:
            continue

        identifier_hash = hashlib.sha256(
            f"{split}:{speaker}:{phrase}:{index}:{rng_seed}".encode()
        ).hexdigest()[:16]
        identifier = f"command-{split}-{speaker}-{identifier_hash}"
        filename = f"{identifier}.wav"
        sf.write(destination / filename, audio, 16_000, subtype="PCM_16")
        row = {
            "id": identifier,
            "speaker": speaker,
            "condition": "dysarthric_composite",
            "corpus": "TORGO-derived literal command composition",
            "duration": round(duration, 4),
            "text": phrase,
            "audio_filepath": str(cloud_audio / filename),
            "component_ids": [item["id"] for item in component_rows],
            "gap_seconds": [round(value, 4) for value in gaps],
            "stretch_factors": [round(value, 4) for value in stretches],
            "word_count": word_count,
            "gap_class": gap_name,
            "stretch_class": stretch_name,
            "slice": f"command_{word_count}word_gap_{gap_name}_stretch_{stretch_name}",
            "synthetic_composition": True,
        }
        rows.append(row)
        phrases.add(phrase)
    if len(rows) != count:
        raise RuntimeError(f"Built only {len(rows)}/{count} unique {split} commands")
    return rows, phrases


def audit(
    train: list[dict], dev: list[dict], test: list[dict], source_speakers: dict[str, set[str]]
) -> dict:
    phrase_sets = [{row["text"] for row in rows} for rows in (train, dev, test)]
    if any(phrase_sets[a] & phrase_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Command phrases overlap between train/dev/test")
    speaker_sets = [{row["speaker"] for row in rows} for rows in (train, dev, test)]
    if speaker_sets[0] & (speaker_sets[1] | speaker_sets[2]) or speaker_sets[1] & speaker_sets[2]:
        raise AssertionError("Command speakers overlap between train/dev/test")
    if speaker_sets != [source_speakers["train"], source_speakers["dev"], source_speakers["test"]]:
        raise AssertionError("Command speakers differ from their source splits")
    return {
        name: {
            "utterances": len(rows),
            "hours": round(sum(float(row["duration"]) for row in rows) / 3600, 4),
            "speakers": sorted({row["speaker"] for row in rows}),
            "two_word": sum(int(row["word_count"]) == 2 for row in rows),
            "three_word": sum(int(row["word_count"]) == 3 for row in rows),
            "gap_classes": dict(Counter(row["gap_class"] for row in rows)),
            "stretch_classes": dict(Counter(row["stretch_class"] for row in rows)),
        }
        for name, rows in (("train", train), ("dev", dev), ("test", test))
    }


def stage(config_path: Path, pilot_data: Path, cloud_root: Path) -> dict:
    config = read_json(config_path)
    manifests = pilot_data / "manifests"
    source = {
        name: localize_staged_audio(
            read_jsonl(manifests / f"torgo_{name}.jsonl"), pilot_data, cloud_root
        )
        for name in ("train", "dev", "test")
    }
    train_grouped = eligible_words(source["train"])
    train_vocabulary = {
        word for words in train_grouped.values() for word in words
    }
    grouped = {
        "train": train_grouped,
        "dev": eligible_words(source["dev"], train_vocabulary),
        "test": eligible_words(source["test"], train_vocabulary),
    }
    expected = {
        "dev": {str(config["development_speaker"])},
        "test": {str(config["outer_test_speaker"])},
    }
    if set(grouped["dev"]) != expected["dev"] or set(grouped["test"]) != expected["test"]:
        raise ValueError("Held-out command speakers do not match the frozen configuration")

    command_root = pilot_data / "audio" / "command_v3"
    command_root.mkdir(parents=True, exist_ok=False)
    cloud_audio_root = cloud_root / "audio" / "command_v3"
    counts = {
        "train": int(config["command_train_pool_size"]),
        "dev": int(config["command_dev_size"]),
        "test": int(config["command_test_size"]),
    }
    built = {}
    phrases: set[str] = set()
    for offset, name in enumerate(("train", "dev", "test")):
        built[name], phrases = build_split(
            name,
            grouped[name],
            counts[name],
            command_root / name,
            cloud_audio_root / name,
            int(config["seed"]) + offset * 1009,
            phrases,
            float(config["maximum_audio_seconds"]),
        )
        write_jsonl(manifests / f"command_{name}.jsonl", built[name])

    source_speakers = {
        name: {row["speaker"] for row in source[name] if row.get("condition") == "dysarthric"}
        for name in ("train", "dev", "test")
    }
    split_report = audit(built["train"], built["dev"], built["test"], source_speakers)

    literal_rows = read_jsonl(manifests / "literal_diagnostic.jsonl")
    diagnostic = list(literal_rows) + built["test"]
    diagnostic_ids = [row["id"] for row in diagnostic]
    if len(diagnostic_ids) != len(set(diagnostic_ids)):
        raise ValueError("Command diagnostic contains duplicate IDs")
    write_jsonl(manifests / "command_diagnostic.jsonl", diagnostic)

    report = {
        "run_id": config["run_id"],
        "method": "same-speaker isolated-word concatenation with controlled gaps and pitch-preserving stretch",
        "clinical_data_claim": False,
        "literal_transcripts_only": True,
        "speaker_disjoint": True,
        "phrase_disjoint": True,
        "train_vocabulary_size": len(train_vocabulary),
        "eligible_word_recordings": {
            name: sum(len(values) for words in grouped[name].values() for values in words.values())
            for name in grouped
        },
        "splits": split_report,
        "diagnostic_utterances": len(diagnostic),
    }
    (pilot_data / "command_v3_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-data", type=Path, required=True)
    parser.add_argument(
        "--cloud-root", type=Path, default=Path("/workspace/echora/pilot_data")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            stage(arguments.config, arguments.pilot_data, arguments.cloud_root),
            indent=2,
        )
    )
