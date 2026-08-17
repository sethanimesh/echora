"""Audit and stage the complete Qwen dysarthric-ASR pilot data payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


REQUIRED = {"id", "speaker", "condition", "duration", "text", "audio_filepath"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED - set(row)
            if missing:
                raise ValueError(f"{path}:{number} is missing {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def eligible(rows: list[dict], maximum_seconds: float) -> tuple[list[dict], list[dict]]:
    kept, excluded = [], []
    for row in rows:
        seconds = float(row["duration"])
        if 0 < seconds <= maximum_seconds:
            kept.append(row)
        else:
            excluded.append(row)
    return kept, excluded


def audit_splits(
    torgo_train: list[dict],
    torgo_dev: list[dict],
    torgo_test: list[dict],
    normal_train: list[dict],
    normal_dev: list[dict],
    normal_test: list[dict],
    config: dict,
) -> dict:
    expected_dev = config["development_speaker"]
    expected_test = config["outer_test_speaker"]
    if {row["speaker"] for row in torgo_dev} != {expected_dev}:
        raise ValueError(f"TORGO development must contain only {expected_dev}")
    if {row["speaker"] for row in torgo_test} != {expected_test}:
        raise ValueError(f"TORGO test must contain only {expected_test}")
    held_out = {expected_dev, expected_test}
    leaked = {row["speaker"] for row in torgo_train} & held_out
    if leaked:
        raise ValueError(f"Held-out TORGO speakers leaked into training: {sorted(leaked)}")
    if {row["condition"] for row in torgo_dev + torgo_test} != {"dysarthric"}:
        raise ValueError("TORGO development and test must be dysarthric-only")
    if not {"dysarthric", "control"}.issubset(
        {row["condition"] for row in torgo_train}
    ):
        raise ValueError("TORGO training must contain dysarthric and control speech")

    test_text_ids = {row.get("text_id") for row in torgo_test}
    if None in test_text_ids:
        raise ValueError("TORGO test rows require text_id for leakage auditing")
    overlaps = {}
    for split_name, rows in (("train", torgo_train), ("dev", torgo_dev)):
        overlap = test_text_ids & {row.get("text_id") for row in rows}
        overlaps[split_name] = len(overlap)
        if overlap and not config.get("allow_outer_test_prompt_overlap", False):
            raise ValueError(
                f"Strict TORGO {split_name} still contains {len(overlap)} outer-test prompts"
            )

    normal_sets = [normal_train, normal_dev, normal_test]
    for name, rows in zip(("train", "dev", "test"), normal_sets):
        if {row["condition"] for row in rows} != {"normal"}:
            raise ValueError(f"Common Voice {name} rows must have condition=normal")
    normal_speakers = [{row["speaker"] for row in rows} for rows in normal_sets]
    if any(normal_speakers[a] & normal_speakers[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Common Voice speakers overlap between train/dev/test")

    all_groups = [torgo_train, torgo_dev, torgo_test, *normal_sets]
    all_ids = [row["id"] for rows in all_groups for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Utterance IDs overlap between manifests")
    return {
        "outer_test_prompt_overlap": overlaps,
        "prompt_overlap_allowed_and_disclosed": bool(
            config.get("allow_outer_test_prompt_overlap", False)
        ),
        "torgo_train_conditions": dict(Counter(row["condition"] for row in torgo_train)),
        "torgo_train_speakers": sorted({row["speaker"] for row in torgo_train}),
        "torgo_dev_speakers": sorted({row["speaker"] for row in torgo_dev}),
        "torgo_test_speakers": sorted({row["speaker"] for row in torgo_test}),
        "normal_speakers": {
            name: len(speakers)
            for name, speakers in zip(("train", "dev", "test"), normal_speakers)
        },
    }


def copy_split(rows: list[dict], output: Path, cloud_root: Path, split: str) -> list[dict]:
    audio_dir = output / "audio" / split
    audio_dir.mkdir(parents=True)
    staged = []
    for row in rows:
        source = Path(row["audio_filepath"])
        if not source.is_file():
            raise FileNotFoundError(f"Missing audio for {row['id']}: {source}")
        suffix = source.suffix.lower() or ".wav"
        name = f"{row['id']}{suffix}"
        destination = audio_dir / name
        shutil.copy2(source, destination)
        if split == "personal":
            destination.with_suffix(".txt").write_text(
                str(row["text"]).strip() + "\n", encoding="utf-8"
            )
        item = dict(row)
        item["audio_filepath"] = str(cloud_root / "audio" / split / name)
        item["split"] = split
        staged.append(item)
    return staged


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def personal_rows(clips: Path, maximum_seconds: float) -> list[dict]:
    import soundfile as sf

    rows = []
    for audio in sorted(clips.glob("*.wav")):
        sidecar = audio.with_suffix(".txt")
        if not sidecar.is_file():
            raise FileNotFoundError(f"Personal smoke clip lacks a transcript: {sidecar}")
        info = sf.info(audio)
        seconds = info.frames / info.samplerate
        if not 0 < seconds <= maximum_seconds:
            raise ValueError(f"Personal clip duration is outside the configured limit: {audio}")
        rows.append(
            {
                "id": f"personal-{audio.stem}",
                "speaker": "external-personal",
                "condition": "external_smoke",
                "corpus": "Echora personal clips",
                "duration": round(seconds, 4),
                "text": sidecar.read_text(encoding="utf-8").strip(),
                "audio_filepath": str(audio.resolve()),
            }
        )
    if not rows:
        raise ValueError(f"No personal WAV clips found in {clips}")
    return rows


def stage(
    config_path: Path,
    torgo_dir: Path,
    normal_dir: Path,
    personal_dir: Path,
    output: Path,
    cloud_root: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing staging directory: {output}")
    config = read_json(config_path)
    maximum = float(config["maximum_audio_seconds"])
    sources = {
        "torgo_train": torgo_dir / "train.jsonl",
        "torgo_dev": torgo_dir / "dev.jsonl",
        "torgo_test": torgo_dir / "test.jsonl",
        "normal_train": normal_dir / "train.jsonl",
        "normal_dev": normal_dir / "dev.jsonl",
        "normal_test": normal_dir / "test.jsonl",
    }
    loaded = {name: read_jsonl(path) for name, path in sources.items()}
    filtered, exclusions = {}, {}
    for name, rows in loaded.items():
        filtered[name], exclusions[name] = eligible(rows, maximum)
        if not filtered[name]:
            raise ValueError(f"No rows remain in {name} after the duration filter")
    audit = audit_splits(
        filtered["torgo_train"],
        filtered["torgo_dev"],
        filtered["torgo_test"],
        filtered["normal_train"],
        filtered["normal_dev"],
        filtered["normal_test"],
        config,
    )
    personal = personal_rows(personal_dir, maximum)
    output.mkdir(parents=True)
    staged = {}
    for name, rows in {**filtered, "personal": personal}.items():
        staged[name] = copy_split(rows, output, cloud_root, name)
        write_jsonl(output / "manifests" / f"{name}.jsonl", staged[name])

    report = {
        "run_id": config["run_id"],
        "config_sha256": sha256(config_path),
        "maximum_audio_seconds": maximum,
        "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()},
        "splits": {
            name: {
                "utterances": len(rows),
                "hours": round(sum(float(row["duration"]) for row in rows) / 3600, 4),
                "excluded_by_duration": len(exclusions.get(name, [])),
            }
            for name, rows in staged.items()
        },
        "audit": audit,
    }
    (output / "stage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--torgo-dir", type=Path, required=True)
    parser.add_argument("--normal-dir", type=Path, required=True)
    parser.add_argument("--personal-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-root", type=Path, default=Path("/workspace/echora/pilot_data"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = stage(
        arguments.config,
        arguments.torgo_dir,
        arguments.normal_dir,
        arguments.personal_dir,
        arguments.output,
        arguments.cloud_root,
    )
    print(json.dumps(result, indent=2))
