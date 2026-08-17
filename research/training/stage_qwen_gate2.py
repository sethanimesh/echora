"""Build the fixed, small data payload for the Qwen Gate 2 GPU probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_rows(rows: list[dict], config: dict) -> list[dict]:
    test_speaker = config["outer_test_speaker"]
    dev_speaker = config["development_speaker"]
    condition = config["training_condition"]
    maximum_seconds = float(config["maximum_probe_audio_seconds"])
    eligible = []
    for row in rows:
        if row.get("speaker") in {test_speaker, dev_speaker}:
            continue
        if row.get("condition") != condition:
            continue
        if not 0 < float(row.get("duration", 0)) <= maximum_seconds:
            continue
        eligible.append(row)
    if not eligible:
        raise ValueError("No eligible dysarthric training rows remain")
    return eligible


def select_probe_rows(rows: list[dict], config: dict) -> tuple[dict, list[dict]]:
    eligible = eligible_rows(rows, config)
    longest = max(eligible, key=lambda row: (float(row["duration"]), row["id"]))

    grouped: dict[str, list[dict]] = defaultdict(list)
    target = float(config["tiny_subset_target_seconds"])
    for row in eligible:
        if row["id"] != longest["id"]:
            grouped[row["speaker"]].append(row)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: (abs(float(row["duration"]) - target), row["id"]))

    expected_speakers = sorted(grouped)
    wanted = int(config["tiny_subset_size"])
    if wanted < len(expected_speakers):
        raise ValueError(
            "tiny_subset_size must be at least the number of eligible training speakers"
        )

    selected = []
    depth = 0
    while len(selected) < wanted:
        added = False
        for speaker in expected_speakers:
            candidates = grouped[speaker]
            if depth < len(candidates):
                selected.append(candidates[depth])
                added = True
                if len(selected) == wanted:
                    break
        if not added:
            raise ValueError(f"Only found {len(selected)} rows for a {wanted}-row probe")
        depth += 1

    if {row["speaker"] for row in selected} != set(expected_speakers):
        raise AssertionError("Tiny probe is not balanced across training speakers")
    return longest, selected


def stage(
    train_manifest: Path,
    config_path: Path,
    output_dir: Path,
    cloud_root: Path,
) -> None:
    config = read_json(config_path)
    rows = read_jsonl(train_manifest)
    longest, tiny = select_probe_rows(rows, config)

    output_dir.mkdir(parents=True, exist_ok=False)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir()
    staged_rows = []
    copied: dict[str, str] = {}
    for purpose, row in [("longest", longest), *(("tiny", item) for item in tiny)]:
        source = Path(row["audio_filepath"])
        if not source.is_file():
            raise FileNotFoundError(f"Missing TORGO audio: {source}")
        destination_name = f"{row['id']}{source.suffix.lower()}"
        if row["id"] not in copied:
            shutil.copy2(source, audio_dir / destination_name)
            copied[row["id"]] = destination_name
        staged = dict(row)
        staged["audio_filepath"] = str(cloud_root / "audio" / destination_name)
        staged["purpose"] = purpose
        staged_rows.append(staged)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in staged_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    selection = {
        "run_id": config["run_id"],
        "source_manifest": str(train_manifest),
        "source_manifest_sha256": sha256(train_manifest),
        "config_sha256": sha256(config_path),
        "excluded_speakers": [
            config["development_speaker"],
            config["outer_test_speaker"],
        ],
        "maximum_probe_audio_seconds": config["maximum_probe_audio_seconds"],
        "longest_eligible": {
            "id": longest["id"],
            "speaker": longest["speaker"],
            "duration": longest["duration"],
            "text": longest["text"],
        },
        "tiny_subset": [
            {
                "id": row["id"],
                "speaker": row["speaker"],
                "duration": row["duration"],
                "text": row["text"],
            }
            for row in tiny
        ],
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cloud-root",
        type=Path,
        default=Path("/workspace/echora/probe"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    stage(
        arguments.train_manifest,
        arguments.config,
        arguments.output,
        arguments.cloud_root,
    )
