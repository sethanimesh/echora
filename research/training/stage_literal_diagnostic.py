"""Add protected literal-ASR diagnostic slices to staged Qwen pilot data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def literal_slice(text: str) -> str:
    count = len(text.split())
    if count == 1:
        return "torgo_1word"
    if count <= 3:
        return "torgo_2_3word"
    return "torgo_4plus"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def add_audio_rows(
    source_dir: Path,
    destination_dir: Path,
    cloud_dir: Path,
    slice_name: str,
) -> list[dict]:
    rows = []
    for source in sorted(source_dir.glob("*.wav")):
        sidecar = source.with_suffix(".txt")
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing literal reference: {sidecar}")
        destination = destination_dir / f"{slice_name}-{source.name}"
        shutil.copy2(source, destination)
        info = sf.info(source)
        rows.append(
            {
                "id": f"literal-{slice_name}-{source.stem}",
                "speaker": "external-personal",
                "condition": "external_smoke",
                "corpus": "Echora literal diagnostic",
                "duration": round(info.frames / info.samplerate, 4),
                "text": sidecar.read_text(encoding="utf-8").strip(),
                "audio_filepath": str(cloud_dir / destination.name),
                "slice": slice_name,
            }
        )
    if not rows:
        raise ValueError(f"No WAV files found in {source_dir}")
    return rows


def stage(pilot_data: Path, pause_root: Path, cloud_root: Path) -> dict:
    manifest_dir = pilot_data / "manifests"
    test_rows = read_jsonl(manifest_dir / "torgo_test.jsonl")
    personal_rows = read_jsonl(manifest_dir / "personal.jsonl")
    diagnostic_audio = pilot_data / "audio" / "literal_diagnostic"
    diagnostic_audio.mkdir(parents=True, exist_ok=False)
    cloud_audio = cloud_root / "audio" / "literal_diagnostic"

    rows = []
    for row in test_rows:
        item = dict(row)
        item["slice"] = literal_slice(item["text"])
        rows.append(item)
    for row in personal_rows:
        item = dict(row)
        item["slice"] = "personal_original"
        rows.append(item)
    for delay in ("0500ms", "1000ms", "2000ms"):
        rows.extend(
            add_audio_rows(
                pause_root / delay,
                diagnostic_audio,
                cloud_audio,
                f"pause_{delay}",
            )
        )

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Literal diagnostic IDs are not unique")
    path = manifest_dir / "literal_diagnostic.jsonl"
    write_jsonl(path, rows)
    counts = {}
    for row in rows:
        counts[row["slice"]] = counts.get(row["slice"], 0) + 1
    report = {
        "manifest": str(path),
        "utterances": len(rows),
        "slices": dict(sorted(counts.items())),
        "training_use": False,
        "semantic_repair": False,
    }
    (pilot_data / "literal_diagnostic_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-data", type=Path, required=True)
    parser.add_argument("--pause-root", type=Path, required=True)
    parser.add_argument(
        "--cloud-root", type=Path, default=Path("/workspace/echora/pilot_data")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            stage(arguments.pilot_data, arguments.pause_root, arguments.cloud_root),
            indent=2,
        )
    )
