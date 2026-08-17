"""Build small, speaker-disjoint Common Voice retention manifests.

The resulting data is not part of the dysarthric headline score.  It is used
to keep ordinary English stable during partial Qwen acoustic adaptation and to
measure catastrophic forgetting on speakers that never enter training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def duration_seconds(row: dict[str, str], audio: Path) -> float:
    for key in ("duration[ms]", "duration_ms", "duration_millis"):
        value = (row.get(key) or "").strip()
        if value:
            return float(value) / 1000.0
    value = (row.get("duration") or "").strip()
    if value:
        parsed = float(value)
        # Some Common Voice exports store duration in milliseconds under the
        # generic name; values above ten minutes cannot be clip seconds.
        return parsed / 1000.0 if parsed > 600 else parsed
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode or not process.stdout.strip():
        raise RuntimeError(f"Could not determine duration for {audio}")
    return float(process.stdout.strip())


def discover_tables(source: Path) -> list[Path]:
    validated = sorted(source.rglob("validated.tsv"))
    if validated:
        return validated
    tables = []
    for name in ("train.tsv", "dev.tsv", "test.tsv"):
        tables.extend(sorted(source.rglob(name)))
    if not tables:
        raise FileNotFoundError(
            f"No validated.tsv, train.tsv, dev.tsv or test.tsv found under {source}"
        )
    return tables


def collect(
    source: Path,
    minimum_seconds: float,
    maximum_seconds: float,
    seed: int,
    target_seconds: float | None = None,
) -> list[dict]:
    # Read metadata first, then duration-probe only a deterministic prefix. A
    # Common Voice subset can contain hundreds of thousands of MP3s and its TSV
    # often has no duration column; probing an entire 390-hour subset to select
    # four hours would waste substantial local time.
    candidates: dict[Path, tuple[dict[str, str], str, str, str]] = {}
    for table in discover_tables(source):
        with table.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"path", "sentence", "client_id"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{table} is missing required columns {sorted(required)}")
            for row in reader:
                relative = (row.get("path") or "").strip()
                text = " ".join((row.get("sentence") or "").split())
                client = (row.get("client_id") or "").strip()
                locale = (row.get("locale") or "en").strip().lower()
                if not relative or not text or not client or (locale and not locale.startswith("en")):
                    continue
                candidate = table.parent / relative
                if not candidate.is_file():
                    candidate = table.parent / "clips" / relative
                if not candidate.is_file():
                    raise FileNotFoundError(f"Common Voice clip is missing: {relative} from {table}")
                candidate = candidate.resolve()
                if candidate in candidates:
                    continue
                speaker = f"cv-{stable_hash(client, seed)[:16]}"
                identifier = f"common-voice-{stable_hash(str(candidate), seed)[:20]}"
                candidates[candidate] = (row, speaker, identifier, text)

    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            stable_hash(item[1][1], seed),
            stable_hash(item[1][2], seed),
        ),
    )
    rows = []
    selected_seconds = 0.0
    selected_speakers: set[str] = set()
    # The buffer absorbs whole-speaker packing overhead in choose_splits.
    buffered_target = None if target_seconds is None else target_seconds * 1.5
    for candidate, (metadata, speaker, identifier, text) in ordered:
        seconds = duration_seconds(metadata, candidate)
        if not minimum_seconds <= seconds <= maximum_seconds:
            continue
        rows.append(
            {
                "id": identifier,
                "speaker": speaker,
                "condition": "normal",
                "corpus": "Common Voice",
                "duration": round(seconds, 4),
                "text": text,
                "audio_filepath": str(candidate),
            }
        )
        selected_seconds += seconds
        selected_speakers.add(speaker)
        if buffered_target is not None and selected_seconds >= buffered_target and len(selected_speakers) >= 12:
            break
    if not rows:
        raise ValueError("No usable English Common Voice recordings were found")
    return rows


def choose_splits(
    rows: list[dict], train_seconds: float, dev_seconds: float, test_seconds: float, seed: int
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["speaker"]].append(row)
    speakers = sorted(grouped, key=lambda speaker: stable_hash(speaker, seed))
    for values in grouped.values():
        values.sort(key=lambda row: stable_hash(row["id"], seed))

    targets = {"test": test_seconds, "dev": dev_seconds, "train": train_seconds}
    selected = {name: [] for name in targets}
    totals = {name: 0.0 for name in targets}
    remaining = iter(speakers)
    for split in ("test", "dev", "train"):
        while totals[split] < targets[split]:
            try:
                speaker = next(remaining)
            except StopIteration as error:
                readable = {name: round(value / 3600, 3) for name, value in totals.items()}
                raise ValueError(
                    f"Not enough speaker-disjoint Common Voice audio; selected hours={readable}"
                ) from error
            selected[split].extend(grouped[speaker])
            totals[split] += sum(float(row["duration"]) for row in grouped[speaker])
    for values in selected.values():
        values.sort(key=lambda row: row["id"])
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(
    source: Path,
    output: Path,
    train_hours: float,
    dev_minutes: float,
    test_minutes: float,
    minimum_seconds: float,
    maximum_seconds: float,
    seed: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    required_seconds = train_hours * 3600 + (dev_minutes + test_minutes) * 60
    rows = collect(
        source,
        minimum_seconds,
        maximum_seconds,
        seed,
        target_seconds=required_seconds,
    )
    splits = choose_splits(
        rows,
        train_hours * 3600,
        dev_minutes * 60,
        test_minutes * 60,
        seed,
    )
    output.mkdir(parents=True)
    for name, values in splits.items():
        write_jsonl(output / f"{name}.jsonl", values)
    summary = {
        "source": str(source.resolve()),
        "seed": seed,
        "duration_filter_seconds": [minimum_seconds, maximum_seconds],
        "splits": {
            name: {
                "utterances": len(values),
                "speakers": len({row["speaker"] for row in values}),
                "hours": round(sum(float(row["duration"]) for row in values) / 3600, 4),
            }
            for name, values in splits.items()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/common_voice"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/common_voice/pilot-v1"))
    parser.add_argument("--train-hours", type=float, default=3.0)
    parser.add_argument("--dev-minutes", type=float, default=20.0)
    parser.add_argument("--test-minutes", type=float, default=20.0)
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--maximum-seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    if min(args.train_hours, args.dev_minutes, args.test_minutes) <= 0:
        parser.error("all requested split durations must be positive")
    if not 0 < args.minimum_seconds < args.maximum_seconds:
        parser.error("duration limits are invalid")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    report = prepare(
        arguments.source,
        arguments.output,
        arguments.train_hours,
        arguments.dev_minutes,
        arguments.test_minutes,
        arguments.minimum_seconds,
        arguments.maximum_seconds,
        arguments.seed,
    )
    print(json.dumps(report, indent=2))
