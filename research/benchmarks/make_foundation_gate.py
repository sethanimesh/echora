"""Build a deterministic speaker- and duration-balanced TORGO screening set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def stable_key(record: dict, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{record['id']}".encode()).hexdigest()


def select_records(
    records: list[dict],
    per_speaker: int,
    duration_bins: int,
    seed: str,
) -> list[dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_speaker[record["speaker"]].append(record)

    selected: list[dict] = []
    for speaker, items in sorted(by_speaker.items()):
        if len(items) < per_speaker:
            raise ValueError(
                f"speaker {speaker} has {len(items)} rows, fewer than requested {per_speaker}"
            )
        duration_order = sorted(items, key=lambda row: (float(row["duration"]), row["id"]))
        bins: list[list[dict]] = [[] for _ in range(duration_bins)]
        for index, record in enumerate(duration_order):
            bin_index = min(duration_bins - 1, index * duration_bins // len(duration_order))
            bins[bin_index].append(record)

        quota, remainder = divmod(per_speaker, duration_bins)
        for bin_index, duration_bin in enumerate(bins):
            take = quota + (1 if bin_index < remainder else 0)
            ranked = sorted(duration_bin, key=lambda row: stable_key(row, seed))
            if len(ranked) < take:
                raise ValueError(f"duration bin {bin_index} for {speaker} is too small")
            selected.extend(ranked[:take])

    return sorted(selected, key=lambda row: (row["speaker"], float(row["duration"]), row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-speaker", type=int, default=50)
    parser.add_argument("--duration-bins", type=int, default=5)
    parser.add_argument("--seed", default="echora-foundation-gate-v1")
    args = parser.parse_args()
    if args.per_speaker <= 0 or args.duration_bins <= 0:
        parser.error("--per-speaker and --duration-bins must be positive")
    if args.duration_bins > args.per_speaker:
        parser.error("--duration-bins cannot exceed --per-speaker")

    records = [
        json.loads(line)
        for line in args.source.read_text().splitlines()
        if line.strip()
    ]
    selected = select_records(records, args.per_speaker, args.duration_bins, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in selected))
    print(
        f"wrote {len(selected)} rows across "
        f"{len({row['speaker'] for row in selected})} speakers to {args.output}"
    )


if __name__ == "__main__":
    main()
