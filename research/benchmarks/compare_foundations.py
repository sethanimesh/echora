"""Compare two literal-ASR result files on the exact same utterances."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from metrics import aggregate, normalize, score


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate utterance IDs")
    return result


def summarize(rows: dict[str, dict[str, str]]) -> dict:
    by_speaker: dict[str, list] = defaultdict(list)
    all_scores = []
    empty = 0
    rtfs: list[float] = []
    for row in rows.values():
        item = score(row["reference"], row["transcript"])
        all_scores.append(item)
        by_speaker[row["speaker"]].append(item)
        empty += not normalize(row["transcript"])
        if row.get("rtf", "").strip():
            rtfs.append(float(row["rtf"]))

    overall = aggregate(all_scores)
    speaker_wer = {
        speaker: aggregate(items).wer for speaker, items in sorted(by_speaker.items())
    }
    return {
        "utterances": len(rows),
        "micro_wer": overall.wer,
        "micro_cer": overall.cer,
        "macro_speaker_wer": sum(speaker_wer.values()) / len(speaker_wer),
        "worst_speaker_wer": max(speaker_wer.values()),
        "deletion_rate": overall.word_deletions / overall.reference_words,
        "empty_rate": empty / len(rows),
        "median_rtf": median(rtfs) if rtfs else None,
        "speaker_wer": speaker_wer,
    }


def decide(
    left: dict,
    right: dict,
    names: tuple[str, str],
    relative_margin: float,
    guardrail_margin: float,
) -> dict:
    if left["macro_speaker_wer"] <= right["macro_speaker_wer"]:
        winner, loser = left, right
        winner_name, loser_name = names
    else:
        winner, loser = right, left
        winner_name, loser_name = names[1], names[0]

    relative_gain = (
        (loser["macro_speaker_wer"] - winner["macro_speaker_wer"])
        / loser["macro_speaker_wer"]
        if loser["macro_speaker_wer"]
        else 0.0
    )
    guardrails = {
        "deletion_rate": winner["deletion_rate"]
        <= loser["deletion_rate"] + guardrail_margin,
        "empty_rate": winner["empty_rate"] <= loser["empty_rate"] + guardrail_margin,
    }
    passes = relative_gain >= relative_margin and all(guardrails.values())
    return {
        "status": "provisional_winner" if passes else "full_evaluation_required",
        "winner": winner_name if passes else None,
        "lower_macro_wer_model": winner_name,
        "other_model": loser_name,
        "relative_macro_wer_gain": relative_gain,
        "required_relative_margin": relative_margin,
        "guardrail_margin_absolute": guardrail_margin,
        "guardrails": guardrails,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-name", default="parakeet_tdt_1_1b")
    parser.add_argument("--right-name", default="qwen3_asr_1_7b")
    parser.add_argument("--relative-margin", type=float, default=0.05)
    parser.add_argument("--guardrail-margin", type=float, default=0.02)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    left_rows = read_rows(args.left)
    right_rows = read_rows(args.right)
    if left_rows.keys() != right_rows.keys():
        left_only = sorted(left_rows.keys() - right_rows.keys())[:5]
        right_only = sorted(right_rows.keys() - left_rows.keys())[:5]
        raise ValueError(
            "result files must contain identical IDs; "
            f"left-only={left_only}, right-only={right_only}"
        )

    left = summarize(left_rows)
    right = summarize(right_rows)
    report = {
        "scope": "literal ASR before any candidate ranker",
        "models": {args.left_name: left, args.right_name: right},
        "decision": decide(
            left,
            right,
            (args.left_name, args.right_name),
            args.relative_margin,
            args.guardrail_margin,
        ),
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
