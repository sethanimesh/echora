#!/usr/bin/env python3
"""Recompute saved foundation metrics and print the archived adapter comparison.

Standard library only. Does not run models, contact providers, or read private data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "benchmarks"))
from compare_foundations import read_rows, summarize  # noqa: E402


def main() -> None:
    directory = ROOT / "research/benchmarks/results/foundation_gate"
    archived = json.loads((directory / "comparison.json").read_text())
    rows = {name: read_rows(directory / f"{name}.tsv") for name in archived["models"]}
    values = list(rows.values())
    if any(item.keys() != values[0].keys() for item in values[1:]):
        raise ValueError("Foundation predictions do not cover identical utterance IDs")
    for identifier, baseline in values[0].items():
        if any((item[identifier]["reference"], item[identifier]["speaker"]) !=
               (baseline["reference"], baseline["speaker"]) for item in values[1:]):
            raise ValueError(f"Mismatched reference or speaker for {identifier}")

    print("# Saved foundation predictions — recomputed")
    print("\n| Model | Utterances | Speaker-macro WER | Deletion rate | Empty rate |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, predictions in rows.items():
        result = summarize(predictions)
        for key in ("utterances", "micro_wer", "micro_cer", "macro_speaker_wer",
                    "worst_speaker_wer", "deletion_rate", "empty_rate"):
            if not math.isclose(result[key], archived["models"][name][key], abs_tol=1e-12):
                raise ValueError(f"Saved comparison mismatch: {name}.{key}")
        print(f"| {name} | {result['utterances']} | {result['macro_speaker_wer']:.2%} | "
              f"{result['deletion_rate']:.2%} | {result['empty_rate']:.2%} |")

    decision_path = ROOT / "models/echora-qwen3-asr-command-v3/evaluation/next_decision.json"
    decision = json.loads(decision_path.read_text())
    print("\n# Command-v3 — archived summary, not a new inference run")
    print("\n| Test | v1 WER | v3 WER |")
    print("| --- | ---: | ---: |")
    for name in ("torgo_test", "command_test", "normal_test", "personal"):
        print(f"| {name} | {decision['v1_test'][name]:.2%} | {decision['v3_test'][name]:.2%} |")
    beams = decision["v3_command_top5"]
    print(f"\nSeparate beam diagnostic: rank-one WER {beams['top1_wer']:.2%}; "
          f"exact-in-top-five coverage {beams['exact_in_topk_rate']:.2%}.")
    print("Oracle coverage is not automatic-selection accuracy. See docs/evaluation.md.")


if __name__ == "__main__":
    main()
