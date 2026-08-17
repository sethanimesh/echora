"""Create the compact evidence package needed for Echora's next ASR decision."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "research" / "benchmarks"))
from metrics import normalize  # noqa: E402


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def target_row(rows: list[dict], reference: str) -> dict:
    matches = [row for row in rows if normalize(row["reference"]) == normalize(reference)]
    personal = [row for row in matches if row.get("slice") == "personal_original"]
    chosen = personal or matches
    if len(chosen) != 1:
        raise ValueError(f"Expected one personal '{reference}' row, found {len(chosen)}")
    row = dict(chosen[0])
    row["qwen_hypotheses"] = json.loads(row["qwen_hypotheses"])
    return row


def metric_delta(after: dict, before: dict, key: str) -> float:
    return float(after[key]) - float(before[key])


def build(v1_summary_path: Path, v1_tsv: Path, v2_summary_path: Path, v2_tsv: Path, result_path: Path):
    v1_summary = read_json(v1_summary_path)
    v2_summary = read_json(v2_summary_path)
    result = read_json(result_path)
    v1_water = target_row(read_tsv(v1_tsv), "I water")
    v2_water = target_row(read_tsv(v2_tsv), "I water")
    v1_texts = [item["text"] for item in v1_water["qwen_hypotheses"]]
    v2_texts = [item["text"] for item in v2_water["qwen_hypotheses"]]
    v1_exact = any(normalize(text) == "i water" for text in v1_texts)
    v2_exact = any(normalize(text) == "i water" for text in v2_texts)
    ctc_exact = normalize(v2_water.get("ctc_transcript", "")) == "i water"

    one_before = v1_summary["slices"]["torgo_1word"]
    one_after = v2_summary["slices"]["torgo_1word"]
    short_before = v1_summary["slices"]["torgo_2_3word"]
    short_after = v2_summary["slices"]["torgo_2_3word"]
    final = result["final_evaluation"]
    baseline = result["baseline_v1"]

    ctc_short_generalizes = (
        one_after.get("ctc_wer") is not None
        and one_after["ctc_wer"] <= one_after["qwen_top1_wer"]
    )
    if ctc_exact and ctc_short_generalizes:
        branch = "keep_dual_literal_outputs"
        reason = (
            "CTC recovered the literal telegraphic phrase exactly and did not trail Qwen "
            "on the 216-utterance one-word dysarthric slice."
        )
    elif ctc_exact:
        branch = "personal_ctc_success_not_yet_universal"
        reason = (
            "CTC recovered the personal phrase, but its one-word dysarthric result does not "
            "yet support universal deployment."
        )
    elif v2_exact:
        branch = "qwen_search_contains_literal_but_top1_ranking_fails"
        reason = "The literal phrase is acoustically reachable in Qwen top-k but is not rank one."
    elif v1_exact:
        branch = "joint_tuning_hurt_literal_search"
        reason = "v1 contained the literal phrase but v2 removed it; do not deploy v2 Qwen weights."
    else:
        branch = "need_real_telegraphic_dysarthric_training_data"
        reason = "Neither Qwen top-k nor CTC recovered the literal phrase; this is not fixed by grammar repair."

    decision = {
        "decision_branch": branch,
        "reason": reason,
        "personal_i_water": {
            "reference": "I water",
            "v1_qwen_top1": v1_water["qwen_top1"],
            "v1_qwen_top5": v1_texts,
            "v1_literal_present_top5": v1_exact,
            "v2_qwen_top1": v2_water["qwen_top1"],
            "v2_qwen_top5": v2_texts,
            "v2_literal_present_top5": v2_exact,
            "v2_ctc": v2_water.get("ctc_transcript", ""),
            "v2_ctc_exact": ctc_exact,
            "ctc_short_generalizes": ctc_short_generalizes,
        },
        "short_literal_diagnostics": {
            "torgo_1word": {
                "utterances": one_after["utterances"],
                "v1_qwen_top1_wer": one_before["qwen_top1_wer"],
                "v2_qwen_top1_wer": one_after["qwen_top1_wer"],
                "v2_ctc_wer": one_after.get("ctc_wer"),
                "v1_qwen_top5_oracle_wer": one_before["qwen_topk_oracle_wer"],
                "v2_qwen_top5_oracle_wer": one_after["qwen_topk_oracle_wer"],
            },
            "torgo_2_3word": {
                "utterances": short_after["utterances"],
                "v1_qwen_top1_wer": short_before["qwen_top1_wer"],
                "v2_qwen_top1_wer": short_after["qwen_top1_wer"],
                "v2_ctc_wer": short_after.get("ctc_wer"),
                "v1_qwen_top5_oracle_wer": short_before["qwen_topk_oracle_wer"],
                "v2_qwen_top5_oracle_wer": short_after["qwen_topk_oracle_wer"],
            },
        },
        "universal_asr_guards": {
            "qwen_torgo_test_wer_before": baseline["qwen_torgo_test"]["micro_wer"],
            "qwen_torgo_test_wer_after": final["qwen_torgo_test"]["micro_wer"],
            "qwen_torgo_test_wer_delta": metric_delta(
                final["qwen_torgo_test"], baseline["qwen_torgo_test"], "micro_wer"
            ),
            "qwen_normal_test_wer_before": baseline["qwen_normal_test"]["micro_wer"],
            "qwen_normal_test_wer_after": final["qwen_normal_test"]["micro_wer"],
            "qwen_normal_test_wer_delta": metric_delta(
                final["qwen_normal_test"], baseline["qwen_normal_test"], "micro_wer"
            ),
            "ctc_torgo_test_wer": final["ctc_torgo_test"]["micro_wer"],
            "ctc_normal_test_wer": final["ctc_normal_test"]["micro_wer"],
        },
        "limitations": [
            "TORGO mostly contains prompted words and sentences, not a broad telegraphic-speech corpus.",
            "The three personal clips are diagnostic only and are too small to establish generalization.",
            "Top-k relative beam probabilities are comparative search scores, not calibrated confidence.",
            "No candidate ranker, intent inference, or grammar correction was used.",
        ],
    }
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-summary", type=Path, required=True)
    parser.add_argument("--v1-tsv", type=Path, required=True)
    parser.add_argument("--v2-summary", type=Path, required=True)
    parser.add_argument("--v2-tsv", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = build(
        args.v1_summary, args.v1_tsv, args.v2_summary, args.v2_tsv, args.result
    )
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    share = args.output.with_name("SHARE_THIS_WITH_CODEX.txt")
    share.write_text(
        "LITERAL_V2_DONE\n"
        + json.dumps(decision, indent=2)
        + "\n\nAlso attach or copy result.json if Codex asks for epoch-level detail.\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))
    print(f"LITERAL_V2_DECISION_DONE: paste {share}")


if __name__ == "__main__":
    main()
