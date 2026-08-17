import csv
import json

from research.training.make_literal_v2_decision import build


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_tsv(path, qwen_top1, hypotheses, ctc=""):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reference", "slice", "qwen_top1", "qwen_hypotheses", "ctc_transcript"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "reference": "I water",
                "slice": "personal_original",
                "qwen_top1": qwen_top1,
                "qwen_hypotheses": json.dumps([{"text": item} for item in hypotheses]),
                "ctc_transcript": ctc,
            }
        )


def summary(ctc=False):
    value = {
        "utterances": 2,
        "qwen_top1_wer": 0.5,
        "qwen_top1_cer": 0.2,
        "qwen_topk_oracle_wer": 0.25,
        "qwen_topk_oracle_cer": 0.1,
        "literal_present_in_topk": 1,
        "literal_present_in_topk_rate": 0.5,
    }
    if ctc:
        value.update({"ctc_wer": 0.2, "ctc_cer": 0.1})
    return {"slices": {"torgo_1word": value, "torgo_2_3word": value}}


def metrics(wer):
    return {"micro_wer": wer}


def test_decision_prefers_exact_literal_ctc(tmp_path):
    v1_summary = tmp_path / "v1.json"
    v2_summary = tmp_path / "v2.json"
    v1_tsv = tmp_path / "v1.tsv"
    v2_tsv = tmp_path / "v2.tsv"
    result = tmp_path / "result.json"
    write_json(v1_summary, summary())
    write_json(v2_summary, summary(ctc=True))
    write_tsv(v1_tsv, "gotten", ["gotten", "high button"])
    write_tsv(v2_tsv, "gotten", ["gotten", "high button"], "i water")
    write_json(
        result,
        {
            "baseline_v1": {"qwen_torgo_test": metrics(0.56), "qwen_normal_test": metrics(0.05)},
            "final_evaluation": {
                "qwen_torgo_test": metrics(0.54),
                "qwen_normal_test": metrics(0.06),
                "ctc_torgo_test": metrics(0.60),
                "ctc_normal_test": metrics(0.20),
            },
        },
    )
    decision = build(v1_summary, v1_tsv, v2_summary, v2_tsv, result)
    assert decision["decision_branch"] == "keep_dual_literal_outputs"
    assert decision["personal_i_water"]["v2_ctc_exact"] is True
    assert decision["universal_asr_guards"]["qwen_torgo_test_wer_delta"] < 0
