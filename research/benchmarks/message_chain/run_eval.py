"""Evaluate the post-ASR message chain against fixed beam sets.

Nothing here touches ASR. Beams are held constant so the only variable is the
Groq chain and the deterministic layers around it.

    .venv/bin/python research/benchmarks/message_chain/run_eval.py --tag before
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import load_settings                                  # noqa: E402
from app.messaging.groq_chain import GroqMessageChain, _tokens  # noqa: E402
from app.schemas import Hypothesis                                    # noqa: E402

HERE = Path(__file__).resolve().parent


def build(case: dict) -> list[Hypothesis]:
    return [
        Hypothesis(id=f"h{i}", literal_text=text, sequence_score=-0.1 * i, search_weight=weight)
        for i, (text, weight) in enumerate(case["beams"], 1)
    ]


def score(case: dict, result) -> dict:
    """Grade the reading (grounded evidence) and the message (free English) apart.

    The message is deliberately allowed to rephrase -- "leg pain" becomes "My leg
    hurts" -- so content-word retention is measured on the reading, which is the
    thing the code actually grounds.
    """
    displayed = result.messages
    top = displayed[0].corrected_text if displayed else ""
    reading = displayed[0].interpreted_intent if displayed else ""
    reading_tokens = set(_tokens(reading))
    beam_tokens = {tok for text, _ in case["beams"] for tok in _tokens(text)}

    decision = result.ranker.decision
    required = case["required_terms"]
    retained = [term for term in required if term in reading_tokens]

    # The one guarantee enforced in code: no reading word the recognizer never heard.
    invented = sorted(
        {
            tok
            for message in displayed
            for tok in _tokens(message.interpreted_intent)
            if tok not in beam_tokens
        }
    )
    forbidden = set(case["forbidden_terms"])
    forbidden_hit = sorted(
        forbidden & (reading_tokens | {t for t in _tokens(top)})
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "context": case["context"],
        "listener": case.get("listener", "familiar"),
        "truth": case["truth"],
        "expect": case["expect"],
        "decision": decision,
        "decision_ok": decision == case["expect"],
        "unavailable": result.ranker.source == "unavailable",
        "options": len(displayed),
        "top_message": top,
        "reading": reading,
        "all_messages": " | ".join(m.corrected_text for m in displayed),
        "retention": len(retained) / len(required) if required else None,
        "missing_terms": sorted(set(required) - reading_tokens),
        "invented": invented,
        "forbidden_hit": forbidden_hit,
        "reason": result.ranker.reason,
        "warnings": "; ".join(result.warnings),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="label for this run, e.g. before / after")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=2,
                        help="runs per case; gpt-oss is not deterministic at temperature=0. "
                             "Budget: ~1.9k ranker tokens per trial against a 200k/day free "
                             "tier, so 20 cases x 2 trials is about 40%% of a day's allowance.")
    args = parser.parse_args()

    payload = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    cases = payload["cases"]
    settings = load_settings()
    if not settings.groq_configured:
        raise SystemExit("GROQ_API_KEY is required for this eval")
    chain = GroqMessageChain(settings)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(case: dict, trial: int) -> dict:
        async with semaphore:
            result = await chain.run(
                build(case), case["context"], listener=case.get("listener", "familiar")
            )
            row = score(case, result)
            row["trial"] = trial
            return row

    trials = await asyncio.gather(
        *(one(case, t) for case in cases for t in range(args.repeat))
    )
    order = [c["id"] for c in cases]
    trials.sort(key=lambda r: (order.index(r["id"]), r["trial"]))

    # Collapse trials per case; a case counts as correct only on the majority verdict.
    rows = []
    for case in cases:
        group = [r for r in trials if r["id"] == case["id"]]
        ok = sum(1 for r in group if r["decision_ok"])
        best = max(group, key=lambda r: r["decision_ok"])
        merged = dict(best)
        merged["decision_ok"] = ok * 2 > len(group)
        merged["trials_ok"] = f"{ok}/{len(group)}"
        merged["stable"] = ok in (0, len(group))
        merged["retention"] = (
            sum(r["retention"] for r in group) / len(group)
            if group[0]["retention"] is not None else None
        )
        merged["any_halluc"] = any(r["invented"] or r["forbidden_hit"] for r in group)
        merged["unavailable_trials"] = sum(1 for r in group if r["unavailable"])
        rows.append(merged)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.tag}.json").write_text(
        json.dumps({"summary": rows, "trials": trials}, indent=2), encoding="utf-8"
    )

    def rate(subset: list[dict], key: str) -> str:
        if not subset:
            return "n/a"
        hits = sum(1 for r in subset if r[key])
        return f"{hits}/{len(subset)} ({100*hits/len(subset):.0f}%)"

    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)

    retention_rows = [r for r in rows if r["retention"] is not None]
    retention = (
        sum(r["retention"] for r in retention_rows) / len(retention_rows)
        if retention_rows else 0.0
    )
    hallucinated = [r for r in rows if r["any_halluc"]]

    print(f"\n=== message-chain eval [{args.tag}] — {len(rows)} cases x {args.repeat} trials ===\n")
    for row in rows:
        flag = "ok " if row["decision_ok"] else "FAIL"
        bad = " HALLUC" if row["any_halluc"] else ""
        miss = f" missing={row['missing_terms']}" if row["missing_terms"] else ""
        print(f"[{flag}]{bad} {row['id']:<22} {row['trials_ok']:>4} {row['decision']:<9} "
              f"opts={row['options']}  {row['top_message']!r}  <- {row['reading']!r}{miss}")
        if row["invented"]:
            print(f"          invented: {row['invented']}")
        if row["forbidden_hit"]:
            print(f"          forbidden: {row['forbidden_hit']}")

    print("\n--- summary ---")
    print(f"decision correct overall : {rate(rows, 'decision_ok')}")
    for cat in ("noise-slot", "meaningful-slot", "hard", "regression"):
        print(f"  {cat:<17}      : {rate(by_cat.get(cat, []), 'decision_ok')}")
    print(f"reading retention        : {100*retention:.0f}%  (over {len(retention_rows)} cases)")
    print(f"cases with invented words: {len(hallucinated)}/{len(rows)}")
    api_fail = sum(r["unavailable_trials"] for r in rows)
    print(f"chain unavailable (API)  : {api_fail}/{len(trials)} trials")
    if api_fail:
        print("  WARNING: API failures fall back to raw beams and score as ambiguous.")
        print("  Any run with a non-zero count here is not a valid before/after datapoint.")
    unstable = [r["id"] for r in rows if not r["stable"]]
    print(f"unstable across trials   : {len(unstable)}/{len(rows)} {unstable}")
    print(f"\nwritten: {out / (args.tag + '.json')}")


if __name__ == "__main__":
    asyncio.run(main())
