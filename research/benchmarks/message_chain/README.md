# Message-chain eval

Measures the post-ASR chain only. Beams are fixed fixtures in `cases.json`, so
ASR is held constant and the sole variable is the Groq chain plus the
deterministic layers around it. The TORGO / command / normal-speech WER
benchmarks are upstream of all of this and cannot move when the chain changes.

    .venv/bin/python research/benchmarks/message_chain/run_eval.py --tag before
    # ... change the chain ...
    .venv/bin/python research/benchmarks/message_chain/run_eval.py --tag after

## Reading the result

`chain unavailable (API)` must be **0**. Anything else means trials fell back to
raw beams and scored as ambiguous, which looks identical to the chain choosing
to ask. A run with API failures is not a valid datapoint.

## Token budget

The chain is a single `gpt-oss-120b` call at roughly 1.9k tokens per trial,
against a 200k tokens/day free tier. Twenty cases at two trials is ~40% of a
day's allowance, so plan on at most two runs per day. `--repeat 1` is not enough: the model is
not deterministic at `temperature=0` and single runs flip.

## Case categories

- `noise-slot` — one word heard several ways; the chain should commit and keep
  the content word. This is the category the slot-resolution redesign targets.
- `meaningful-slot` — variants are genuinely different messages; the chain must
  keep asking. Guards against the redesign becoming reckless.
- `hard` — no reading is well supported; must stay ambiguous and invent nothing.
- `regression` — behavior that already worked and must not break.
