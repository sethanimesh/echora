"""Repair accepted-message records written by two bugs in the acceptance path.

Both bugs wrote to `data/personal/*/history.jsonl`, and both are fixed now, but
the records they left behind keep working: they are retrieved as few-shot
examples and they teach the model the wrong thing.

  1. `Personalizer.remember` took a listener and never passed it on, so every
     message the application stored was stamped `familiar`. Outdoors is the only
     setting whose default is `unfamiliar`, so those are the rows that are wrong.
     One of them -- Grace's "Tea, please." accepted on a platform -- is now the
     highest-scoring example for outdoors-and-familiar, a stance it was never
     said in, and the model copies its shape.

  2. Choosing an option while the message assistant was unavailable stored a raw
     beam as an accepted message. Those are recognizable because the message is
     byte-identical to the recognizer's own words: `nead coffee -> nead coffee`.

`data/personas/history/` is the authority. It is tracked, it is never written
to, and its listeners were always right -- so a live row that came from it has
its listener restored rather than guessed at. That also repairs a third, older
drift: some live copies were made before accepted messages carried a listener at
all and read back with none, and one was rewritten wholesale as `familiar`.

Only rows with no baseline behind them are reasoned about, and only two ways:
an outdoors row stamped `familiar` could only have come from the broken path,
and a message identical to the recognizer's own words was never a message.

    ./scripts/repair_personal_history.py            # show what would change
    ./scripts/repair_personal_history.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.schemas import default_listener  # noqa: E402

BASELINE = ROOT / "data" / "personas" / "history"
LIVE = ROOT / "data" / "personal"


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    total_fixed = total_dropped = 0
    for directory in sorted(p for p in LIVE.iterdir() if p.is_dir()):
        history = directory / "history.jsonl"
        live = rows(history)
        if not live:
            continue
        shipped = {
            (r.get("heard"), r.get("message")): r.get("listener")
            for r in rows(BASELINE / f"{directory.name}.jsonl")
        }

        kept: list[dict] = []
        fixed = dropped = 0
        for row in live:
            key = (row.get("heard"), row.get("message"))
            context = row.get("context")
            was = row.get("listener")
            # A row that came from the baseline has a known-correct listener, and
            # the baseline is never written to. Restoring it needs no inference:
            # some live copies predate the field entirely and read back as null,
            # others were rewritten wholesale as `familiar`.
            if key in shipped:
                if was != shipped[key]:
                    print(f"  stamp {directory.name}: {row.get('message')!r} [{context}] {was} -> {shipped[key]}")
                    row = {**row, "listener": shipped[key]}
                    fixed += 1
                kept.append(row)
                continue
            # Raw ASR: the assistant never ran, so the "message" is the beam.
            if row.get("message") == row.get("heard"):
                print(f"  drop  {directory.name}: {row.get('message')!r} (raw ASR, never repaired)")
                dropped += 1
                continue
            # Written by the application, and the only stamp its broken path could
            # produce was `familiar`. Outdoors is the one setting whose default
            # differs, so it is the only one that can have been mislabelled -- and
            # the browser did send `unfamiliar` for it.
            correct = default_listener(context) if context else None
            if correct and (was is None or (was == "familiar" and correct != "familiar")):
                print(f"  stamp {directory.name}: {row.get('message')!r} [{context}] {was} -> {correct}")
                row = {**row, "listener": correct}
                fixed += 1
            kept.append(row)

        if not (fixed or dropped):
            continue
        total_fixed += fixed
        total_dropped += dropped
        if args.apply:
            shutil.copy2(history, history.with_suffix(".jsonl.before-repair"))
            history.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
            )

    print()
    if not (total_fixed or total_dropped):
        print("Nothing to repair.")
        return
    print(f"{total_fixed} row(s) restamped, {total_dropped} dropped.")
    if args.apply:
        print("Written. The previous file is beside each one as .jsonl.before-repair.")
    else:
        print("Nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
