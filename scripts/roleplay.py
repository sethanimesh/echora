"""Run the roleplay scripts in `docs/roleplay-scripts.md`, in either of two lanes.

The two plays make claims about three layers that fail independently, and a
single lane cannot separate them:

  the act      -- statement or question, long or short. Decided by the setting
                  and the listener, and deterministic given both.
  the detail   -- a profile's own wording in place of a heard word. Fires only
                  where the profile declared it, where it is in scope for the
                  setting, AND where the beams settled the word it rides on --
                  and that last condition belongs to the recognizer.
  the ambiguity-- two options rather than one. Only ever a choice among words
                  the recognizer actually produced.

`--voice` speaks each line with `say` and posts it to the running API, which is
the whole path including the recognizer. It is the honest end-to-end test and it
cannot demonstrate scoping, because a line the recognizer half-hears drops the
detail for a reason that has nothing to do with scope.

The default lane holds the beams fixed and varies only the setting, the listener
and the profile. That is the lane the scoping and precedence claims live in: when
the evidence cannot move, a detail that appears at home and not on the ward
appeared because of the scope and nothing else.

    ./scripts/roleplay.py --list
    ./scripts/roleplay.py --act "One:I"
    ./scripts/roleplay.py --act "One:I" --voice

Every scene is one Groq call, so `--list` prints the cost before you spend it.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.messaging import GroqMessageChain  # noqa: E402
from app.personal import Personalizer  # noqa: E402
from app.schemas import Hypothesis, resolve_listener  # noqa: E402


API = "http://127.0.0.1:8000"
# `say` is a stand-in for a microphone, not for a dysarthric voice. It is here so
# the voice lane can be run at all without a speaker present; a real run of these
# plays is a person reading them aloud.
VOICES = {"grace": "Moira", "tomas": "Daniel", "krishnan": "Rishi"}


def beams(*pairs: tuple[str, float]) -> list[Hypothesis]:
    """Beams with explicit search weights, as the recognizer actually returns them.

    The weights are the point. A detail may only ride on a word holding at least
    `ECHORA_PERSONAL_ANCHOR_SHARE` (0.75) of its slot, so writing `coffee 0.88`
    and `coffee 0.40` are two different tests, not two spellings of one.
    """
    return [
        Hypothesis(id=f"h{i}", literal_text=text, sequence_score=-0.1 * i, search_weight=weight)
        for i, (text, weight) in enumerate(pairs, 1)
    ]


@dataclass
class Scene:
    act: str
    persona: str
    context: str
    say: str
    evidence: list[Hypothesis]
    expect: str
    # What the place the speaker tapped declares. None is the ordinary case: the
    # place says nothing and the profile is asked before the setting's default.
    declared: str | None = None
    watch: str = ""


def _s(act, persona, context, say, evidence, expect, declared=None, watch="") -> Scene:
    return Scene(act, persona, context, say, evidence, expect, declared, watch)


# --------------------------------------------------------------------- the plays

WASHROOM = beams(("washroom", 0.70), ("wash room", 0.18), ("washed room", 0.07), ("washboard", 0.05))
TEA = beams(("tea", 0.86), ("t", 0.06), ("tee", 0.05), ("teaa", 0.03))
SOUP = beams(("soup", 0.82), ("suit", 0.08), ("suite", 0.06), ("seed", 0.04))
CREAM = beams(("cream", 0.84), ("creem", 0.08), ("scream", 0.05), ("crem", 0.03))
BUZZER = beams(("buzzer", 0.80), ("buzzah", 0.12), ("busser", 0.08))
HELP = beams(("help", 0.85), ("helped", 0.08), ("hep", 0.07))
WATER = beams(("water", 0.84), ("warter", 0.09), ("waiter", 0.07))
# Two real words, near-evenly split. This is what a genuine ambiguity looks like,
# and it is the only thing that can produce the choice screen.
DONNA_DAWN = beams(("donna", 0.38), ("dawn", 0.34), ("dorn", 0.16), ("danna", 0.12))
# The same sound with the weight collapsed onto one word. Nothing to choose.
DAWN_ONLY = beams(("dawn", 0.81), ("dorn", 0.11), ("darn", 0.08))
# The anchor contested rather than settled: `soup` is there, and holds too little
# of its slot for a detail to be allowed to ride on it.
SOUP_UNSURE = beams(("suit", 0.34), ("soup", 0.30), ("sleep", 0.20), ("suite", 0.16))

TABLET = beams(
    ("I need my tablet", 0.78),
    ("I need my tablets", 0.10),
    ("I need my table", 0.07),
    ("I need my tabla", 0.05),
)
HEADSET = beams(
    ("where is my headset", 0.80),
    ("where is my head set", 0.10),
    ("where is my head sit", 0.06),
    ("where is my hedset", 0.04),
)
COFFEE = beams(("coffee", 0.88), ("coffey", 0.06), ("coffe", 0.06))
STANDUP = beams(
    ("the standup is at ten", 0.74),
    ("the stand up is at ten", 0.18),
    ("the standup is at 10", 0.08),
)
PLATFORM = beams(
    ("which platform cascais train", 0.79),
    ("which platform cascai train", 0.12),
    ("which platform kashkaish train", 0.09),
)


SCENES: list[Scene] = [
    # -------------------------------------------------- Play One, Grace Okonkwo
    _s("One:I", "grace", "home", "washroom", WASHROOM, "a statement: the carer can be sent"),
    _s("One:I", "grace", "home", "tea", TEA, "the Lipton detail, marked and revertible"),
    _s("One:I", "grace", "home", "soup", SOUP, "the egusi detail: scoped `home`, and this is home"),
    _s("One:I", "grace", "home", "cream", CREAM, "the blue tub detail: scoped `home, care`"),
    _s("One:I", "grace", "home", "buzzer", BUZZER, "lexicon only, no detail to apply"),
    _s(
        "One:I", "grace", "home", "Dawn", DONNA_DAWN,
        "TWO options. Donna and Dawn are both real words in the evidence",
        watch="One option here is the failure. The evidence is genuinely split, so it must ask.",
    ),
    _s(
        "One:I", "grace", "home", "Dawn (weight collapsed)", DAWN_ONLY,
        "ONE option. `donna` is not in the evidence, so nothing can offer it",
        watch="The prior chooses among the recognizer's words. It cannot add one.",
    ),
    _s("One:II", "grace", "care", "washroom", WASHROOM, "unchanged: care is familiar too"),
    _s("One:II", "grace", "care", "tea", TEA, "still specialized: `home, care` covers the ward"),
    _s("One:II", "grace", "care", "soup", SOUP, "plain soup. Scoped `home`, so no egusi on the ward"),
    _s("One:II", "grace", "care", "cream", CREAM, "still specialized"),
    _s(
        "One:II", "grace", "home", "soup (anchor unsettled)", SOUP_UNSURE,
        "no detail, and this is home. The gate is the anchor share, not the scope",
        watch="This is the row that tells a missing detail from a scoping bug.",
    ),
    _s("One:III", "grace", "outdoors", "washroom", WASHROOM, "a question, and no longer"),
    _s("One:III", "grace", "outdoors", "help", HELP, "asked directly, no preamble"),
    _s("One:III", "grace", "outdoors", "tea", TEA, "scoped out AND flipped to a counter request"),
    _s("One:III", "grace", "outdoors", "water", WATER, "over a counter, not a statement of want"),
    _s(
        "One:IV", "grace", "outdoors", "washroom", WASHROOM,
        "a statement again. The setting did not change; the listener did",
        declared="familiar",
    ),
    _s(
        "One:IV", "grace", "outdoors", "tea", TEA,
        "the detail stays away: still scoped `home, care`",
        declared="familiar",
        watch="Judge this act on washroom. A bare noun carries no act for the listener to flip.",
    ),
    # ------------------------------------------------------- Play Two, Tomás
    _s("Two:I", "tomas", "home", "I need my tablet", TABLET, "the baclofen detail: scoped `home, care`"),
    _s("Two:I", "tomas", "home", "where is my headset", HEADSET, "the noise cancelling detail: `home, general`"),
    _s("Two:I", "tomas", "home", "coffee", COFFEE, "unscoped, so it applies here and everywhere"),
    _s(
        "Two:I", "tomas", "home", "the standup is at ten", STANDUP,
        "a whole adult sentence, not a two-word telegram",
        watch="Grace's brevity is not his. A flattened message here is the bug.",
    ),
    _s("Two:II", "tomas", "care", "I need my tablet", TABLET, "unchanged: `home, care`"),
    _s("Two:II", "tomas", "care", "where is my headset", HEADSET, "the detail drops: the clinic is neither"),
    _s("Two:II", "tomas", "care", "coffee", COFFEE, "unchanged: unscoped"),
    _s("Two:III", "tomas", "outdoors", "which platform, Cascais", PLATFORM, "the Cascais detail survives: unscoped"),
    _s(
        "Two:III", "tomas", "outdoors", "coffee", COFFEE,
        "detail kept, act flipped -- but see the Cascais row, which is the reliable one",
        watch="offered=1 applied=0 refused=1 here means the model chose brevity, not a scoping failure.",
    ),
    _s("Two:III", "tomas", "outdoors", "washroom", WASHROOM, "asked, not stated"),
    _s("Two:III", "tomas", "outdoors", "I need my tablet", TABLET, "detail dropped: `home, care` does not cover this"),
    _s(
        "Two:IV", "tomas", "outdoors", "coffee", COFFEE,
        "stated: the Office place declares a familiar listener",
        declared="familiar",
    ),
    _s("Two:IV", "tomas", "outdoors", "coffee", COFFEE, "asked: plain Outdoors declares nothing, and Tomás declares nothing"),
    _s(
        "Two:IV", "krishnan", "outdoors", "washroom", WASHROOM,
        "stated: no place said so, his profile did (`listener_by_setting`)",
        watch="The middle rung of the chain. It is only reachable when the place abstains.",
    ),
]


# ------------------------------------------------------------------- the lanes


async def fixed_lane(scenes: list[Scene]) -> None:
    settings = load_settings()
    personal = Personalizer(settings)
    chain = GroqMessageChain(settings)
    for scene in scenes:
        listener = resolve_listener(
            scene.context,  # type: ignore[arg-type]
            scene.declared,  # type: ignore[arg-type]
            personal.listener_for(scene.persona, scene.context),  # type: ignore[arg-type]
        )
        brief = await personal.brief(scene.evidence, scene.context, scene.persona, listener)  # type: ignore[arg-type]
        result = await chain.run(scene.evidence, scene.context, brief=brief, listener=listener)  # type: ignore[arg-type]
        _report(
            scene,
            listener,
            [h.literal_text for h in scene.evidence],
            result.ranker,
            result.messages,
            warnings=result.warnings,
        )


def _say(text: str, voice: str) -> Path:
    target = Path(__file__).resolve().parent / ".roleplay-clips"
    target.mkdir(exist_ok=True)
    path = target / f"{voice}-{''.join(c if c.isalnum() else '_' for c in text)}.wav"
    if not path.is_file():
        subprocess.run(
            ["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text], check=True
        )
    return path


def voice_lane(scenes: list[Scene]) -> None:
    import httpx

    from app.schemas import RankerDecision

    with httpx.Client(timeout=300) as client:
        health = client.get(f"{API}/api/v1/health")
        if health.status_code != 200 or not health.json().get("model_ready"):
            raise SystemExit("The API is not ready. Start it with ./scripts/dev.sh first.")
        for scene in scenes:
            clip = _say(scene.say, VOICES.get(scene.persona, "Samantha"))
            with clip.open("rb") as handle:
                response = client.post(
                    f"{API}/api/v1/transcriptions",
                    files={"audio": (clip.name, handle, "audio/wav")},
                    data={
                        "context": scene.context,
                        "persona": scene.persona,
                        "listener": scene.declared or "",
                    },
                )
            response.raise_for_status()
            body = response.json()
            _report(
                scene,
                body["listener"],
                [h["literal_text"] for h in body["hypotheses"]],
                RankerDecision(**body["ranker"]),
                body["messages"],
                raw=True,
                warnings=body.get("warnings"),
            )


def _report(
    scene: Scene,
    listener: str,
    literals: list[str],
    ranker,
    messages,
    raw: bool = False,
    warnings: list[str] | None = None,
) -> None:
    print(f"=== {scene.act}  {scene.persona} · {scene.context} · {listener}")
    print(f'  said     "{scene.say}"')
    print(f"  heard    {literals}")
    # A failed Groq call comes back as `ambiguous` with the raw beams as the
    # options, which reads exactly like a genuine ambiguity -- and a rate limit
    # is the commonest way to see one. Saying so here is the difference between
    # recording a real finding and recording an outage.
    if ranker.source == "unavailable":
        print("  DID NOT RUN -- the message assistant was unavailable, so these are raw beams.")
        for warning in warnings or ([ranker.reason] if ranker.reason else []):
            print(f"           {warning}")
        print()
        return
    print(f"  decision {ranker.decision}")
    for item in messages:
        message = item["corrected_text"] if raw else item.corrected_text
        details = item["specializations"] if raw else item.specializations
        plain = item["plain_text"] if raw else item.plain_text
        mark = ""
        if details:
            surfaces = [d["surface"] if raw else d.surface for d in details]
            mark = f"   [detail: {', '.join(surfaces)} · plain: {plain!r}]"
        print(f"    -> {message!r}{mark}")
    trace = ranker.personalization
    if trace:
        print(
            f"  personal offered={trace.specializations_offered} applied={trace.specializations_applied} "
            f"refused={trace.specializations_refused} hints={trace.lexicon_hints} examples={trace.examples_used}"
        )
    print(f"  expected {scene.expect}")
    if scene.watch:
        print(f"  watch    {scene.watch}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act", action="append", default=[], help='e.g. "One" or "One:I". Repeatable.')
    parser.add_argument("--say", default="", help="only lines whose spoken text contains this")
    parser.add_argument("--voice", action="store_true", help="speak each line into the running API")
    parser.add_argument("--list", action="store_true", help="show the acts and what a run would cost")
    args = parser.parse_args()

    # `One:II` must not sweep up `One:III`, so a prefix only matches on a
    # separator. `One` still selects the whole play.
    def wanted(scene: Scene) -> bool:
        if args.say and args.say.lower() not in scene.say.lower():
            return False
        return not args.act or any(
            scene.act == a or scene.act.startswith(f"{a}:") for a in args.act
        )

    chosen = [s for s in SCENES if wanted(s)]
    if args.list or not chosen:
        acts: dict[str, int] = {}
        for scene in SCENES:
            acts[scene.act] = acts.get(scene.act, 0) + 1
        print("act        lines   (one Groq call each)")
        for act, count in acts.items():
            print(f"  {act:<9} {count:>3}")
        print(f"\n  all       {len(SCENES):>3}")
        if not chosen and args.act:
            raise SystemExit(f"\nNo act matched {args.act}.")
        return

    print(f"{len(chosen)} scene(s), one Groq call each.\n")
    if args.voice:
        voice_lane(chosen)
    else:
        asyncio.run(fixed_lane(chosen))


if __name__ == "__main__":
    main()
