"""Pretty-print one command-v3 server result from stdin.

Alternatives are raw beam-search hypotheses.  The printed probabilities are
relative beam scores, not calibrated confidence, and no candidate ranking,
semantic repair or personal context has been applied.
"""

from __future__ import annotations

import json
import sys

RULE = "=" * 72
THIN = "-" * 72


def normalize(text: str) -> list[str]:
    kept = "".join(c if (c.isalnum() or c.isspace()) else " " for c in text.lower())
    return kept.split()


def main() -> int:
    payload = json.load(sys.stdin)
    if "error" in payload:
        print("SERVER ERROR:", payload["error"])
        if payload.get("traceback"):
            print(payload["traceback"])
        return 1

    reference = payload.get("reference", "")
    wanted = normalize(reference) if reference else None
    quality = payload.get("audio_quality", {})

    print()
    print(RULE)
    print("clip       {}   label: {}".format(payload.get("clip", ""), payload.get("label", "")))
    if reference:
        print("reference  {}".format(reference))
    print(
        "audio      {}s   peak {} dBFS   est SNR {} dB   clipped {}".format(
            quality.get("seconds"),
            quality.get("peak_dbfs"),
            quality.get("estimated_snr_db"),
            quality.get("clipped_samples"),
        )
    )
    if quality.get("low_level_warning"):
        print("           WARNING: level is low; move closer to the microphone")

    for name, result in payload.get("results", {}).items():
        print(THIN)
        tag = "no fine-tuning" if not result.get("fine_tuned", True) else "fine-tuned"
        print(
            "[{}]  ({})  greedy: {!r}   ({}s)".format(
                name, tag, result.get("greedy_top1", ""), result.get("decode_seconds")
            )
        )
        alternatives = result.get("alternatives", [])
        if not alternatives:
            print("   (no hypotheses returned)")
        for index, alternative in enumerate(alternatives, 1):
            text = alternative.get("text", "")
            flag = ""
            if wanted is not None and normalize(text) == wanted:
                flag = "   <== EXACT MATCH"
            print(
                "   {}. {:.3f}  {}{}".format(
                    index, alternative.get("relative_beam_probability", 0.0), text, flag
                )
            )
        if wanted is not None:
            hit = any(normalize(a.get("text", "")) == wanted for a in alternatives)
            print("   reference recovered in top-{}: {}".format(len(alternatives), "YES" if hit else "no"))

    print(RULE)
    print("Relative beam scores, not calibrated confidence. No ranking or repair applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
