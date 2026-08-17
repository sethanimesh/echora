"""Create deterministic pause-stress copies of referenced benchmark clips."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import common

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "research" / "training"))
from augment import insert_pause_at_low_energy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=common.CLIPS_DIR)
    parser.add_argument(
        "--output", type=Path, default=REPO_DIR / "data" / "derived" / "pause_stress"
    )
    parser.add_argument("--seconds", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    clips = common.find_clips(args.clips)
    written = 0
    for seconds in args.seconds:
        if seconds <= 0:
            parser.error("pause durations must be positive")
        output = args.output / f"{round(seconds * 1000):04d}ms"
        output.mkdir(parents=True, exist_ok=True)
        for clip in clips:
            reference = common.reference_for(clip)
            if not reference:
                continue
            audio = common.load_audio(clip)
            # Reset to a clip-specific seed for every duration: the insertion
            # boundary stays fixed, so only pause length changes across suites.
            name_seed = int.from_bytes(hashlib.sha256(clip.name.encode()).digest()[:8], "big")
            rng = np.random.default_rng(args.seed ^ name_seed)
            stressed = insert_pause_at_low_energy(audio, common.SAMPLE_RATE, seconds, rng)
            target = output / clip.with_suffix(".wav").name
            sf.write(target, stressed, common.SAMPLE_RATE, subtype="PCM_16")
            target.with_suffix(".txt").write_text(reference + "\n")
            written += 1
    print(f"wrote {written} referenced stress clips under {args.output}")


if __name__ == "__main__":
    main()
