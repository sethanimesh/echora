"""Shared plumbing for the TORGO bench scripts.

Every run_*.py script does the same three things: find the clips, decode them to
16 kHz mono, print a transcript per clip. That lives here so each model script is
just its own loading and decoding logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
CLIPS_DIR = BENCH_DIR / "clips"
RESULTS_DIR = BENCH_DIR / "results"
SAMPLE_RATE = 16000
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aiff", ".aif", ".ogg"}


@dataclass(frozen=True)
class Hypothesis:
    """One raw decoder hypothesis; score is acoustic/model-derived only."""

    text: str
    score: float | None = None


@dataclass(frozen=True)
class Prediction:
    """Ordered raw ASR hypotheses, before any candidate ranker or rewriting."""

    hypotheses: tuple[Hypothesis, ...]


def prediction_from(value: str | Hypothesis | Prediction | Sequence[str | Hypothesis]) -> Prediction:
    if isinstance(value, Prediction):
        prediction = value
    elif isinstance(value, Hypothesis):
        prediction = Prediction((value,))
    elif isinstance(value, str):
        prediction = Prediction((Hypothesis(value),))
    else:
        hypotheses = tuple(item if isinstance(item, Hypothesis) else Hypothesis(item) for item in value)
        prediction = Prediction(hypotheses)
    if not prediction.hypotheses:
        return Prediction((Hypothesis(""),))
    return prediction


def parse_args(model_id: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Transcribe every clip with {model_id}.")
    parser.add_argument(
        "--clips",
        type=Path,
        default=CLIPS_DIR,
        help="directory of audio files (default: research/benchmarks/clips)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="cpu is the safe local default; use cuda for the cloud benchmark",
    )
    parser.add_argument(
        "--tag",
        help="optional snapshot label, e.g. zero-shot-20260815",
    )
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[a-zA-Z0-9._-]+", args.tag):
        parser.error("--tag may contain only letters, digits, dot, underscore, and hyphen")
    return args


def find_clips(clips_dir: Path) -> list[Path]:
    if not clips_dir.is_dir():
        sys.exit(f"No such directory: {clips_dir}\nRecord something first:  python record.py")
    files = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit(f"No audio files in {clips_dir}\nRecord something first:  python record.py")
    return files


def load_audio(path: Path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "quiet", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        sys.exit(f"ffmpeg could not decode {path.name}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def internal_gap_stats(audio: np.ndarray) -> tuple[float, float]:
    """Estimate total/max internal low-energy gaps without trimming the audio.

    This is a reproducible benchmark descriptor, not clinical VAD. Low-energy
    phones may be counted as gaps, so retain the waveform and interpret the
    values only as relative stress slices.
    """
    frame_size = round(SAMPLE_RATE * 0.02)
    usable = len(audio) - (len(audio) % frame_size)
    if usable < frame_size:
        return 0.0, 0.0
    frames = audio[:usable].reshape(-1, frame_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64))
    noise_floor = float(np.quantile(rms, 0.20))
    speech_level = float(np.quantile(rms, 0.95))
    threshold = max(noise_floor + 0.10 * (speech_level - noise_floor), 1e-4)
    active = np.flatnonzero(rms > threshold)
    if len(active) < 2:
        return 0.0, 0.0
    internal_quiet = rms[active[0] : active[-1] + 1] <= threshold
    runs: list[int] = []
    current = 0
    for quiet in internal_quiet:
        if quiet:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    minimum_frames = round(0.12 / 0.02)
    seconds = [count * 0.02 for count in runs if count >= minimum_frames]
    return sum(seconds), max(seconds, default=0.0)


def whisper_processor(model_id: str, num_mel_bins: int):
    """Build a Whisper processor that matches the checkpoint's encoder.

    transformers >= 5 defaults the feature extractor to 128 mel bins, but every
    TORGO fine-tune here was trained on an 80-bin base. Without pinning this the
    first conv layer rejects the input.
    """
    from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizerFast

    extractor = WhisperFeatureExtractor.from_pretrained(model_id, feature_size=num_mel_bins)
    tokenizer = WhisperTokenizerFast.from_pretrained(model_id)
    return WhisperProcessor(extractor, tokenizer)


def reference_for(clip: Path) -> str | None:
    """The sentence you meant to say, if record.py stored one alongside the clip."""
    sidecar = clip.with_suffix(".txt")
    return sidecar.read_text().strip() if sidecar.is_file() else None


def run(key: str, model_id: str, note: str, load) -> None:
    """Load the model once, then transcribe every clip in the directory.

    `load(device)` returns a transcriber. It may emit one string or an ordered
    `Prediction` of raw decoder alternatives. No semantic ranker belongs here.
    """
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # these checkpoints trip harmless generate() warnings

    args = parse_args(model_id)
    clips = find_clips(args.clips)

    print(f"\n  model  {model_id}")
    print(f"  base   {note}")
    print(f"  device {args.device}")
    print(f"  clips  {len(clips)} file(s) from {args.clips}\n")

    print("  loading model (first run downloads weights)...", flush=True)
    started = time.time()
    transcribe = load(args.device)
    print(f"  ready in {time.time() - started:.1f}s\n")

    rows = []
    for clip in clips:
        audio = load_audio(clip)
        started = time.time()
        prediction = prediction_from(transcribe(audio))
        elapsed = time.time() - started
        audio_seconds = len(audio) / SAMPLE_RATE
        gap_total, gap_max = internal_gap_stats(audio)
        text = prediction.hypotheses[0].text.strip()
        alternatives = [item.text.strip() for item in prediction.hypotheses[1:]]

        print(f"  {clip.name}  ({audio_seconds:.1f}s audio, {elapsed:.1f}s decode)")
        reference = reference_for(clip)
        if reference:
            print(f"    said  {reference}")
        print(f"    got   {text or '(empty)'}\n")
        rows.append(
            {
                "clip": clip.name,
                "seconds": f"{elapsed:.2f}",
                "audio_seconds": f"{audio_seconds:.2f}",
                "rtf": f"{elapsed / audio_seconds:.3f}",
                "internal_gap_total": f"{gap_total:.2f}",
                "max_internal_gap": f"{gap_max:.2f}",
                "reference": reference or "",
                "transcript": text,
                "alternatives": json.dumps(alternatives) if alternatives else "",
                "acoustic_score": "" if prediction.hypotheses[0].score is None else prediction.hypotheses[0].score,
                "model_id": model_id,
                "device": args.device,
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    result_key = f"{key}__{args.tag}" if args.tag else key
    out = RESULTS_DIR / f"{result_key}.tsv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip", "seconds", "audio_seconds", "rtf",
                "internal_gap_total", "max_internal_gap", "reference",
                "transcript", "alternatives", "acoustic_score", "model_id", "device",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {out.relative_to(BENCH_DIR)}\n")

    if any(row["reference"] for row in rows):
        from metrics import score_file

        result = score_file(out)
        print(
            f"  literal ASR  WER {result.top1.wer:.1%}  CER {result.top1.cer:.1%}  "
            f"({result.utterances} referenced clip(s))\n"
        )
