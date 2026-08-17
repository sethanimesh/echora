"""Build clean TORGO manifests and nested speaker-disjoint evaluation folds.

Only one microphone is used for each utterance.  Prompt images and vocal-task
instructions are excluded because their prompt files are not transcripts of
what the speaker actually said.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import wave
from collections import Counter
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_DIR / "data" / "raw" / "torgo"
DEFAULT_OUTPUT = REPO_DIR / "data" / "derived" / "torgo"
DYSARTHRIC_GROUPS = {"F", "M"}
MIC_DIRS = {"head": "wav_headMic", "array": "wav_arrayMic"}
ANNOTATION_RE = re.compile(r"\s*\[as in\b.*\]\s*$", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[^\W_]+(?:['\u2019][^\W_]+)*", text, re.UNICODE))


def clean_prompt(raw: str) -> tuple[str | None, str | None]:
    """Return a usable literal transcript or a documented exclusion reason."""
    raw = unicodedata.normalize("NFKC", raw).strip()
    if not raw:
        return None, "empty_prompt"
    if any(unicodedata.category(char).startswith("C") and not char.isspace() for char in raw):
        return None, "control_characters"
    compact = SPACE_RE.sub(" ", raw)
    lowered = compact.lower()
    if lowered.startswith("input/images/"):
        return None, "spontaneous_audio_without_transcript"
    if compact.startswith("["):
        return None, "non_asr_vocal_instruction"
    compact = ANNOTATION_RE.sub("", compact).strip()
    if not normalized_text(compact):
        return None, "no_lexical_transcript"
    return compact, None


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def collect(source: Path, mic: str) -> tuple[list[dict], Counter]:
    records: list[dict] = []
    excluded: Counter = Counter()
    mic_dir = MIC_DIRS[mic]
    for prompt in sorted(source.glob("*/*/Session*/prompts/*.txt")):
        session_dir = prompt.parent.parent
        audio = session_dir / mic_dir / f"{prompt.stem}.wav"
        if not audio.is_file():
            excluded["missing_selected_microphone"] += 1
            continue
        raw = prompt.read_text(errors="replace")
        text, reason = clean_prompt(raw)
        if reason:
            excluded[reason] += 1
            continue
        relative = prompt.relative_to(source)
        group, speaker, session = relative.parts[0], relative.parts[1], relative.parts[2]
        canonical = normalized_text(text or "")
        records.append(
            {
                "id": f"torgo-{speaker}-{session}-{prompt.stem}",
                "audio_filepath": str(audio.resolve()),
                "duration": round(wav_duration(audio), 4),
                "text": text,
                "text_id": hashlib.sha256(canonical.encode()).hexdigest()[:16],
                "speaker": speaker,
                "session": session,
                "gender": "female" if group.startswith("F") else "male",
                "condition": "control" if group.endswith("C") else "dysarthric",
                "corpus": "TORGO",
                "microphone": mic,
                "prompt_type": "read",
            }
        )
    return records, excluded


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def make_folds(records: list[dict], output: Path) -> list[dict]:
    dysarthric_speakers = sorted(
        {record["speaker"] for record in records if record["condition"] == "dysarthric"}
    )
    summaries: list[dict] = []
    for test_speaker in dysarthric_speakers:
        test = [record for record in records if record["speaker"] == test_speaker]
        test_text_ids = {record["text_id"] for record in test}
        # A simple rotation can leave a zero-example strict dev set because
        # TORGO repeats so many prompts. Select an inner held-out speaker with
        # the most material not present in the outer test speaker instead.
        dev_speaker = max(
            (speaker for speaker in dysarthric_speakers if speaker != test_speaker),
            key=lambda speaker: sum(
                record["text_id"] not in test_text_ids
                for record in records
                if record["speaker"] == speaker
            ),
        )
        dev = [record for record in records if record["speaker"] == dev_speaker]
        train = [
            record
            for record in records
            if record["speaker"] not in {test_speaker, dev_speaker}
        ]
        strict_train = [record for record in train if record["text_id"] not in test_text_ids]
        strict_dev = [record for record in dev if record["text_id"] not in test_text_ids]

        fold_dir = output / "folds" / test_speaker
        for name, subset in (
            ("train", train),
            ("dev", dev),
            ("test", test),
            ("train_no_test_text", strict_train),
            ("dev_no_test_text", strict_dev),
        ):
            write_jsonl(fold_dir / f"{name}.jsonl", subset)
        summaries.append(
            {
                "test_speaker": test_speaker,
                "dev_speaker": dev_speaker,
                "train": len(train),
                "dev": len(dev),
                "test": len(test),
                "train_no_test_text": len(strict_train),
                "dev_no_test_text": len(strict_dev),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mic", choices=sorted(MIC_DIRS), default="head")
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"TORGO directory does not exist: {args.source}")

    records, excluded = collect(args.source, args.mic)
    if not records:
        parser.error("no paired, usable prompt/audio records found")
    write_jsonl(args.output / "all.jsonl", records)
    write_jsonl(
        args.output / "dysarthric.jsonl",
        [record for record in records if record["condition"] == "dysarthric"],
    )
    write_jsonl(
        args.output / "control.jsonl",
        [record for record in records if record["condition"] == "control"],
    )
    folds = make_folds(records, args.output)
    summary = {
        "microphone": args.mic,
        "utterances": len(records),
        "hours": round(sum(record["duration"] for record in records) / 3600, 3),
        "speakers": sorted({record["speaker"] for record in records}),
        "counts_by_condition": dict(Counter(record["condition"] for record in records)),
        "excluded": dict(excluded),
        "folds": folds,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
