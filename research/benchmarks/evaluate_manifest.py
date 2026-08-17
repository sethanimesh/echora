"""Resumable, literal-ASR evaluation of a JSONL speech manifest.

This evaluator appends one completed utterance at a time, so a long local or
cloud run can resume without retranscribing successful rows. It never invokes
Echora's semantic candidate ranker.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import common
from metrics import aggregate, score


RUNNERS = {
    "parakeet": "run_parakeet",
    "parakeet_ctc": "run_parakeet_ctc",
    "parakeet_tdt_1_1b": "run_parakeet_tdt_1_1b",
    "qwen3_asr": "run_qwen3_asr",
}
warnings.filterwarnings(
    "ignore",
    message=r"Using the model-agnostic default `max_length`.*",
    category=UserWarning,
)
FIELDS = [
    "id",
    "speaker",
    "condition",
    "audio_filepath",
    "audio_seconds",
    "decode_seconds",
    "rtf",
    "reference",
    "transcript",
    "alternatives",
    "acoustic_score",
    "model_id",
    "device",
]


def read_manifest(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    required = {"id", "speaker", "condition", "audio_filepath", "duration", "text"}
    for line_number, record in enumerate(records, start=1):
        missing = required - set(record)
        if missing:
            raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
        if not Path(record["audio_filepath"]).is_file():
            raise FileNotFoundError(record["audio_filepath"])
    return records


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {row["id"] for row in csv.DictReader(handle, delimiter="\t")}


def summarize(path: Path) -> dict:
    by_speaker: dict[str, list] = defaultdict(list)
    all_scores = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            item = score(row["reference"], row["transcript"])
            all_scores.append(item)
            by_speaker[row["speaker"]].append(item)
    overall = aggregate(all_scores)
    speaker_wer = {speaker: aggregate(items).wer for speaker, items in sorted(by_speaker.items())}
    return {
        "utterances": len(all_scores),
        "micro_wer": overall.wer,
        "micro_cer": overall.cer,
        "macro_speaker_wer": sum(speaker_wer.values()) / len(speaker_wer),
        "speaker_wer": speaker_wer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, help="validation subset; omitted means the full manifest")
    parser.add_argument("--speaker", action="append", help="include only this speaker; repeatable")
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    records = read_manifest(args.manifest)
    if args.speaker:
        speakers = set(args.speaker)
        records = [record for record in records if record["speaker"] in speakers]
    if args.limit:
        records = records[: args.limit]
    if not records:
        parser.error("no records selected")

    runner = importlib.import_module(RUNNERS[args.runner])
    done = completed_ids(args.output)
    remaining = [record for record in records if record["id"] not in done]
    if args.batch_size > 1:
        # Similar-duration batches avoid paying for extreme padding. Output
        # order is immaterial because IDs and speakers are explicit.
        remaining.sort(key=lambda record: (float(record["duration"]), record["id"]))
    print(
        f"runner={args.runner} model={runner.MODEL} selected={len(records)} "
        f"complete={len(done & {r['id'] for r in records})} remaining={len(remaining)}"
    )
    if not remaining:
        print(json.dumps(summarize(args.output), indent=2))
        return

    transcribe_batch = runner.load_batch(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not args.output.is_file() or args.output.stat().st_size == 0
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        if needs_header:
            writer.writeheader()
            handle.flush()
        completed_new = 0
        for offset in range(0, len(remaining), args.batch_size):
            batch_records = remaining[offset : offset + args.batch_size]
            audio_batch = [
                common.load_audio(Path(record["audio_filepath"])) for record in batch_records
            ]
            started = time.perf_counter()
            raw_predictions = transcribe_batch(audio_batch)
            elapsed = time.perf_counter() - started
            if len(raw_predictions) != len(batch_records):
                raise RuntimeError("batch decoder returned the wrong number of predictions")
            seconds_per_item = elapsed / len(batch_records)
            for record, audio, raw_prediction in zip(batch_records, audio_batch, raw_predictions):
                hypotheses = common.prediction_from(raw_prediction).hypotheses
                writer.writerow(
                    {
                        "id": record["id"],
                        "speaker": record["speaker"],
                        "condition": record["condition"],
                        "audio_filepath": record["audio_filepath"],
                        "audio_seconds": f"{len(audio) / common.SAMPLE_RATE:.3f}",
                        "decode_seconds": f"{seconds_per_item:.3f}",
                        "rtf": f"{seconds_per_item / (len(audio) / common.SAMPLE_RATE):.4f}",
                        "reference": record["text"],
                        "transcript": hypotheses[0].text.strip(),
                        "alternatives": json.dumps([item.text.strip() for item in hypotheses[1:]]),
                        "acoustic_score": "" if hypotheses[0].score is None else hypotheses[0].score,
                        "model_id": runner.MODEL,
                        "device": args.device,
                    }
                )
                completed_new += 1
            handle.flush()
            if completed_new % args.report_every < len(batch_records) or completed_new == len(remaining):
                print(f"completed {completed_new}/{len(remaining)} new utterances", flush=True)
    print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
