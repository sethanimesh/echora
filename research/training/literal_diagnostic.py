"""Score Qwen top-k hypotheses and optional character-CTC literal output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "research" / "benchmarks"))
from metrics import aggregate, normalize, score  # noqa: E402

try:
    from .literal_ctc import LiteralCTCHead, greedy_decode
    from .qwen_torgo_pilot import load_audio, load_model, read_json, read_jsonl
except ImportError:
    from literal_ctc import LiteralCTCHead, greedy_decode
    from qwen_torgo_pilot import load_audio, load_model, read_json, read_jsonl


def load_qwen_adapter(model, path: Path) -> None:
    state = load_file(path, device="cpu")
    parameters = dict(model.named_parameters())
    unknown = set(state) - set(parameters)
    if unknown:
        raise RuntimeError(f"Adapter contains unknown Qwen parameters: {sorted(unknown)[:5]}")
    with torch.no_grad():
        for name, tensor in state.items():
            if parameters[name].shape != tensor.shape:
                raise RuntimeError(f"Adapter shape mismatch: {name}")
            parameters[name].copy_(tensor.to(parameters[name].device, parameters[name].dtype))


def qwen_nbest(model, processor, audio, beams: int, max_new_tokens: int) -> list[dict]:
    inputs = processor.apply_transcription_request(
        audio=[audio], language=["English"]
    ).to(model.device, model.dtype)
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=beams,
                num_return_sequences=beams,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        model.config.use_cache = previous_cache
    token_ids = generated.sequences[:, inputs["input_ids"].shape[1] :]
    texts = processor.decode(token_ids, return_format="transcription_only")
    if generated.sequences_scores is None:
        raw_scores = [float("nan")] * len(texts)
    else:
        raw_scores = generated.sequences_scores.detach().float().cpu().tolist()
    finite = [value for value in raw_scores if math.isfinite(value)]
    if len(finite) == len(raw_scores):
        confidence = torch.softmax(torch.tensor(raw_scores), dim=0).tolist()
    else:
        confidence = [None] * len(raw_scores)
    deduplicated = []
    seen = set()
    for text, raw_score, relative in zip(texts, raw_scores, confidence):
        cleaned = text.strip()
        key = normalize(cleaned)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(
            {
                "text": cleaned,
                "sequence_score": raw_score if math.isfinite(raw_score) else None,
                "relative_beam_probability": relative,
            }
        )
    return deduplicated


def ctc_transcribe(model, processor, head, audio) -> str:
    inputs = processor.apply_transcription_request(audio=[audio], language=["English"])
    features = inputs["input_features"].to("cuda")
    mask = inputs["input_features_mask"].to("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        audio_output = model.model.get_audio_features(features, mask, return_dict=True)
        logits = head(audio_output.pooler_output)
    return greedy_decode(logits)


def oracle_score(reference: str, hypotheses: list[str]):
    return min((score(reference, text) for text in hypotheses), key=lambda item: (item.word_errors, item.character_errors))


def summarize(rows: list[dict]) -> dict:
    top1_scores, oracle_scores, ctc_scores = [], [], []
    literal_in_topk = 0
    for row in rows:
        hypotheses = json.loads(row["qwen_hypotheses"])
        texts = [item["text"] for item in hypotheses] or [""]
        top1_scores.append(score(row["reference"], texts[0]))
        oracle_scores.append(oracle_score(row["reference"], texts))
        literal_in_topk += any(normalize(text) == normalize(row["reference"]) for text in texts)
        if row.get("ctc_transcript", ""):
            ctc_scores.append(score(row["reference"], row["ctc_transcript"]))
    top1 = aggregate(top1_scores)
    oracle = aggregate(oracle_scores)
    result = {
        "utterances": len(rows),
        "qwen_top1_wer": top1.wer,
        "qwen_top1_cer": top1.cer,
        "qwen_topk_oracle_wer": oracle.wer,
        "qwen_topk_oracle_cer": oracle.cer,
        "literal_present_in_topk": literal_in_topk,
        "literal_present_in_topk_rate": literal_in_topk / len(rows),
    }
    if ctc_scores:
        ctc = aggregate(ctc_scores)
        result.update({"ctc_wer": ctc.wer, "ctc_cer": ctc.cer})
    return result


def run(
    config_path: Path,
    model_dir: Path,
    adapter_path: Path,
    manifest_path: Path,
    output: Path,
    beams: int,
    ctc_path: Path | None,
) -> dict:
    config = read_json(config_path)
    rows = read_jsonl(manifest_path)
    processor, model = load_model(config, model_dir)
    load_qwen_adapter(model, adapter_path)
    model.eval()
    head = None
    if ctc_path is not None:
        head = LiteralCTCHead().to("cuda")
        head.load_state_dict(load_file(ctc_path, device="cpu"))
        head.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "speaker",
        "slice",
        "audio_filepath",
        "reference",
        "qwen_top1",
        "qwen_hypotheses",
        "ctc_transcript",
        "decode_seconds",
    ]
    recorded = []
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            audio = load_audio(row)
            started = time.perf_counter()
            hypotheses = qwen_nbest(
                model, processor, audio, beams, int(config["generation_max_new_tokens"])
            )
            ctc_text = "" if head is None else ctc_transcribe(model, processor, head, audio)
            item = {
                "id": row["id"],
                "speaker": row["speaker"],
                "slice": row.get("slice", "unspecified"),
                "audio_filepath": row["audio_filepath"],
                "reference": row["text"],
                "qwen_top1": hypotheses[0]["text"] if hypotheses else "",
                "qwen_hypotheses": json.dumps(hypotheses, ensure_ascii=False),
                "ctc_transcript": ctc_text,
                "decode_seconds": f"{time.perf_counter() - started:.4f}",
            }
            writer.writerow(item)
            handle.flush()
            recorded.append(item)
            if index % 25 == 0 or index == len(rows):
                print(f"literal diagnostic {index}/{len(rows)}", flush=True)
    temporary.replace(output)

    by_slice: dict[str, list[dict]] = defaultdict(list)
    for row in recorded:
        by_slice[row["slice"]].append(row)
    summary = {
        "adapter": str(adapter_path),
        "ctc_head": None if ctc_path is None else str(ctc_path),
        "beams": beams,
        "overall": summarize(recorded),
        "slices": {name: summarize(values) for name, values in sorted(by_slice.items())},
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--ctc-head", type=Path)
    args = parser.parse_args()
    if args.beams < 2:
        parser.error("--beams must be at least two")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.config,
        arguments.model,
        arguments.adapter,
        arguments.manifest,
        arguments.output,
        arguments.beams,
        arguments.ctc_head,
    )
