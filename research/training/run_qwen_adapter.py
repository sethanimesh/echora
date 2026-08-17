"""Transcribe audio with the Qwen foundation plus an Echora partial adapter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "research" / "benchmarks"))
from metrics import aggregate, score  # noqa: E402

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_adapter(model, adapter: Path, model_dir: Path) -> dict:
    with safe_open(adapter, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected_weight = model_dir / "model.safetensors"
    if metadata.get("format") != "echora-qwen3-asr-partial-v1":
        raise RuntimeError("Unknown Echora adapter format")
    if metadata.get("base_model_sha256") != sha256(expected_weight):
        raise RuntimeError("Adapter foundation SHA-256 does not match the staged Qwen model")
    state = load_file(adapter, device="cpu")
    model_parameters = dict(model.named_parameters())
    missing = set(state) - set(model_parameters)
    if missing:
        raise RuntimeError(f"Adapter has parameters absent from Qwen: {sorted(missing)[:5]}")
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = model_parameters[name]
            if parameter.shape != tensor.shape:
                raise RuntimeError(f"Adapter shape mismatch: {name}")
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))
    return metadata


def transcribe(model, processor, audio: np.ndarray, max_new_tokens: int) -> str:
    inputs = processor.apply_transcription_request(
        audio=[audio], language=["English"]
    ).to(model.device, model.dtype)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    return processor.decode(generated, return_format="transcription_only")[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True, help="one audio file or a directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required; this script does not silently change devices")
    if args.audio.is_file():
        files = [args.audio]
    elif args.audio.is_dir():
        files = sorted(path for path in args.audio.iterdir() if path.suffix.lower() in AUDIO_EXTENSIONS)
    else:
        parser.error(f"Audio path does not exist: {args.audio}")
    if not files:
        parser.error("No supported audio files were found")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    metadata = load_adapter(model, args.adapter, args.model)
    print(f"adapter {metadata.get('run_id')} epoch={metadata.get('selected_epoch')}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, scores = [], []
    for path in files:
        audio, _ = librosa.load(path, sr=16_000, mono=True, dtype=np.float32)
        started = time.perf_counter()
        transcript = transcribe(model, processor, np.asarray(audio, dtype=np.float32), args.max_new_tokens)
        elapsed = time.perf_counter() - started
        sidecar = path.with_suffix(".txt")
        reference = sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else ""
        if reference:
            scores.append(score(reference, transcript))
        print(f"{path.name}: {transcript or '(empty)'}", flush=True)
        rows.append(
            {
                "audio": str(path),
                "audio_seconds": f"{len(audio) / 16000:.4f}",
                "decode_seconds": f"{elapsed:.4f}",
                "reference": reference,
                "transcript": transcript,
            }
        )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "utterances": len(rows),
        "referenced_utterances": len(scores),
        "output": str(args.output),
    }
    if scores:
        total = aggregate(scores)
        result.update({"wer": total.wer, "cer": total.cer})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
