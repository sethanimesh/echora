"""Run a command-v3 adapter and return literal top-k ASR alternatives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from safetensors import safe_open

try:
    from .qwen_command_model import load_model, load_safetensors_adapter
    from .qwen_command_v3 import load_audio, nbest, read_json
except ImportError:
    from qwen_command_model import load_model, load_safetensors_adapter
    from qwen_command_v3 import load_audio, nbest, read_json


def sidecar_reference(path: Path) -> str:
    sidecar = path.with_suffix(".txt")
    return sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else ""


def run(config_path: Path, model_dir: Path, adapter: Path, audio: Path, output: Path, beams: int) -> dict:
    config = read_json(config_path)
    with safe_open(adapter, framework="pt") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != "echora-qwen3-asr-command-v3":
        raise RuntimeError("Adapter is not an Echora command-v3 artifact")
    processor, model, _ = load_model(config, model_dir)
    load_safetensors_adapter(model, adapter, exact=True)
    model.eval()
    paths = sorted(audio.glob("*.wav")) if audio.is_dir() else [audio]
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"No WAV audio found at {audio}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = ["audio", "reference", "top1", "alternatives"]
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for path in paths:
            row = {"audio_filepath": str(path), "id": path.stem}
            hypotheses = nbest(
                model,
                processor,
                load_audio(row),
                str(config["literal_prompt"]),
                beams,
                int(config["generation_max_new_tokens"]),
            )
            texts = [item["text"] for item in hypotheses]
            writer.writerow(
                {
                    "audio": str(path),
                    "reference": sidecar_reference(path),
                    "top1": texts[0] if texts else "",
                    "alternatives": json.dumps(hypotheses, ensure_ascii=False),
                }
            )
            print(f"{path.name}: {texts}", flush=True)
    temporary.replace(output)
    result = {
        "utterances": len(paths),
        "beams": beams,
        "output": str(output),
        "relative_beam_probabilities_are_calibrated": False,
        "semantic_repair": False,
        "candidate_ranker": False,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beams", type=int, default=5)
    args = parser.parse_args()
    if args.beams < 2:
        parser.error("--beams must be at least two")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config, arguments.model, arguments.adapter, arguments.audio, arguments.output, arguments.beams)
