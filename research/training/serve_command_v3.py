"""Persistent literal-ASR inference server for interactive command-v3 testing.

Loads the foundation once, then watches an inbox directory for WAV files and
writes one JSON result per clip to an outbox directory.  Both the verified v1
baseline adapter and the command-v3 adapter can be evaluated on the same clip
so that alternatives are directly comparable.

This is a diagnostic tool.  It uses the frozen literal prompt, the frozen
decoding helpers and the frozen adapter loader, so its outputs match the
training run's protected diagnostics.  Beam probabilities remain relative
search scores, not calibrated confidence, and nothing here re-ranks, repairs
or applies personal context.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

try:
    from .qwen_command_model import (
        LoRALinear,
        load_model,
        load_partial_state,
        load_safetensors_adapter,
        partial_state,
    )
    from .qwen_command_v3 import load_audio, nbest, read_json, transcribe
except ImportError:
    from qwen_command_model import (
        LoRALinear,
        load_model,
        load_partial_state,
        load_safetensors_adapter,
        partial_state,
    )
    from qwen_command_v3 import load_audio, nbest, read_json, transcribe


COMMAND_V3_FORMAT = "echora-qwen3-asr-command-v3"
# The untouched foundation, restored from the weights captured before any
# adapter is applied.  Its LoRA residual is identity because lora_B starts zero.
BASE = "base"


def adapter_format(path: Path) -> str:
    with safe_open(path, framework="pt") as handle:
        return (handle.metadata() or {}).get("format", "")


def zero_lora(model) -> None:
    """Restore the identity LoRA residual used by the v1 baseline evaluation."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.zero_()


def audio_quality(audio: np.ndarray, sample_rate: int = 16_000) -> dict:
    """Reproducible level descriptors, not clinical voice-activity analysis."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    frame = round(sample_rate * 0.02)
    usable = len(audio) - (len(audio) % frame) if frame else 0
    estimated = None
    if usable >= frame * 5:
        frames = audio[:usable].reshape(-1, frame)
        energy = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)) + 1e-12
        decibels = 20 * np.log10(energy)
        high, low = np.percentile(decibels, 90), np.percentile(decibels, 10)
        speech, noise = decibels > (high - 15), decibels < (low + 6)
        if speech.any() and noise.any():
            estimated = float(
                20 * np.log10(energy[speech].mean()) - 20 * np.log10(energy[noise].mean())
            )
    def dbfs(value: float) -> float | None:
        return round(float(20 * np.log10(value)), 2) if value > 0 else None
    return {
        "seconds": round(len(audio) / sample_rate, 3),
        "peak_dbfs": dbfs(peak),
        "rms_dbfs": dbfs(rms),
        "estimated_snr_db": round(estimated, 1) if estimated is not None else None,
        "clipped_samples": int(np.sum(np.abs(audio) > 0.99)),
        "low_level_warning": bool(peak > 0 and 20 * np.log10(peak) < -30.0),
    }


class Engine:
    """Holds one loaded foundation and swaps adapters on demand."""

    def __init__(self, config: dict, model_dir: Path, adapters: dict[str, Path]) -> None:
        self.config = config
        self.prompt = str(config["literal_prompt"])
        self.max_tokens = int(config["generation_max_new_tokens"])
        self.adapters = adapters
        print(f"loading foundation from {model_dir} ...", flush=True)
        self.processor, self.model, _ = self.model_bundle(model_dir)
        self.model.eval()
        # Snapshot the untouched foundation weights before any adapter is
        # applied, so "base" is a true no-fine-tuning comparison rather than a
        # partially overwritten model.
        self.base_state = partial_state(self.model)
        self.current: str | None = None
        print("foundation ready", flush=True)

    def model_bundle(self, model_dir: Path):
        return load_model(self.config, model_dir)

    def names(self) -> list[str]:
        return [BASE] + sorted(self.adapters)

    def select(self, name: str) -> None:
        if self.current == name:
            return
        if name == BASE:
            load_partial_state(self.model, self.base_state, exact=True)
            self.current = name
            print("adapter active: base (no fine-tuning)", flush=True)
            return
        path = self.adapters[name]
        is_command_v3 = adapter_format(path) == COMMAND_V3_FORMAT
        # A v1 adapter carries no decoder-LoRA tensors, so the residual must be
        # returned to identity before its audio/projector weights are applied.
        zero_lora(self.model)
        load_safetensors_adapter(self.model, path, exact=is_command_v3)
        self.current = name
        print(f"adapter active: {name}", flush=True)

    def run(self, audio: np.ndarray, name: str, beams: int) -> dict:
        self.select(name)
        started = time.perf_counter()
        greedy = transcribe(self.model, self.processor, audio, self.prompt, self.max_tokens)
        hypotheses = nbest(
            self.model, self.processor, audio, self.prompt, beams, self.max_tokens
        )
        return {
            "adapter": name,
            "fine_tuned": name != BASE,
            "adapter_file": "" if name == BASE else str(self.adapters[name]),
            "greedy_top1": greedy,
            "beam_top1": hypotheses[0]["text"] if hypotheses else "",
            "alternatives": hypotheses,
            "decode_seconds": round(time.perf_counter() - started, 2),
        }


def settled(path: Path, checks: int = 2, pause: float = 0.20) -> bool:
    """Reject a file that is still growing, so partial uploads are never read."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return False
    for _ in range(checks):
        time.sleep(pause)
        try:
            current = path.stat().st_size
        except FileNotFoundError:
            return False
        if current != size or current == 0:
            return False
        size = current
    return True


def serve(
    config_path: Path,
    model_dir: Path,
    v3_adapter: Path,
    v1_adapter: Path | None,
    inbox: Path,
    outbox: Path,
    beams: int,
    poll_seconds: float,
) -> None:
    config = read_json(config_path)
    adapters = {"v3": v3_adapter}
    if v1_adapter is not None:
        adapters["v1"] = v1_adapter
    for name, path in adapters.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} adapter is missing: {path}")

    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    processed = inbox / "processed"
    processed.mkdir(exist_ok=True)

    engine = Engine(config, model_dir, adapters)
    ready = outbox / "SERVER_READY"
    ready.write_text(
        json.dumps(
            {
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "adapters": engine.names(),
                "base_means": "untouched Qwen3-ASR foundation, no fine-tuning",
                "default_beams": beams,
                "literal_prompt": engine.prompt,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"watching {inbox} (adapters: {', '.join(engine.names())})", flush=True)

    while True:
        wavs = sorted(p for p in inbox.glob("*.wav") if p.is_file())
        if not wavs:
            time.sleep(poll_seconds)
            continue
        for wav in wavs:
            if not settled(wav):
                continue
            request, target = {}, outbox / f"{wav.stem}.json"
            sidecar = wav.with_suffix(".json")
            if sidecar.is_file():
                try:
                    request = json.loads(sidecar.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    request = {}
            available = engine.names()
            wanted = [
                name for name in request.get("adapters", available) if name in available
            ] or available
            use_beams = max(2, int(request.get("beams", beams)))
            reference = str(request.get("reference", "")).strip()
            print(f"--- {wav.name}: adapters={wanted} beams={use_beams}", flush=True)
            payload: dict = {
                "clip": wav.name,
                "label": request.get("label", wav.stem),
                "reference": reference,
                "received": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "beams": use_beams,
                "literal_prompt": engine.prompt,
                "relative_beam_probabilities_are_calibrated": False,
                "semantic_repair": False,
                "candidate_ranker": False,
            }
            try:
                audio = load_audio({"audio_filepath": str(wav), "id": wav.stem})
                payload["audio_quality"] = audio_quality(audio)
                payload["results"] = {
                    name: engine.run(audio, name, use_beams) for name in wanted
                }
                for name in wanted:
                    print(
                        f"    {name}: {[item['text'] for item in payload['results'][name]['alternatives']]}",
                        flush=True,
                    )
            except Exception as error:  # keep serving after one bad clip
                payload["error"] = f"{type(error).__name__}: {error}"
                payload["traceback"] = traceback.format_exc()
                print(f"    FAILED {wav.name}: {error}", flush=True)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            temporary.replace(target)
            wav.replace(processed / wav.name)
            if sidecar.is_file():
                sidecar.replace(processed / sidecar.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True, help="command-v3 adapter")
    parser.add_argument("--v1-adapter", type=Path, default=None, help="verified v1 baseline")
    parser.add_argument("--inbox", type=Path, default=Path("/workspace/echora/serve/inbox"))
    parser.add_argument("--outbox", type=Path, default=Path("/workspace/echora/serve/outbox"))
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if args.beams < 2:
        parser.error("--beams must be at least two")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    serve(
        arguments.config,
        arguments.model,
        arguments.adapter,
        arguments.v1_adapter,
        arguments.inbox,
        arguments.outbox,
        arguments.beams,
        arguments.poll_seconds,
    )
