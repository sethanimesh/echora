import json
from pathlib import Path

import numpy as np
import soundfile as sf

from research.training.stage_literal_diagnostic import literal_slice, stage


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_literal_slices():
    assert literal_slice("water") == "torgo_1word"
    assert literal_slice("I water") == "torgo_2_3word"
    assert literal_slice("I want some cold water") == "torgo_4plus"


def test_stage_combines_protected_slices(tmp_path):
    pilot = tmp_path / "pilot_data"
    manifest_dir = pilot / "manifests"
    common = {
        "speaker": "M04",
        "condition": "dysarthric",
        "duration": 1.0,
        "audio_filepath": "/workspace/echora/pilot_data/audio/torgo_test/x.wav",
    }
    write_jsonl(
        manifest_dir / "torgo_test.jsonl",
        [{**common, "id": "one", "text": "water"}, {**common, "id": "two", "text": "I water"}],
    )
    write_jsonl(
        manifest_dir / "personal.jsonl",
        [{**common, "id": "personal-one", "speaker": "external-personal", "text": "I water"}],
    )
    pause = tmp_path / "pause"
    for delay in ("0500ms", "1000ms", "2000ms"):
        directory = pause / delay
        directory.mkdir(parents=True)
        sf.write(directory / "clip.wav", np.zeros(1600, dtype=np.float32), 16000)
        (directory / "clip.txt").write_text("I water\n", encoding="utf-8")
    report = stage(pilot, pause, Path("/workspace/echora/pilot_data"))
    assert report["utterances"] == 6
    assert report["training_use"] is False
    rows = [json.loads(line) for line in (manifest_dir / "literal_diagnostic.jsonl").read_text().splitlines()]
    assert {row["slice"] for row in rows} == {
        "torgo_1word",
        "torgo_2_3word",
        "personal_original",
        "pause_0500ms",
        "pause_1000ms",
        "pause_2000ms",
    }
