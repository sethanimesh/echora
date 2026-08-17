import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from research.training.qwen_command_model import LoRALinear
from research.training.stage_qwen_command_v3 import stage


class CommandV3StageTest(unittest.TestCase):
    def test_command_splits_are_speaker_and_phrase_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root / "pilot_data"
            manifests = pilot / "manifests"
            audio_root = pilot / "audio"
            manifests.mkdir(parents=True)
            vocabulary = [f"word{letter}" for letter in "abcdefghijklmnopqrstuvwxy"]
            split_speakers = {
                "train": ["F01", "M01"],
                "dev": ["F03"],
                "test": ["M04"],
            }
            for split, speakers in split_speakers.items():
                rows = []
                split_audio = audio_root / f"torgo_{split}"
                split_audio.mkdir(parents=True)
                for speaker in speakers:
                    for index, word in enumerate(vocabulary):
                        identifier = f"{split}-{speaker}-{index}"
                        path = split_audio / f"{identifier}.wav"
                        seconds = 0.28 + index * 0.001
                        samples = np.arange(round(16_000 * seconds))
                        waveform = (0.08 * np.sin(2 * np.pi * (180 + index) * samples / 16_000)).astype(np.float32)
                        sf.write(path, waveform, 16_000)
                        rows.append(
                            {
                                "id": identifier,
                                "speaker": speaker,
                                "condition": "dysarthric",
                                "duration": seconds,
                                "text": word,
                                "audio_filepath": f"/workspace/echora/pilot_data/audio/torgo_{split}/{path.name}",
                            }
                        )
                with (manifests / f"torgo_{split}.jsonl").open("w") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
            # The command diagnostic builder appends to this protected manifest.
            diagnostic_audio = audio_root / "literal"
            diagnostic_audio.mkdir()
            path = diagnostic_audio / "personal.wav"
            sf.write(path, np.zeros(3200, dtype=np.float32), 16_000)
            (manifests / "literal_diagnostic.jsonl").write_text(
                json.dumps(
                    {
                        "id": "personal",
                        "speaker": "external",
                        "condition": "external_smoke",
                        "duration": 0.2,
                        "text": "I water",
                        "audio_filepath": "/workspace/echora/pilot_data/audio/literal/personal.wav",
                        "slice": "personal_original",
                    }
                )
                + "\n"
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "run_id": "test",
                        "development_speaker": "F03",
                        "outer_test_speaker": "M04",
                        "maximum_audio_seconds": 10,
                        "seed": 7,
                        "command_train_pool_size": 12,
                        "command_dev_size": 8,
                        "command_test_size": 8,
                    }
                )
            )
            report = stage(
                config,
                pilot,
                Path("/workspace/echora/pilot_data"),
            )
            self.assertTrue(report["speaker_disjoint"])
            self.assertTrue(report["phrase_disjoint"])
            phrases = []
            for split in ("train", "dev", "test"):
                rows = [
                    json.loads(line)
                    for line in (manifests / f"command_{split}.jsonl").read_text().splitlines()
                ]
                phrases.append({row["text"] for row in rows})
                self.assertEqual(len(rows), {"train": 12, "dev": 8, "test": 8}[split])
                self.assertTrue(all(row["synthetic_composition"] for row in rows))
            self.assertFalse(phrases[0] & phrases[1])
            self.assertFalse(phrases[0] & phrases[2])
            self.assertFalse(phrases[1] & phrases[2])


class LoRALinearTest(unittest.TestCase):
    def test_zero_initialized_residual_preserves_foundation_output(self):
        torch.manual_seed(3)
        base = torch.nn.Linear(5, 4, bias=False)
        values = torch.randn(2, 3, 5)
        expected = base(values).detach()
        layer = LoRALinear(base, rank=2, alpha=4, dropout=0)
        actual = layer(values)
        self.assertTrue(torch.equal(expected, actual.detach()))
        self.assertFalse(layer.base.weight.requires_grad)
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)


if __name__ == "__main__":
    unittest.main()
