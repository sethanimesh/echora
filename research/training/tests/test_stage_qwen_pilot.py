import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from research.training.stage_qwen_pilot import audit_splits, copy_split, eligible


def row(identifier, speaker, condition, text_id=None, duration=1.0):
    value = {
        "id": identifier,
        "speaker": speaker,
        "condition": condition,
        "duration": duration,
        "text": identifier,
        "audio_filepath": "/unused",
    }
    if text_id is not None:
        value["text_id"] = text_id
    return value


class QwenPilotStagingTest(unittest.TestCase):
    def setUp(self):
        self.config = {"development_speaker": "F03", "outer_test_speaker": "M04"}
        self.train = [
            row("dys", "F04", "dysarthric", "train-text"),
            row("control", "FC01", "control", "control-text"),
        ]
        self.dev = [row("dev", "F03", "dysarthric", "dev-text")]
        self.test = [row("test", "M04", "dysarthric", "test-text")]
        self.normal_train = [row("nt", "cv-a", "normal")]
        self.normal_dev = [row("nd", "cv-b", "normal")]
        self.normal_test = [row("ne", "cv-c", "normal")]

    def test_clean_splits_pass(self):
        report = audit_splits(
            self.train,
            self.dev,
            self.test,
            self.normal_train,
            self.normal_dev,
            self.normal_test,
            self.config,
        )
        self.assertEqual(report["outer_test_prompt_overlap"], {"train": 0, "dev": 0})

    def test_test_prompt_leak_is_rejected(self):
        self.train[0]["text_id"] = "test-text"
        with self.assertRaisesRegex(ValueError, "outer-test prompts"):
            audit_splits(
                self.train,
                self.dev,
                self.test,
                self.normal_train,
                self.normal_dev,
                self.normal_test,
                self.config,
            )

    def test_explicit_demo_condition_discloses_prompt_overlap(self):
        self.train[0]["text_id"] = "test-text"
        config = {**self.config, "allow_outer_test_prompt_overlap": True}
        report = audit_splits(
            self.train,
            self.dev,
            self.test,
            self.normal_train,
            self.normal_dev,
            self.normal_test,
            config,
        )
        self.assertEqual(report["outer_test_prompt_overlap"]["train"], 1)
        self.assertTrue(report["prompt_overlap_allowed_and_disclosed"])

    def test_duration_filter_reports_exclusion(self):
        kept, excluded = eligible([row("a", "s", "normal", duration=1), row("b", "s", "normal", duration=31)], 30)
        self.assertEqual([item["id"] for item in kept], ["a"])
        self.assertEqual([item["id"] for item in excluded], ["b"])

    def test_personal_staging_keeps_reference_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.zeros(1600, dtype=np.float32), 16_000)
            item = row("personal-a", "speaker", "external_smoke")
            item["audio_filepath"] = str(source)
            item["text"] = "I want water"
            staged = copy_split([item], root / "stage", Path("/cloud"), "personal")
            sidecar = root / "stage" / "audio" / "personal" / "personal-a.txt"
            self.assertEqual(sidecar.read_text().strip(), "I want water")
            self.assertEqual(staged[0]["audio_filepath"], "/cloud/audio/personal/personal-a.wav")


if __name__ == "__main__":
    unittest.main()
