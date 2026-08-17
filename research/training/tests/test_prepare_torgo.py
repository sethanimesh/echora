import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_torgo import clean_prompt, collect, make_folds
from finetune_parakeet_ctc import audit_split, balanced_weights


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


class PrepareTorgoTest(unittest.TestCase):
    def test_prompt_cleaning(self):
        self.assertEqual(clean_prompt('tear [as in "tear in my eye"]'), ("tear", None))
        self.assertEqual(clean_prompt("input/images/kitchen.jpg")[1], "spontaneous_audio_without_transcript")
        self.assertEqual(clean_prompt("[say Ah for 5 seconds]")[1], "non_asr_vocal_instruction")

    def test_collect_uses_only_selected_mic_and_folds_are_speaker_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw"
            output = Path(directory) / "out"
            for group, speaker in (("F", "F01"), ("F", "F02"), ("FC", "FC01")):
                session = source / group / speaker / "Session1"
                (session / "prompts").mkdir(parents=True)
                (session / "prompts" / "0001.txt").write_text("water")
                write_wav(session / "wav_headMic" / "0001.wav")
                write_wav(session / "wav_arrayMic" / "0001.wav")
            records, excluded = collect(source, "head")
            self.assertEqual(len(records), 3)
            self.assertFalse(excluded)
            summaries = make_folds(records, output)
            self.assertEqual(len(summaries), 2)
            self.assertNotEqual(summaries[0]["test_speaker"], summaries[0]["dev_speaker"])
            # The shared prompt must be absent from strict training.
            self.assertEqual(summaries[0]["train_no_test_text"], 0)

    def test_balancing_targets_condition_and_equal_speaker_mass(self):
        records = [
            {"speaker": "D1", "condition": "dysarthric"},
            {"speaker": "D1", "condition": "dysarthric"},
            {"speaker": "D2", "condition": "dysarthric"},
            {"speaker": "C1", "condition": "control"},
        ]
        weights = balanced_weights(records, 0.8)
        self.assertAlmostEqual(weights[0] + weights[1] + weights[2], 0.8)
        self.assertAlmostEqual(weights[3], 0.2)
        self.assertAlmostEqual(weights[0] + weights[1], weights[2])

    def test_split_audit_rejects_speaker_overlap(self):
        train = [{"id": "a", "speaker": "D1", "condition": "dysarthric", "text_id": "x"}]
        dev = [{"id": "b", "speaker": "D1", "condition": "dysarthric", "text_id": "y"}]
        with self.assertRaisesRegex(ValueError, "speaker leakage"):
            audit_split(train, dev)


if __name__ == "__main__":
    unittest.main()
