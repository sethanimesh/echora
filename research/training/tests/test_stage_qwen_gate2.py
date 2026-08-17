import unittest

from research.training.stage_qwen_gate2 import eligible_rows, select_probe_rows


class QwenGate2StagingTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "outer_test_speaker": "M04",
            "development_speaker": "F03",
            "training_condition": "dysarthric",
            "maximum_probe_audio_seconds": 30.0,
            "tiny_subset_size": 2,
            "tiny_subset_target_seconds": 5.0,
        }

    def test_eligibility_excludes_held_out_control_and_overlength_rows(self):
        rows = [
            self.row("train-a", "F01", 5.0),
            self.row("test", "M04", 5.0),
            self.row("dev", "F03", 5.0),
            self.row("control", "F01", 5.0, condition="control"),
            self.row("too-long", "F01", 31.0),
        ]
        self.assertEqual([item["id"] for item in eligible_rows(rows, self.config)], ["train-a"])

    def test_selection_is_balanced_deterministic_and_keeps_longest_separate(self):
        rows = [
            self.row("a-near", "F01", 4.9),
            self.row("a-far", "F01", 2.0),
            self.row("b-near", "M01", 5.1),
            self.row("b-far", "M01", 7.0),
            self.row("longest", "M01", 20.0),
        ]
        longest, tiny = select_probe_rows(rows, self.config)
        self.assertEqual(longest["id"], "longest")
        self.assertEqual([item["id"] for item in tiny], ["a-near", "b-near"])
        self.assertEqual({item["speaker"] for item in tiny}, {"F01", "M01"})

    @staticmethod
    def row(identifier, speaker, duration, condition="dysarthric"):
        return {
            "id": identifier,
            "speaker": speaker,
            "duration": duration,
            "condition": condition,
            "text": identifier,
            "audio_filepath": "/unused.wav",
        }


if __name__ == "__main__":
    unittest.main()
