import unittest

from research.training.qwen_torgo_pilot import (
    development_guard,
    epoch_schedule,
    score_predictions,
    select_coverage_slice,
    select_speaker_balanced,
)


def row(identifier, speaker, condition, text="hello world"):
    return {
        "id": identifier,
        "speaker": speaker,
        "condition": condition,
        "duration": 1.0,
        "text": text,
        "audio_filepath": "/unused",
    }


class QwenPilotTest(unittest.TestCase):
    def test_epoch_contains_every_dysarthric_row_once(self):
        manifests = {
            "torgo_train": [
                row("d1", "F01", "dysarthric"),
                row("d2", "M01", "dysarthric"),
                row("c1", "FC01", "control"),
                row("c2", "MC01", "control"),
            ],
            "normal_train": [row("n1", "cv1", "normal"), row("n2", "cv2", "normal")],
        }
        config = {
            "seed": 3,
            "torgo_control_examples_per_epoch": 2,
            "normal_examples_per_epoch": 2,
        }
        scheduled = epoch_schedule(manifests, config, 1)
        dys_ids = [item["id"] for item in scheduled if item["condition"] == "dysarthric"]
        self.assertCountEqual(dys_ids, ["d1", "d2"])
        self.assertEqual(len(scheduled), 6)

    def test_speaker_balancing_is_deterministic(self):
        values = [row("a1", "a", "normal"), row("a2", "a", "normal"), row("b1", "b", "normal")]
        first = select_speaker_balanced(values, 4, 9)
        second = select_speaker_balanced(values, 4, 9)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual({item["speaker"] for item in first[:2]}, {"a", "b"})

    def test_coverage_slice_sees_pool_before_repeating(self):
        values = [row(f"n{i}", f"s{i % 2}", "normal") for i in range(5)]
        seen = select_coverage_slice(values, 3, 1, 5) + select_coverage_slice(values, 3, 2, 5)
        self.assertEqual(len({item["id"] for item in seen[:5]}), 5)

    def test_metrics_include_deletions_and_empty_rate(self):
        metrics = score_predictions(
            [row("a", "speaker", "dysarthric", "I want water"), row("b", "speaker", "dysarthric", "please")],
            ["I water", ""],
        )
        self.assertEqual(metrics["word_deletions"], 2)
        self.assertEqual(metrics["empty_outputs"], 1)

    def test_retention_guard_rejects_normal_regression(self):
        baseline = {
            "normal_dev": {"micro_wer": 0.10},
            "torgo_dev": {"deletion_rate": 0.10, "empty_rate": 0.0},
        }
        candidate = {
            "normal_dev": {"micro_wer": 0.13},
            "torgo_dev": {"deletion_rate": 0.10, "empty_rate": 0.0},
        }
        config = {
            "maximum_normal_dev_wer_degradation_absolute": 0.02,
            "maximum_dev_deletion_rate_increase_absolute": 0.02,
            "maximum_dev_empty_rate_increase_absolute": 0.02,
        }
        passed, reasons = development_guard(candidate, baseline, config)
        self.assertFalse(passed)
        self.assertIn("normal-dev WER guard", reasons)


if __name__ == "__main__":
    unittest.main()
