import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import edit_counts, normalize, score, score_file


class MetricsTest(unittest.TestCase):
    def test_normalization_ignores_case_and_punctuation(self):
        self.assertEqual(normalize("I want water, please."), "i want water please")

    def test_edit_types(self):
        result = score("I want water", "I water now")
        self.assertEqual(result.reference_words, 3)
        self.assertEqual(result.word_errors, 2)
        self.assertEqual(edit_counts("abc", "adc"), (1, 0, 0))

    def test_top_k_oracle_does_not_replace_top1_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.tsv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["clip", "reference", "transcript", "alternatives"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "clip": "one.wav",
                        "reference": "I want water",
                        "transcript": "I want quarter",
                        "alternatives": json.dumps(["I want water"]),
                    }
                )
            result = score_file(path)
            self.assertEqual(result.top1.word_errors, 1)
            self.assertEqual(result.oracle.word_errors, 0)


if __name__ == "__main__":
    unittest.main()
