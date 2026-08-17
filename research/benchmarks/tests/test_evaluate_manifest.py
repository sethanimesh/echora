import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_manifest import completed_ids, summarize


class ManifestEvaluationTest(unittest.TestCase):
    def test_resume_ids_and_macro_speaker_wer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.tsv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["id", "speaker", "reference", "transcript"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow({"id": "a", "speaker": "S1", "reference": "water", "transcript": "water"})
                writer.writerow({"id": "b", "speaker": "S2", "reference": "want water", "transcript": "want"})
            self.assertEqual(completed_ids(path), {"a", "b"})
            result = summarize(path)
            self.assertAlmostEqual(result["micro_wer"], 1 / 3)
            self.assertAlmostEqual(result["macro_speaker_wer"], 0.25)


if __name__ == "__main__":
    unittest.main()
