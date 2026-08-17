import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_foundations import decide, read_rows, summarize
from make_foundation_gate import select_records


class FoundationGateTest(unittest.TestCase):
    def test_balanced_selection_is_deterministic_and_duration_stratified(self):
        records = [
            {
                "id": f"{speaker}-{index}",
                "speaker": speaker,
                "duration": float(index + 1),
            }
            for speaker in ("S1", "S2")
            for index in range(20)
        ]
        first = select_records(records, per_speaker=10, duration_bins=5, seed="fixed")
        second = select_records(records, per_speaker=10, duration_bins=5, seed="fixed")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual({row["speaker"] for row in first}, {"S1", "S2"})
        for speaker in ("S1", "S2"):
            durations = [row["duration"] for row in first if row["speaker"] == speaker]
            self.assertEqual(len(durations), 10)
            self.assertLessEqual(min(durations), 4)
            self.assertGreaterEqual(max(durations), 17)

    def test_decision_requires_wer_margin_and_deletion_guardrail(self):
        better = {
            "macro_speaker_wer": 0.40,
            "deletion_rate": 0.10,
            "empty_rate": 0.05,
        }
        worse = {
            "macro_speaker_wer": 0.50,
            "deletion_rate": 0.11,
            "empty_rate": 0.06,
        }
        result = decide(better, worse, ("better", "worse"), 0.05, 0.02)
        self.assertEqual(result["winner"], "better")

        deletion_heavy = dict(better, deletion_rate=0.20)
        result = decide(deletion_heavy, worse, ("better", "worse"), 0.05, 0.02)
        self.assertIsNone(result["winner"])

    def test_summary_counts_empty_transcripts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.tsv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["id", "speaker", "reference", "transcript", "rtf"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {"id": "a", "speaker": "S1", "reference": "want water", "transcript": "", "rtf": "1"}
                )
                writer.writerow(
                    {"id": "b", "speaker": "S2", "reference": "water", "transcript": "water", "rtf": "2"}
                )
            result = summarize(read_rows(path))
            self.assertEqual(result["empty_rate"], 0.5)
            self.assertEqual(result["median_rtf"], 1.5)


if __name__ == "__main__":
    unittest.main()
