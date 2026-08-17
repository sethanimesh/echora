import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from research.training.prepare_common_voice import choose_splits, collect


class CommonVoicePreparationTest(unittest.TestCase):
    def test_collect_and_speaker_disjoint_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clips = root / "clips"
            clips.mkdir()
            fields = ["client_id", "path", "sentence", "locale", "duration[ms]"]
            rows = []
            for speaker_index in range(6):
                for clip_index in range(2):
                    name = f"s{speaker_index}-{clip_index}.wav"
                    sf.write(clips / name, np.zeros(16_000, dtype=np.float32), 16_000)
                    rows.append(
                        {
                            "client_id": f"speaker-{speaker_index}",
                            "path": name,
                            "sentence": f"Sentence {speaker_index} {clip_index}",
                            "locale": "en",
                            "duration[ms]": "1000",
                        }
                    )
            with (root / "validated.tsv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            collected = collect(root, 0.5, 2.0, seed=7)
            selected = choose_splits(collected, 2.0, 2.0, 2.0, seed=7)
            speaker_sets = [{row["speaker"] for row in selected[name]} for name in ("train", "dev", "test")]
            self.assertFalse(speaker_sets[0] & speaker_sets[1])
            self.assertFalse(speaker_sets[0] & speaker_sets[2])
            self.assertFalse(speaker_sets[1] & speaker_sets[2])
            self.assertTrue(all(row["condition"] == "normal" for row in collected))


if __name__ == "__main__":
    unittest.main()
