import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import SAMPLE_RATE, internal_gap_stats, prediction_from


class CommonTest(unittest.TestCase):
    def test_prediction_coercion_keeps_raw_order(self):
        result = prediction_from(["water", "quarter"])
        self.assertEqual([item.text for item in result.hypotheses], ["water", "quarter"])

    def test_internal_gap_is_measured_but_not_removed(self):
        tone = np.ones(round(SAMPLE_RATE * 0.5), dtype=np.float32) * 0.1
        audio = np.concatenate((tone, np.zeros(round(SAMPLE_RATE * 0.4)), tone))
        total, maximum = internal_gap_stats(audio)
        self.assertAlmostEqual(total, 0.4, places=2)
        self.assertAlmostEqual(maximum, 0.4, places=2)


if __name__ == "__main__":
    unittest.main()
