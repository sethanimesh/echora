import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from augment import add_noise_at_snr, insert_pause_at_low_energy, speed_perturb


class AugmentTest(unittest.TestCase):
    def test_speed_changes_length_without_nan(self):
        audio = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
        changed = speed_perturb(audio, 1.2)
        self.assertEqual(len(changed), 120)
        self.assertTrue(np.isfinite(changed).all())

    def test_pause_only_increases_length(self):
        audio = np.ones(16000, dtype=np.float32) * 0.1
        audio[7000:7500] = 0.0
        changed = insert_pause_at_low_energy(audio, 16000, 0.5, np.random.default_rng(1))
        self.assertEqual(len(changed), len(audio) + 8000)

    def test_silent_audio_stays_silent_under_noise_transform(self):
        audio = np.zeros(100, dtype=np.float32)
        changed = add_noise_at_snr(audio, 20, np.random.default_rng(1))
        self.assertTrue(np.array_equal(audio, changed))


if __name__ == "__main__":
    unittest.main()
