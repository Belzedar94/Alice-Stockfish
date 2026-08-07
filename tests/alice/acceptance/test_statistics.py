from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_acceptance.statistics import paired_statistics


class PairedStatisticsTests(unittest.TestCase):
    def test_empty_and_degenerate_samples(self) -> None:
        self.assertEqual(paired_statistics([0, 0, 0, 0, 0])["los_percent_display"], "50.0")
        self.assertEqual(paired_statistics([0, 0, 7, 0, 0])["los_percent_display"], "50.0")
        self.assertEqual(paired_statistics([7, 0, 0, 0, 0])["los_percent_display"], "0.0")
        self.assertEqual(paired_statistics([0, 0, 0, 0, 7])["los_percent_display"], "100.0")

    def test_paired_counterexample_does_not_use_decisive_only_los(self) -> None:
        summary = paired_statistics([20, 6, 15, 5, 5])
        self.assertEqual(summary["pair_count"], 51)
        self.assertEqual(summary["los_percent_display"], "0.1")
        self.assertGreater(float(summary["los_probability"]), 0.0005)

    def test_binary64_identity_and_elo_are_recomputable(self) -> None:
        summary = paired_statistics([2, 3, 5, 7, 11])
        probability = float.fromhex(str(summary["los_probability_binary64_hex"]))
        self.assertEqual(probability, summary["los_probability"])
        self.assertTrue(math.isfinite(probability))
        elo = summary["elo_95"]
        self.assertIsInstance(elo, dict)
        self.assertLess(elo["lower"], elo["estimate"])
        self.assertLess(elo["estimate"], elo["upper"])

    def test_invalid_counts_are_rejected(self) -> None:
        for counts in ([1, 2], [0, 0, 0, 0, -1], [0, 0, 0, 0, 1.0]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                paired_statistics(counts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
