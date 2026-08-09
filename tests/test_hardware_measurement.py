import unittest

from search.hardware_measurement import (
    bootstrap_median_ci,
    percentile,
    summarize_samples,
)


class HardwareMeasurementTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)

    def test_bootstrap_is_deterministic(self):
        first = bootstrap_median_ci([1.0, 2.0, 3.0, 4.0], resamples=200, seed=7)
        second = bootstrap_median_ci([1.0, 2.0, 3.0, 4.0], resamples=200, seed=7)
        self.assertEqual(first, second)

    def test_summary_preserves_distribution_statistics(self):
        summary = summarize_samples([1.0, 2.0, 3.0], bootstrap_resamples=200)
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["mean"], 2.0)
        self.assertGreaterEqual(summary["ci95_high"], summary["ci95_low"])

    def test_invalid_samples_fail(self):
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            summarize_samples([1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
