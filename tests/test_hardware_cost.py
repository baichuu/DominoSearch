import json
import tempfile
import unittest
from pathlib import Path

from search.hardware_cost import HardwareCostModel, fit_cost_predictor, validate_weights


def make_layer(name, scale):
    features = {
        "batch_size": 1,
        "in_channels": 16 * scale,
        "out_channels": 32 * scale,
        "input_elements": 1000 * scale,
        "output_elements": 500 * scale,
        "kernel_elements": 9,
        "groups": 1,
        "dense_parameters": 4608 * scale,
        "dense_macs": 100000 * scale,
    }
    candidates = []
    for n, latency in ((4, 0.6), (8, 0.8), (16, 1.0)):
        density = n / 16
        candidates.append(
            {
                "n": n,
                "m": 16,
                "metrics": {
                    "latency_ms": latency * scale,
                    "energy_mj": None,
                    "bandwidth_bytes": 1000 * density * scale,
                    "memory_bytes": 500 * density * scale,
                },
            }
        )
    return {"name": name, "type": "conv2d", "features": features, "candidates": candidates}


def make_profile():
    profile = {
        "schema_version": 1,
        "hardware": {"name": "synthetic-device"},
        "model": {"name": "tiny", "layout": "NCHW"},
        "cost_definition": {
            "weights": {
                "latency_ms": 0.5,
                "energy_mj": 0.0,
                "bandwidth_bytes": 0.25,
                "memory_bytes": 0.25,
            }
        },
        "layers": [make_layer("layer_a", 1), make_layer("layer_b", 2)],
    }
    profile["predictor"] = fit_cost_predictor(profile)
    return profile


class HardwareCostTest(unittest.TestCase):
    def write_profile(self, profile):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def test_lookup_cost_and_reduction_are_normalized_to_dense(self):
        model = HardwareCostModel(self.write_profile(make_profile()), mode="lookup")
        dense = {"layer_a": [16, 16], "layer_b": [16, 16]}
        sparse = {"layer_a": [8, 16], "layer_b": [8, 16]}
        self.assertAlmostEqual(model.total_cost(dense), 1.0)
        self.assertAlmostEqual(model.reduction(dense), 0.0)
        self.assertAlmostEqual(model.reduction(sparse), 0.35)
        self.assertEqual(
            model.best_independent_scheme(),
            {"layer_a": [4, 16], "layer_b": [4, 16]},
        )
        self.assertAlmostEqual(model.maximum_independent_reduction(), 0.575)
        normalization = model.normalization(dense)
        self.assertAlmostEqual(normalization["layer_a"], 0.5)
        self.assertAlmostEqual(normalization["layer_b"], 1.0)

    def test_predictor_produces_finite_nonnegative_cost(self):
        profile = make_profile()
        self.assertEqual(
            profile["predictor"]["validation"]["latency_ms"]["method"],
            "leave-one-layer-out",
        )
        model = HardwareCostModel(self.write_profile(profile), mode="predictor")
        cost = model.cost("layer_a", 6, 16)
        self.assertGreaterEqual(cost, 0.0)

    def test_profile_must_exactly_match_model_layers(self):
        model = HardwareCostModel(self.write_profile(make_profile()), mode="lookup")
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            model.validate_model_layers(["layer_a"], 16)

    def test_lookup_rejects_unmeasured_configuration(self):
        model = HardwareCostModel(self.write_profile(make_profile()), mode="lookup")
        with self.assertRaisesRegex(ValueError, "No measured hardware cost"):
            model.cost("layer_a", 2, 16)

    def test_positive_energy_weight_requires_measurements(self):
        profile = make_profile()
        profile["cost_definition"]["weights"] = {
            "latency_ms": 0.5,
            "energy_mj": 0.5,
            "bandwidth_bytes": 0.0,
            "memory_bytes": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "no additive dense reference"):
            HardwareCostModel(self.write_profile(profile), mode="lookup")

    def test_invalid_weights_fail(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            validate_weights(
                {
                    "latency_ms": 1.0,
                    "energy_mj": 0.0,
                    "bandwidth_bytes": 0.5,
                    "memory_bytes": 0.0,
                }
            )


if __name__ == "__main__":
    unittest.main()
