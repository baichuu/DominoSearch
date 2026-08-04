import json
import tempfile
import unittest
from pathlib import Path

from search.hardware_cost import HardwareCostModel
from search.layer_sensitivity import LayerSensitivityProfile
from search.select_sensitivity_aware_scheme import choose_scheme


def hardware_layer(name, layer_type, dense_macs, sparse_latency):
    return {
        "name": name,
        "type": layer_type,
        "features": {
            "batch_size": 1,
            "in_channels": 16,
            "out_channels": 16,
            "input_elements": 100,
            "output_elements": 100,
            "kernel_elements": 1,
            "groups": 1,
            "dense_parameters": 256,
            "dense_macs": dense_macs,
        },
        "candidates": [
            {
                "n": 2,
                "m": 4,
                "metrics": {
                    "latency_ms": sparse_latency,
                    "energy_mj": None,
                    "bandwidth_bytes": 500,
                    "memory_bytes": 500,
                },
            },
            {
                "n": 4,
                "m": 4,
                "metrics": {
                    "latency_ms": 1.0,
                    "energy_mj": None,
                    "bandwidth_bytes": 1000,
                    "memory_bytes": 1000,
                },
            },
        ],
    }


def sensitivity_layer(name, sparse_drop):
    return {
        "name": name,
        "candidates": [
            {
                "n": 2,
                "m": 4,
                "sensitivity": {
                    "loss_increase": sparse_drop,
                    "top1_drop_percent": sparse_drop,
                    "top5_drop_percent": sparse_drop,
                },
            },
            {
                "n": 4,
                "m": 4,
                "sensitivity": {
                    "loss_increase": 0.0,
                    "top1_drop_percent": 0.0,
                    "top5_drop_percent": 0.0,
                },
            },
        ],
    }


class SensitivityAwareSchemeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        hardware = {
            "schema_version": 1,
            "hardware": {"name": "synthetic"},
            "model": {"name": "tiny", "layout": "NCHW"},
            "cost_definition": {
                "weights": {
                    "latency_ms": 1.0,
                    "energy_mj": 0.0,
                    "bandwidth_bytes": 0.0,
                    "memory_bytes": 0.0,
                }
            },
            "layers": [
                hardware_layer("first", "conv2d", 100, 0.5),
                hardware_layer("second", "conv2d", 300, 0.5),
                hardware_layer("head", "linear", 100, 0.5),
            ],
        }
        sensitivity = {
            "schema_version": 1,
            "method": "one-layer-at-a-time-nm-sensitivity",
            "model": {"name": "tiny"},
            "dataset": {"samples": 10},
            "measurement": {"baseline": {"top1_percent": 80}},
            "layers": [
                sensitivity_layer("first", 8.0),
                sensitivity_layer("second", 0.2),
                sensitivity_layer("head", 4.0),
            ],
        }
        hardware_path = root / "hardware.json"
        sensitivity_path = root / "sensitivity.json"
        hardware_path.write_text(json.dumps(hardware), encoding="utf-8")
        sensitivity_path.write_text(json.dumps(sensitivity), encoding="utf-8")
        self.hardware = HardwareCostModel(hardware_path, mode="lookup")
        self.sensitivity = LayerSensitivityProfile(sensitivity_path)

    def test_prefers_less_sensitive_layer_for_hardware_target(self):
        scheme, penalty = choose_scheme(
            self.hardware,
            self.sensitivity,
            0.15,
            "hardware-cost",
            "top1_drop_percent",
        )
        self.assertEqual(scheme["second"], [2, 4])
        self.assertEqual(scheme["first"], [4, 4])
        self.assertAlmostEqual(penalty, 0.2)

    def test_mac_target_uses_layer_macs_and_sensitivity(self):
        scheme, _ = choose_scheme(
            self.hardware,
            self.sensitivity,
            0.25,
            "macs",
            "loss_increase",
        )
        self.assertEqual(scheme["second"], [2, 4])

    def test_boundary_protection_keeps_first_and_linear_dense(self):
        scheme, _ = choose_scheme(
            self.hardware,
            self.sensitivity,
            0.10,
            "hardware-cost",
            "loss_increase",
            protect_first_conv=True,
            protect_linear=True,
        )
        self.assertEqual(scheme["first"], [4, 4])
        self.assertEqual(scheme["head"], [4, 4])
        self.assertEqual(scheme["second"], [2, 4])

    def test_profile_requires_exact_layer_coverage(self):
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            self.sensitivity.validate_candidates(["first"], [(2, 4), (4, 4)])


if __name__ == "__main__":
    unittest.main()
