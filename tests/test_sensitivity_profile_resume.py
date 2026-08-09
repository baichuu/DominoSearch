import json
import tempfile
import unittest
from pathlib import Path

from search.layer_sensitivity import (
    atomic_write_json,
    load_partial_profile,
    partial_profile_path,
)


def profile(checkpoint_sha="checkpoint-a", status="incomplete"):
    return {
        "schema_version": 1,
        "method": "conditioned-one-layer-at-a-time-nm-sensitivity",
        "model": {
            "name": "tiny",
            "checkpoint": {"sha256": checkpoint_sha},
        },
        "dataset": {"format": "parquet", "files": 14},
        "measurement": {"seed": 42, "baseline": {"samples": 10}},
        "base_scheme_file": {"sha256": "scheme-a"},
        "base_scheme": {"layer": [3, 4]},
        "candidate_n": [2, 3, 4],
        "m": 4,
        "layers": [],
        "progress": {"status": status, "completed_candidates": 0},
    }


class SensitivityProfileResumeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_partial_path_does_not_replace_final_suffix(self):
        output = self.root / "profile.json"
        self.assertEqual(partial_profile_path(output), self.root / "profile.json.partial.json")

    def test_atomic_write_produces_complete_json(self):
        output = self.root / "profile.partial.json"
        value = profile()
        atomic_write_json(output, value)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), value)
        self.assertFalse(Path(str(output) + ".tmp").exists())

    def test_resume_rejects_changed_checkpoint(self):
        output = self.root / "profile.partial.json"
        atomic_write_json(output, profile(checkpoint_sha="checkpoint-a"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_partial_profile(output, profile(checkpoint_sha="checkpoint-b"))

    def test_resume_rejects_completed_profile(self):
        output = self.root / "profile.partial.json"
        atomic_write_json(output, profile(status="complete"))
        with self.assertRaisesRegex(ValueError, "not marked incomplete"):
            load_partial_profile(output, profile(status="complete"))


if __name__ == "__main__":
    unittest.main()
