import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from imagenet_data import ParquetImageNetDataset


class ParquetImageNetDatasetTest(unittest.TestCase):
    def make_dataset(self, num_samples=10, rank=0, world_size=1):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        shard = Path(temporary.name) / "train-00000.parquet"
        shard.touch()
        dataset = ParquetImageNetDataset(
            temporary.name,
            "train-*.parquet",
            "train",
            None,
            num_samples,
            shuffle=False,
            seed=42,
            shuffle_buffer=1,
            rank=rank,
            world_size=world_size,
            expected_shards=1,
        )
        examples = [
            {"image": Image.new("RGB", (1, 1)), "label": index}
            for index in range(20)
        ]
        dataset._stream = lambda: iter(examples)
        return dataset

    def test_rank_length_is_exact_for_non_divisible_sample_count(self):
        self.assertEqual(len(self.make_dataset(rank=0, world_size=3)), 4)
        self.assertEqual(len(self.make_dataset(rank=1, world_size=3)), 3)
        self.assertEqual(len(self.make_dataset(rank=2, world_size=3)), 3)

    def test_each_worker_stops_at_its_share_of_reported_length(self):
        expected = [4, 3, 3]
        for worker_id, expected_count in enumerate(expected):
            dataset = self.make_dataset()
            worker = SimpleNamespace(id=worker_id, num_workers=3)
            with patch("imagenet_data.get_worker_info", return_value=worker):
                self.assertEqual(len(list(dataset)), expected_count)

    def test_single_process_stops_at_declared_sample_count(self):
        dataset = self.make_dataset(num_samples=7)
        with patch("imagenet_data.get_worker_info", return_value=None):
            self.assertEqual(len(list(dataset)), 7)


if __name__ == "__main__":
    unittest.main()
