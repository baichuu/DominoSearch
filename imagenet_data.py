"""Shared ImageNet dataset helpers for local and Google Drive Parquet shards.

The Parquet loader is intentionally opt-in. Existing ImageFolder and meta-file
workflows keep their original behavior, while Colab can stream the downloaded
Hugging Face ImageNet shards directly from a mounted Google Drive directory.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image
from torch.utils.data import IterableDataset


IMAGENET_SPLIT_SAMPLES = {"train": 1_281_167, "validation": 50_000}
IMAGENET_SPLIT_SHARDS = {"train": 294, "validation": 14}


def resolve_parquet_files(root: str | Path, pattern: str) -> tuple[Path, ...]:
    """Resolve and validate a deterministic list of local Parquet shards."""

    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Parquet dataset root does not exist: {dataset_root}")
    files = tuple(sorted(path for path in dataset_root.glob(pattern) if path.is_file()))
    if not files:
        raise FileNotFoundError(
            f"No Parquet shards matched {pattern!r} under {dataset_root}"
        )
    return files


def parquet_manifest(root: str | Path, pattern: str) -> dict[str, Any]:
    files = resolve_parquet_files(root, pattern)
    return {
        "root": str(Path(root).expanduser().resolve()),
        "pattern": pattern,
        "shard_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "first_shard": str(files[0]),
        "last_shard": str(files[-1]),
    }


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        encoded = value.get("bytes")
        path = value.get("path")
        if encoded is not None:
            with Image.open(io.BytesIO(encoded)) as image:
                return image.convert("RGB")
        if path:
            with Image.open(path) as image:
                return image.convert("RGB")
    raise TypeError(
        "Parquet image column must decode to PIL.Image or contain bytes/path; "
        f"received {type(value).__name__}"
    )


class ParquetImageNetDataset(IterableDataset):
    """Stream ImageNet Parquet shards without materializing millions of files.

    Hugging Face Datasets performs lazy Parquet reads. Rank splitting happens
    before worker splitting, so each process receives a disjoint stream. The
    reported length is exact for the single-GPU Colab workflow and a balanced
    estimate for distributed execution.
    """

    def __init__(
        self,
        root: str | Path,
        pattern: str,
        split: str,
        transform: Callable[[Image.Image], Any] | None,
        num_samples: int,
        *,
        shuffle: bool,
        seed: int,
        shuffle_buffer: int,
        rank: int = 0,
        world_size: int = 1,
        image_column: str = "image",
        label_column: str = "label",
        expected_shards: int | None = None,
    ) -> None:
        super().__init__()
        if split not in IMAGENET_SPLIT_SAMPLES:
            raise ValueError(f"Unsupported ImageNet split: {split!r}")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive for a streaming dataset")
        if not 0 <= rank < world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")

        self.files = resolve_parquet_files(root, pattern)
        required_shards = (
            IMAGENET_SPLIT_SHARDS[split] if expected_shards is None else expected_shards
        )
        if required_shards <= 0:
            raise ValueError("expected_shards must be positive")
        if len(self.files) != required_shards:
            raise ValueError(
                f"Incomplete ImageNet {split} split: expected {required_shards} "
                f"Parquet shard(s), found {len(self.files)} under {root!s}"
            )
        self.root = Path(root).expanduser().resolve()
        self.pattern = pattern
        self.split = split
        self.transform = transform
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.rank = rank
        self.world_size = world_size
        self.image_column = image_column
        self.label_column = label_column
        self.expected_shards = required_shards
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.world_size)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def _stream(self):
        try:
            from datasets import load_dataset
            from datasets.distributed import split_dataset_by_node
        except ImportError as exc:
            raise RuntimeError(
                "Parquet streaming requires the 'datasets' package. "
                "Install requirements.txt before running."
            ) from exc

        dataset = load_dataset(
            "parquet",
            data_files={self.split: [str(path) for path in self.files]},
            split=self.split,
            streaming=True,
        )
        if self.shuffle:
            dataset = dataset.shuffle(
                seed=self.seed + self.epoch,
                buffer_size=self.shuffle_buffer,
            )
        if self.world_size > 1:
            dataset = split_dataset_by_node(
                dataset, rank=self.rank, world_size=self.world_size
            )
        # Hugging Face IterableDataset reads PyTorch worker information inside
        # its own iterator and assigns input shards automatically. Sharding it
        # again here would drop data when DataLoader uses multiple workers.
        return dataset

    def __iter__(self) -> Iterator[tuple[Any, int]]:
        for example in self._stream():
            if self.image_column not in example or self.label_column not in example:
                raise KeyError(
                    "Parquet rows must contain columns "
                    f"{self.image_column!r} and {self.label_column!r}; "
                    f"found {sorted(example)}"
                )
            image = _decode_image(example[self.image_column])
            if self.transform is not None:
                image = self.transform(image)
            yield image, int(example[self.label_column])

    def manifest(self) -> dict[str, Any]:
        result = parquet_manifest(self.root, self.pattern)
        result.update(
            {
                "split": self.split,
                "expected_samples": self.num_samples,
                "expected_shards": self.expected_shards,
                "shuffle": self.shuffle,
                "shuffle_buffer": self.shuffle_buffer if self.shuffle else None,
            }
        )
        return result
