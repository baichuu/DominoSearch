#!/usr/bin/env python3
"""Download only ImageNet-1K validation shards and create an ImageFolder tree.

The Hugging Face ImageNet repository contains hundreds of training shards.  Using
``load_dataset(..., split="validation")`` may still resolve/download every split,
so this utility lists repository files first and explicitly downloads only files
whose names start with ``data/validation-``.

Access to ILSVRC/imagenet-1k is gated.  Accept its terms on Hugging Face and run
``huggingface-cli login`` (or set ``HF_TOKEN``) before invoking this script.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

DEFAULT_REPO_ID = "ILSVRC/imagenet-1k"
EXPECTED_CLASSES = 1_000
EXPECTED_IMAGES = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the gated ImageNet-1K validation split from Hugging "
            "Face and materialize it as an ImageFolder directory."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination ImageFolder directory, for example /content/imagenet-val.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/content/hf-imagenet-val-cache"),
        help="Cache used for validation parquet shards (default: Colab /content cache).",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Gated Hugging Face dataset repository (default: {DEFAULT_REPO_ID}).",
    )
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from datasets import Image, load_dataset
        from huggingface_hub import get_token, hf_hub_download, list_repo_files
    except ImportError as exc:
        raise SystemExit(
            "Missing dataset dependencies. Install them with: "
            "pip install -r requirements.txt"
        ) from exc
    return Image, load_dataset, get_token, hf_hub_download, list_repo_files


def validation_shards(repo_id: str, token: str, list_repo_files: Any) -> list[str]:
    files = list_repo_files(repo_id, repo_type="dataset", token=token)
    shards = sorted(
        name
        for name in files
        if name.startswith("data/validation-") and name.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError(
            "No validation parquet shards were found. Confirm that you accepted "
            f"the access terms for https://huggingface.co/datasets/{repo_id}."
        )
    if any("train-" in name or "test-" in name for name in shards):
        raise RuntimeError("Shard filter unexpectedly selected a non-validation file.")
    return shards


def download_shards(
    repo_id: str,
    shards: list[str],
    token: str,
    cache_dir: Path,
    hf_hub_download: Any,
) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_files: list[str] = []
    for index, filename in enumerate(shards, start=1):
        print(f"[{index}/{len(shards)}] Downloading {filename}", flush=True)
        local_files.append(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
                cache_dir=str(cache_dir),
            )
        )
    return local_files


def materialize_imagefolder(dataset: Any, output: Path, image_feature: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dataset = dataset.cast_column("image", image_feature(decode=False))

    for index, sample in enumerate(dataset):
        label = int(sample["label"])
        if not 0 <= label < EXPECTED_CLASSES:
            raise ValueError(f"Image {index} has invalid class label {label}.")

        class_dir = output / f"{label:04d}"
        class_dir.mkdir(exist_ok=True)
        destination = class_dir / f"{index:05d}.JPEG"
        if destination.exists():
            continue

        image = sample["image"]
        image_bytes = image.get("bytes")
        image_path = image.get("path")
        if image_bytes is not None:
            destination.write_bytes(image_bytes)
        elif image_path:
            shutil.copyfile(image_path, destination)
        else:
            raise ValueError(f"Image {index} contains neither bytes nor a source path.")

        if (index + 1) % 1_000 == 0:
            print(f"Materialized {index + 1}/{len(dataset)} images", flush=True)


def verify_imagefolder(output: Path) -> None:
    class_count = sum(1 for path in output.iterdir() if path.is_dir())
    image_count = sum(1 for _ in output.rglob("*.JPEG"))
    if class_count != EXPECTED_CLASSES or image_count != EXPECTED_IMAGES:
        raise RuntimeError(
            "Incomplete ImageNet validation directory: "
            f"expected {EXPECTED_CLASSES} classes/{EXPECTED_IMAGES} images, "
            f"found {class_count} classes/{image_count} images."
        )
    print(f"Ready: {output.resolve()}")
    print(f"Classes: {class_count}; images: {image_count}")


def main() -> None:
    args = parse_args()
    Image, load_dataset, get_token, hf_hub_download, list_repo_files = (
        import_dependencies()
    )

    token = get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Accept access at "
            f"https://huggingface.co/datasets/{args.repo_id}, then run "
            "`huggingface-cli login` or set HF_TOKEN."
        )

    shards = validation_shards(args.repo_id, token, list_repo_files)
    print(f"Selected {len(shards)} validation shard(s); no train/test shards selected.")
    local_files = download_shards(
        args.repo_id, shards, token, args.cache_dir, hf_hub_download
    )
    dataset = load_dataset(
        "parquet",
        data_files={"validation": local_files},
        split="validation",
        cache_dir=str(args.cache_dir / "datasets"),
    )
    if len(dataset) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} validation samples, found {len(dataset)}."
        )

    materialize_imagefolder(dataset, args.output, Image)
    verify_imagefolder(args.output)


if __name__ == "__main__":
    main()
