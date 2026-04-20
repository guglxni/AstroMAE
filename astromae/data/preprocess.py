"""Preprocess raw PNG galaxy images into ``.pt`` tensors."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
_IMG_EXT = {".png", ".jpg", ".jpeg"}


def _build_transform(
    image_size: int,
    mean: Iterable[float] | None = None,
    std: Iterable[float] | None = None,
) -> transforms.Compose:
    steps: list = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if mean is not None and std is not None:
        steps.append(transforms.Normalize(mean=list(mean), std=list(std)))
    return transforms.Compose(steps)


def _iter_images(raw_dir: Path) -> Iterable[tuple[str, Path]]:
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in _IMG_EXT:
                yield class_dir.name, img_path


def compute_channel_stats(
    raw_dir: Path, image_size: int, max_images: int = 500
) -> tuple[list[float], list[float]]:
    """Compute per-channel mean / std on a sample of raw images."""
    resize = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    mean = torch.zeros(3)
    sq = torch.zeros(3)
    n = 0
    for _, path in _iter_images(raw_dir):
        img = Image.open(path).convert("RGB")
        t = resize(img)
        mean += t.mean(dim=(1, 2))
        sq += (t**2).mean(dim=(1, 2))
        n += 1
        if n >= max_images:
            break
    mean /= max(n, 1)
    var = sq / max(n, 1) - mean**2
    std = var.clamp(min=1e-6).sqrt()
    return mean.tolist(), std.tolist()


def preprocess_images(
    raw_dir: os.PathLike | str = DEFAULT_RAW_DIR,
    processed_dir: os.PathLike | str = DEFAULT_PROCESSED_DIR,
    image_size: int = 128,
    normalize: bool = False,
    clean: bool = True,
) -> int:
    """Resize & tensorise images under ``raw_dir``, saving to ``processed_dir``."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. Run `astromae data download` first."
        )

    if clean and processed_dir.exists():
        logger.warning("Wiping existing processed directory: %s", processed_dir)
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    mean: list[float] | None = None
    std: list[float] | None = None
    if normalize:
        mean, std = compute_channel_stats(raw_dir, image_size)
        logger.info("Computed channel stats: mean=%s std=%s", mean, std)
        stats_path = processed_dir / "channel_stats.pt"
        torch.save({"mean": mean, "std": std, "image_size": image_size}, stats_path)

    transform = _build_transform(image_size, mean, std)

    count = 0
    for class_name, img_path in _iter_images(raw_dir):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:  # pragma: no cover - robust against bad files
            logger.error("Failed to open %s: %s", img_path, e)
            continue
        tensor = transform(img)
        out_dir = processed_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, out_dir / f"{img_path.stem}.pt")
        count += 1

    logger.info("Processed %d images to %s", count, processed_dir)
    return count


def generate_synthetic_dataset(
    processed_dir: os.PathLike | str,
    num_classes: int = 2,
    per_class: int = 8,
    image_size: int = 64,
    in_chans: int = 3,
    seed: int = 0,
) -> int:
    """Create a synthetic labelled dataset for tests / CI smoke runs."""
    processed_dir = Path(processed_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator().manual_seed(seed)
    count = 0
    for c in range(num_classes):
        class_dir = processed_dir / f"class_{c}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            # Bias each class's mean so downstream probes have signal to learn.
            base = torch.rand(in_chans, image_size, image_size, generator=rng)
            tensor = (base + 0.5 * c).clamp(0, 1)
            torch.save(tensor, class_dir / f"sample_{i:03d}.pt")
            count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Preprocess AstroMAE raw images")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="If > 0, skip raw images and generate N synthetic samples per class.",
    )
    parser.add_argument("--synthetic-classes", type=int, default=2)
    args = parser.parse_args(argv)

    if args.synthetic > 0:
        generate_synthetic_dataset(
            args.processed_dir,
            num_classes=args.synthetic_classes,
            per_class=args.synthetic,
            image_size=args.image_size,
        )
    else:
        preprocess_images(args.raw_dir, args.processed_dir, args.image_size, args.normalize)


if __name__ == "__main__":
    main()
