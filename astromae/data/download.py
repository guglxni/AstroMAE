"""Galaxy10 dataset downloader.

Uses :mod:`astroNN` when available. The dataset is cached under ``data/raw/``
organised by class so that :class:`torchvision.datasets.ImageFolder` works
out of the box.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
DEFAULT_NUM_IMAGES = 1000


def download_and_save_galaxy10(
    num_images: int = DEFAULT_NUM_IMAGES,
    save_dir: os.PathLike | str = DEFAULT_RAW_DIR,
) -> int:
    """Download Galaxy10 and save PNGs under ``save_dir/<class>/*.png``.

    Returns the number of images successfully saved.
    """
    try:
        from astroNN.datasets import load_galaxy10
        from astroNN.datasets.galaxy10 import galaxy10cls_lookup
    except ImportError as e:
        raise RuntimeError(
            "astroNN is required for `astromae data download`. Install with `pip install astroNN`."
        ) from e

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Galaxy10 dataset via astroNN...")
    images, labels = load_galaxy10()
    logger.info("Loaded %d images from Galaxy10.", len(images))

    num_images = min(num_images, len(images))
    images = images[:num_images]
    labels = labels[:num_images]

    saved = 0
    for i, (img_array, label_idx) in enumerate(zip(images, labels, strict=False)):
        class_name = galaxy10cls_lookup(label_idx)
        class_dir = save_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        arr = np.asarray(img_array)
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)

        img = Image.fromarray(arr)
        img.save(class_dir / f"galaxy_{i:05d}.png")
        saved += 1

    logger.info("Saved %d images to %s", saved, save_dir)
    return saved


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Download Galaxy10 dataset")
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args(argv)
    download_and_save_galaxy10(args.num_images, args.output_dir)


if __name__ == "__main__":
    main()
