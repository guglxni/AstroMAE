"""Datasets and dataloaders for AstroMAE."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


class PreprocessedAstroDataset(Dataset):
    """Load preprocessed ``(C, H, W)`` tensors (``.pt``) from a class-foldered tree.

    Expected layout::

        <root_dir>/
            <class_a>/*.pt
            <class_b>/*.pt

    The class is inferred from the parent directory name.
    """

    def __init__(
        self,
        root_dir: os.PathLike | str,
        image_size: int | None = None,
        in_memory: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.in_memory = in_memory

        self.tensor_paths: list[Path] = sorted(
            Path(p) for p in glob.glob(str(self.root_dir / "**" / "*.pt"), recursive=True)
        )
        if not self.tensor_paths:
            logger.warning("No .pt files found recursively in %s", self.root_dir)

        class_names = sorted({p.parent.name for p in self.tensor_paths})
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.classes = class_names

        self._cache: list[torch.Tensor] | None = None
        if in_memory:
            logger.info("Caching %d tensors in memory", len(self.tensor_paths))
            self._cache = [self._load(p) for p in self.tensor_paths]

        logger.info(
            "PreprocessedAstroDataset: %d samples, %d classes at %s",
            len(self.tensor_paths),
            len(self.classes),
            self.root_dir,
        )

    def _load(self, path: Path) -> torch.Tensor:
        # ``weights_only=True`` is safer and the default in newer torch.
        try:
            tensor = torch.load(path, weights_only=True)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover - older torch
            tensor = torch.load(path)
        return tensor.float()

    def __len__(self) -> int:
        return len(self.tensor_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        tensor = self._cache[idx] if self._cache is not None else self._load(self.tensor_paths[idx])
        label = self.class_to_idx[self.tensor_paths[idx].parent.name]
        return tensor, label


class _ImageOnlyWrapper(Dataset):
    """Drop labels from the underlying dataset, keeping only the image tensor."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.base[idx]
        if isinstance(sample, tuple):
            return sample[0]
        return sample


def create_dataloader(
    dataset_dir: os.PathLike | str,
    batch_size: int,
    num_workers: int = 0,
    image_size: int | None = None,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    in_memory: bool = False,
    return_labels: bool = False,
) -> DataLoader | None:
    """Build a DataLoader for MAE pretraining (images only) or eval (with labels)."""
    dataset = PreprocessedAstroDataset(
        root_dir=dataset_dir, image_size=image_size, in_memory=in_memory
    )
    if len(dataset) == 0:
        logger.error("Dataset at %s is empty.", dataset_dir)
        return None

    backing: Dataset = dataset if return_labels else _ImageOnlyWrapper(dataset)
    return DataLoader(
        backing,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_train_val_dataloaders(
    dataset_dir: os.PathLike | str,
    batch_size: int,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    image_size: int | None = None,
    seed: int = 42,
    in_memory: bool = False,
    return_labels: bool = False,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train/val dataloaders with a deterministic split."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    dataset = PreprocessedAstroDataset(
        root_dir=dataset_dir, image_size=image_size, in_memory=in_memory
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {dataset_dir}")

    if val_ratio == 0.0:
        train_loader = DataLoader(
            dataset if return_labels else _ImageOnlyWrapper(dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader, None

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    def _wrap(ds: Dataset) -> Dataset:
        return ds if return_labels else _ImageOnlyWrapper(ds)

    train_loader = DataLoader(
        _wrap(train_ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        _wrap(val_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
