from pathlib import Path

import torch

from astromae.data.dataset import (
    PreprocessedAstroDataset,
    create_train_val_dataloaders,
)
from astromae.data.preprocess import generate_synthetic_dataset


def _make_synth(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    generate_synthetic_dataset(processed, num_classes=3, per_class=6, image_size=32, in_chans=3)
    return processed


def test_dataset_discovers_classes(tmp_path):
    root = _make_synth(tmp_path)
    ds = PreprocessedAstroDataset(root)
    assert len(ds) == 18
    assert ds.classes == ["class_0", "class_1", "class_2"]
    sample, label = ds[0]
    assert sample.shape == (3, 32, 32)
    assert isinstance(label, int)


def test_train_val_split_is_deterministic(tmp_path):
    root = _make_synth(tmp_path)
    train1, val1 = create_train_val_dataloaders(root, batch_size=2, val_ratio=0.25, seed=0)
    train2, val2 = create_train_val_dataloaders(root, batch_size=2, val_ratio=0.25, seed=0)
    assert len(train1.dataset) == len(train2.dataset)
    assert len(val1.dataset) == len(val2.dataset)


def test_dataloader_yields_image_only_tensors(tmp_path):
    root = _make_synth(tmp_path)
    train, _ = create_train_val_dataloaders(root, batch_size=4, val_ratio=0.0, seed=1)
    batch = next(iter(train))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 3, 32, 32)
