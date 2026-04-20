"""Unit tests for FITS I/O and the synthetic FITS generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("astropy")

from astromae.data.fits import FITSCutoutDataset, load_fits_cutout, save_fits_cutout
from astromae.data.synth import (
    SURVEY_A,
    make_synthetic_cutout,
    make_synthetic_fits_dataset,
)


def test_synthetic_cutout_shapes_and_ivar_positivity():
    image, ivar, meta = make_synthetic_cutout(size=32, bands=("g", "r", "z"))
    assert image.shape == (3, 32, 32)
    assert ivar.shape == (3, 32, 32)
    assert np.all(ivar >= 0)
    assert np.isfinite(image).all()
    assert meta["class_idx"] == 0
    assert meta["survey"] == SURVEY_A.name


def test_save_and_load_roundtrip(tmp_path):
    image, ivar, _ = make_synthetic_cutout(size=32, bands=("g", "r", "z"), class_idx=1)
    path = tmp_path / "sample.fits"
    save_fits_cutout(path, image=image, ivar=ivar, bands=("g", "r", "z"))
    cutout = load_fits_cutout(path, bands=("g", "r", "z"))
    assert cutout.image.shape == (3, 32, 32)
    assert cutout.ivar.shape == (3, 32, 32)
    assert torch.allclose(cutout.image, torch.from_numpy(image), atol=1e-5)
    assert torch.allclose(cutout.ivar, torch.from_numpy(ivar), atol=1e-5)


def test_load_file_without_ivar_returns_ones(tmp_path):
    image, _, _ = make_synthetic_cutout(size=16, bands=("g", "r"))
    path = tmp_path / "noivar.fits"
    save_fits_cutout(path, image=image, ivar=None, bands=("g", "r"))
    cutout = load_fits_cutout(path, bands=("g", "r"))
    assert cutout.ivar.shape == (2, 16, 16)
    assert torch.all(cutout.ivar > 0)
    assert torch.allclose(cutout.ivar, torch.ones_like(cutout.ivar))


def test_fits_dataset_lists_classes(tmp_path):
    make_synthetic_fits_dataset(tmp_path / "set", num_classes=3, per_class=2, size=16, seed=0)
    ds = FITSCutoutDataset(tmp_path / "set", bands=("g", "r", "z"))
    assert len(ds) == 6
    assert ds.classes == ["class_0", "class_1", "class_2"]
    sample = ds[0]
    assert set(sample.keys()) == {"image", "ivar", "label"}
    assert sample["image"].shape == (3, 16, 16)
    assert sample["ivar"].shape == (3, 16, 16)
    assert isinstance(sample["label"], int)


def test_load_handles_nan_and_clamps_ivar(tmp_path):
    image, ivar, _ = make_synthetic_cutout(size=16, bands=("g",))
    image[0, 0, 0] = np.nan
    ivar[0, 0, 0] = np.inf
    path = tmp_path / "bad.fits"
    save_fits_cutout(path, image=image, ivar=ivar, bands=("g",))
    cutout = load_fits_cutout(path, bands=("g",))
    assert torch.isfinite(cutout.image).all()
    assert torch.isfinite(cutout.ivar).all()
    assert torch.all(cutout.ivar >= 0)


def test_load_uses_variance_plane_when_ivar_missing(tmp_path):
    from astropy.io import fits

    image = np.ones((1, 8, 8), dtype=np.float32)
    var = np.full((8, 8), 4.0, dtype=np.float32)  # ivar expected 0.25
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.append(fits.ImageHDU(image[0], name="G"))
    hdul.append(fits.ImageHDU(var, name="VAR"))
    path = Path(tmp_path) / "var.fits"
    hdul.writeto(path, overwrite=True)

    cutout = load_fits_cutout(path, bands=("g",))
    assert cutout.ivar.shape == (1, 8, 8)
    assert torch.allclose(cutout.ivar, torch.full_like(cutout.ivar, 0.25), atol=1e-5)


def test_mask_plane_zeros_ivar_on_flagged_pixels(tmp_path):
    from astropy.io import fits

    image = np.ones((1, 4, 4), dtype=np.float32)
    ivar = np.ones((1, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.int16)
    mask[0, 0] = 1

    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.append(fits.ImageHDU(image[0], name="G"))
    hdul.append(fits.ImageHDU(ivar[0], name="G_IVAR"))
    hdul.append(fits.ImageHDU(mask, name="MASK"))
    path = Path(tmp_path) / "masked.fits"
    hdul.writeto(path, overwrite=True)

    cutout = load_fits_cutout(path, bands=("g",))
    assert float(cutout.ivar[0, 0, 0]) <= 1e-10
    assert torch.all(cutout.ivar[0, 1:, 1:] > 0.9)
