"""Tests for astromae.data.transforms."""

import torch

from astromae.data.transforms import (
    AsinhStretch,
    BandNormalize,
    Compose,
    RandomFlipRotate90,
)


def test_asinh_stretch_is_monotone_and_propagates_ivar():
    image = torch.linspace(-1, 10, steps=32).reshape(1, 4, 8)
    ivar = torch.full_like(image, 4.0)
    stretched, new_ivar = AsinhStretch(softening=0.5)(image, ivar)

    assert stretched.shape == image.shape
    assert torch.all(torch.diff(stretched[0].flatten()) >= 0)
    assert new_ivar.shape == ivar.shape
    # For large x, asinh slope -> 1/x, so ivar grows with |x|
    assert float(new_ivar[0, -1, -1]) > float(new_ivar[0, 0, 0])


def test_band_normalize_reverses_with_std():
    image = torch.randn(3, 4, 4) * 2.0 + 1.0
    ivar = torch.full_like(image, 0.25)
    t = BandNormalize(mean=(1.0, 1.0, 1.0), std=(2.0, 2.0, 2.0))
    out, out_ivar = t(image, ivar)
    assert torch.allclose(out.mean(dim=(1, 2)), torch.zeros(3), atol=1.5)
    # Var(x/s) = Var(x)/s^2 → ivar *= s^2
    assert torch.allclose(out_ivar, ivar * 4.0, atol=1e-6)


def test_compose_runs_all_steps():
    image = torch.rand(3, 4, 4)
    pipe = Compose(
        [
            AsinhStretch(softening=0.1),
            BandNormalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ]
    )
    out, _ = pipe(image, None)
    assert out.shape == image.shape


def test_random_flip_rotate_90_preserves_shape():
    image = torch.rand(3, 8, 8)
    ivar = torch.rand(3, 8, 8)
    t = RandomFlipRotate90(seed=42)
    for _ in range(5):
        out_img, out_iv = t(image, ivar)
        assert out_img.shape == image.shape
        assert out_iv.shape == ivar.shape
