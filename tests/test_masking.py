import torch

from astromae.models.mae import random_masking


def test_random_masking_shapes_and_counts():
    x = torch.rand(2, 16, 768)

    x_masked, mask, ids_restore = random_masking(x, 0.5)

    assert x_masked.shape == (2, 8, 768)
    assert mask.shape == (2, 16)
    assert ids_restore.shape == (2, 16)

    assert int(mask.sum().item()) == 16
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})
    assert torch.all(mask.sum(dim=1) == 8)


def test_random_masking_zero_ratio():
    x = torch.rand(3, 10, 8)
    x_masked, mask, _ = random_masking(x, 0.0)

    assert x_masked.shape == (3, 10, 8)
    assert torch.equal(mask, torch.zeros(3, 10))


def test_random_masking_deterministic_with_generator():
    x = torch.rand(2, 12, 8)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)

    a, mask_a, _ = random_masking(x, 0.5, generator=g1)
    b, mask_b, _ = random_masking(x, 0.5, generator=g2)

    assert torch.equal(mask_a, mask_b)
    assert torch.allclose(a, b)


def test_random_masking_rejects_bad_ratio():
    import pytest

    with pytest.raises(ValueError):
        random_masking(torch.rand(1, 4, 2), 1.0)
