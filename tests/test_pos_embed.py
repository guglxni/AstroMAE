import numpy as np
import pytest

from astromae.models.pos_embed import get_2d_sincos_pos_embed


def test_shapes_without_cls():
    emb = get_2d_sincos_pos_embed(embed_dim=16, grid_size=4, cls_token=False)
    assert emb.shape == (16, 16)


def test_shapes_with_cls():
    emb = get_2d_sincos_pos_embed(embed_dim=16, grid_size=4, cls_token=True)
    assert emb.shape == (17, 16)
    assert np.all(emb[0] == 0.0)


def test_invalid_dim():
    with pytest.raises(ValueError):
        get_2d_sincos_pos_embed(embed_dim=6, grid_size=2)
