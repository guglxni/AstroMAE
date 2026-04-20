"""2D sine-cosine positional embeddings (Vaswani et al. 2017 / MAE)."""

from __future__ import annotations

import numpy as np


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """Return sin-cos positional embeddings for a ``grid_size`` x ``grid_size`` grid.

    Args:
        embed_dim: the embedding dimension, must be divisible by 4.
        grid_size: height (= width) of the token grid.
        cls_token: if True, prepend a zero row for the [CLS] token.

    Returns:
        ``np.ndarray`` of shape ``[grid_size*grid_size (+ 1), embed_dim]``.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim], dtype=pos_embed.dtype), pos_embed], axis=0
        )
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"1D embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
