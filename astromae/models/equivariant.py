"""Rotation-equivariant components for astro ViT-MAE (Contribution C2).

We build a *C₄-equivariant* patch embedding by averaging the output of the
same convolutional kernel at 0°, 90°, 180°, 270° rotations. The positional
embedding is constructed from polar coordinates (r, θ) so a 90° rotation of
the image corresponds to a known permutation of the patch grid and, after
averaging the position-embedding contributions, the CLS-token embedding is
approximately invariant under C₄.

This is intentionally the *simplest* way to get approximate C₄-equivariance
in a ViT-MAE — sufficient for a falsifiable smoke test and a CPU-runnable
demonstration. A full e2cnn/ESCNN-style exact steerable implementation is
an obvious extension.

Implementation notes
--------------------
- ``RotEquivPatchEmbed`` wraps a ``nn.Conv2d`` of stride=patch_size and
  applies it to the four C₄ rotations of the image, then averages the
  per-rotation patch-token grids (each also appropriately un-rotated back
  to a canonical patch orientation). The resulting token features are
  exactly C₄-equivariant at the level of the patch grid up to averaging.
- ``C4PolarPositionalEmbedding`` produces sin-cos features of (r, θ) so
  rotating the image by 90° rotates θ by π/2 for every patch token
  consistently. Used as a *non-learned* add-in to the encoder token
  stream.
- ``MaskedAutoencoderC4`` subclasses the existing :class:`~astromae.
  models.mae.MaskedAutoencoder` and swaps the patch embedding and
  position embedding.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from astromae.models.mae import _DEFAULT_NORM, MaskedAutoencoder

# ---------------------------------------------------------------------------
# Patch embed
# ---------------------------------------------------------------------------


class RotEquivPatchEmbed(nn.Module):
    """C₄-equivariant patch embedding via group averaging.

    Mirrors the API of ``timm.layers.PatchEmbed`` enough for the MAE code
    path (``num_patches``, ``patch_size``, ``proj.weight``).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.grid_size = img_size // patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute patch features at each rotation, then un-rotate the
        # resulting (B, D, H, W) token grid back to a canonical orientation
        # and average. The result is C₄-invariant at the level of the token
        # grid (strongest form of equivariance we can get without a
        # representation-theoretic decomposition).
        acc = None
        for k in range(4):
            x_rot = torch.rot90(x, k=k, dims=(-2, -1))
            feats = self.proj(x_rot)  # (B, D, H', W')
            # Un-rotate the feature grid back so tokens align.
            feats = torch.rot90(feats, k=-k, dims=(-2, -1))
            acc = feats if acc is None else acc + feats
        feats = acc / 4.0
        feats = feats.flatten(2).transpose(1, 2)  # (B, N, D)
        return feats


# ---------------------------------------------------------------------------
# Positional embedding
# ---------------------------------------------------------------------------


def c4_polar_positional_embedding(
    embed_dim: int, grid_size: int, cls_token: bool = True
) -> np.ndarray:
    """Polar (r, θ)-based sin-cos positional embedding.

    The grid is placed on a unit square centred at the origin; each token
    has coordinates (r, θ). A 90° rotation of the image corresponds to
    θ → θ + π/2, so sin/cos features with integer multiples of 4θ are
    exactly C₄-invariant, while features with θ-multiples of 1/2/3 are
    equivariant in a known way. We interleave both in roughly equal parts
    so the network can use whichever it finds useful.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

    ys = np.arange(grid_size, dtype=np.float32) + 0.5
    xs = np.arange(grid_size, dtype=np.float32) + 0.5
    xs = (xs / grid_size) - 0.5
    ys = (ys / grid_size) - 0.5
    xg, yg = np.meshgrid(xs, ys)
    r = np.sqrt(xg**2 + yg**2).reshape(-1)
    theta = np.arctan2(yg, xg).reshape(-1)

    half = embed_dim // 2
    freqs_r = np.arange(1, half // 2 + 1, dtype=np.float32)
    freqs_theta = np.arange(1, embed_dim // 4 + 1, dtype=np.float32)

    # r-channels: sin / cos at several frequencies.
    r_sin = np.stack([np.sin(2 * math.pi * f * r) for f in freqs_r], axis=-1)  # (N, F_r)
    r_cos = np.stack([np.cos(2 * math.pi * f * r) for f in freqs_r], axis=-1)
    # θ-channels: sin/cos of 4f*θ is C₄-invariant.
    t_sin = np.stack([np.sin(4.0 * f * theta) for f in freqs_theta], axis=-1)
    t_cos = np.stack([np.cos(4.0 * f * theta) for f in freqs_theta], axis=-1)

    emb = np.concatenate([r_sin, r_cos, t_sin, t_cos], axis=-1).astype(np.float32)
    # Pad or truncate to embed_dim in case of off-by-one from integer arithmetic.
    if emb.shape[1] < embed_dim:
        pad = np.zeros((emb.shape[0], embed_dim - emb.shape[1]), dtype=np.float32)
        emb = np.concatenate([emb, pad], axis=1)
    elif emb.shape[1] > embed_dim:
        emb = emb[:, :embed_dim]

    if cls_token:
        emb = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), emb], axis=0)
    return emb


# ---------------------------------------------------------------------------
# Full C₄-equivariant MAE
# ---------------------------------------------------------------------------


class MaskedAutoencoderC4(MaskedAutoencoder):
    """C₄-approximately-equivariant MAE.

    Inherits from :class:`~astromae.models.mae.MaskedAutoencoder` and:
      1. Replaces the patch embedding with :class:`RotEquivPatchEmbed`.
      2. Replaces the fixed 2D sin-cos positional embedding with a
         :func:`c4_polar_positional_embedding`.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer=None,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer or _DEFAULT_NORM,
        )

        # Replace patch_embed; keep num_patches consistent.
        self.patch_embed = RotEquivPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        grid_size = self.patch_embed.grid_size

        # Replace encoder position embedding with the polar one.
        pos = c4_polar_positional_embedding(embed_dim, grid_size, cls_token=True)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos).float().unsqueeze(0), requires_grad=False
        )

        # Keep the MAE decoder position embedding as-is; decoder needs to
        # be *expressive*, not equivariant, since we want to reconstruct
        # pixels at specific locations.


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def mae_c4_tiny(
    img_size: int = 128, patch_size: int = 16, in_chans: int = 3
) -> MaskedAutoencoderC4:
    return MaskedAutoencoderC4(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=64,
        depth=2,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2.0,
    )


def mae_c4_small(
    img_size: int = 128, patch_size: int = 16, in_chans: int = 3
) -> MaskedAutoencoderC4:
    return MaskedAutoencoderC4(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4.0,
    )


def mae_c4_base(
    img_size: int = 128, patch_size: int = 16, in_chans: int = 3
) -> MaskedAutoencoderC4:
    return MaskedAutoencoderC4(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
    )


def c4_equivariance_error(model: MaskedAutoencoder, imgs: torch.Tensor) -> torch.Tensor:
    """Mean relative embedding distance ``||f(R_k x) - f(x)|| / ||f(x)||``
    for k in {1, 2, 3} in the C_4 group.

    Used by the falsifiable-prediction test for C2: vanilla MAE should
    have error ≫ 0 while C₄-MAE should be near zero.
    """
    model.eval()
    with torch.no_grad():
        feats = model.forward_features(imgs)
        errs = []
        for k in (1, 2, 3):
            rot = torch.rot90(imgs, k=k, dims=(-2, -1))
            feats_r = model.forward_features(rot)
            num = (feats_r - feats).norm(dim=-1)
            den = feats.norm(dim=-1).clamp(min=1e-8)
            errs.append((num / den).mean())
    return torch.stack(errs).mean()
