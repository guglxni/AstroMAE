"""Masked Autoencoder (He et al., 2021) with a Vision Transformer backbone.

This is a cleaned-up re-implementation of the original ``MaskedAutoencoder`` in
``src/model.py``. Fixes vs. the original:

* ``initialize_weights`` no longer calls a non-existent method
  ``self.get_2d_sincos_pos_embed``; it uses the helper from
  :mod:`astromae.models.pos_embed`.
* ``patchify``/``unpatchify`` honour ``self.in_chans`` rather than hardcoding 3.
* ``timm`` imports are defensive: we prefer ``timm.layers.PatchEmbed`` and fall
  back to ``timm.models.vision_transformer.PatchEmbed`` on older releases.
* Optional ``generator``-aware random masking for deterministic tests.
* Adds a ``forward_features`` entry-point used by the evaluation pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from astromae.models.pos_embed import get_2d_sincos_pos_embed

_DEFAULT_NORM: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6)

from timm.models.vision_transformer import Block as _Block  # noqa: E402 (defensive import below)

try:
    from timm.layers import PatchEmbed as _PatchEmbed
except ImportError:  # pragma: no cover - older timm
    from timm.models.vision_transformer import PatchEmbed as _PatchEmbed  # type: ignore


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample random masking.

    Args:
        x: ``[N, L, D]`` tokens.
        mask_ratio: fraction of tokens to drop.
        generator: optional ``torch.Generator`` for deterministic masking.

    Returns:
        ``x_masked`` of shape ``[N, L*(1-mask_ratio), D]``; ``mask`` of shape
        ``[N, L]`` with 0=keep / 1=drop; ``ids_restore`` of shape ``[N, L]``.
    """
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")

    n, length, dim = x.shape
    len_keep = int(length * (1.0 - mask_ratio))

    if generator is not None:
        noise = torch.rand(n, length, device=x.device, generator=generator)
    else:
        noise = torch.rand(n, length, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))

    mask = torch.ones([n, length], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class MaskedAutoencoder(nn.Module):
    """MAE with a ViT encoder and a lightweight ViT decoder."""

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
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = _DEFAULT_NORM
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")

        self.in_chans = in_chans
        self.patch_size = patch_size

        # --- encoder ---
        self.patch_embed = _PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                _Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # --- decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self._initialize_weights()

    # ------------------------------------------------------------------
    # weight init
    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        grid_size = int(self.patch_embed.num_patches**0.5)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # patch utilities
    # ------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """``(N, C, H, W)`` -> ``(N, L, p*p*C)`` where ``L = (H/p)*(W/p)``."""
        p = self.patch_size
        c = self.in_chans
        if imgs.shape[2] != imgs.shape[3] or imgs.shape[2] % p != 0:
            raise ValueError(f"Expected square input divisible by {p}, got {tuple(imgs.shape)}")
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p * p * c)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`patchify`."""
        p = self.patch_size
        c = self.in_chans
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"Non-square token grid: {x.shape[1]} tokens cannot be reshaped.")
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio > 0.0:
            x, mask, ids_restore = random_masking(x, mask_ratio, generator=generator)
        else:
            n, length, _ = x.shape
            mask = torch.zeros(n, length, device=x.device)
            ids_restore = torch.arange(length, device=x.device).unsqueeze(0).expand(n, -1)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, generator=generator)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ------------------------------------------------------------------
    # feature extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return the CLS embedding of the (unmasked) encoder.

        Intended for downstream evaluation: linear probes, clustering, etc.
        Shape: ``(N, embed_dim)``.
        """
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS


# ---------------------------------------------------------------------------
# model profiles
# ---------------------------------------------------------------------------
def mae_tiny(img_size: int = 128, patch_size: int = 16, in_chans: int = 3) -> MaskedAutoencoder:
    """Tiny MAE suitable for CPU smoke tests and unit tests."""
    return MaskedAutoencoder(
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


def mae_small(img_size: int = 128, patch_size: int = 16, in_chans: int = 3) -> MaskedAutoencoder:
    return MaskedAutoencoder(
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


def mae_base(img_size: int = 128, patch_size: int = 16, in_chans: int = 3) -> MaskedAutoencoder:
    return MaskedAutoencoder(
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
