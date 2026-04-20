"""I-JEPA style objective (Contribution C3, M4).

Joint-Embedding Predictive Architecture for images, after Assran et al.
ICML 2023 ("Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture"). This is a feature-space alternative to the
MAE pixel-reconstruction objective.

Quick recap of the math:
- ``context_encoder``: the trainable encoder f_θ.
- ``target_encoder``: an EMA (exponential moving average) of f_θ.
- A set of ``context`` tokens is fed to f_θ to produce context features.
- A set of ``target`` tokens is fed to f_ξ (EMA) to produce target
  features. Gradients do NOT flow into f_ξ.
- A small ``predictor`` head g_φ predicts target features from context
  features + target position embeddings.
- Loss = smooth L1 between predicted and target features, averaged
  over target tokens.

We keep the implementation minimal: ViT encoder shared with the MAE
codepath (so our evaluation protocol is architecture-matched between
MAE and JEPA), a random-block masking scheme that produces one context
and four target blocks per sample, and an EMA wrapper.

Why for astronomy
-----------------
JEPA doesn't reconstruct pixels, so it doesn't waste capacity modelling
sky noise. That's a structural advantage over MAE on low-information-
density astronomical images.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from astromae.models.mae import _Block, _DEFAULT_NORM, _PatchEmbed
from astromae.models.pos_embed import get_2d_sincos_pos_embed


@dataclass
class JEPAOutput:
    loss: torch.Tensor
    pred: torch.Tensor  # (N, M, D) predicted target features
    target: torch.Tensor  # (N, M, D) target features (stop-grad)
    context_mask: torch.Tensor  # (N, L) 1=in context
    target_mask: torch.Tensor  # (N, L) 1=in target


class _ViTEncoder(nn.Module):
    """Plain ViT encoder shared between JEPA context/target branches."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        norm_layer=_DEFAULT_NORM,
    ) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )
        pos = get_2d_sincos_pos_embed(
            embed_dim, int(self.num_patches**0.5), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        self.blocks = nn.ModuleList(
            [
                _Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def forward(
        self,
        imgs: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return token features. If ``keep_mask`` (N, L) is given, only
        keep_mask == 1 tokens are kept; output is (N, K, D)."""
        x = self.patch_embed(imgs) + self.pos_embed
        if keep_mask is not None:
            n, length, d = x.shape
            idx = keep_mask.bool()  # (N, L)
            # Variable-length selection per sample. We pack via masked_select
            # but for simplicity (and MAE-matched architecture) we assume
            # every sample keeps the same number of tokens.
            counts = idx.sum(dim=1)
            if not torch.all(counts == counts[0]):
                raise ValueError(
                    "_ViTEncoder expects a fixed number of kept tokens per sample"
                )
            k = int(counts[0].item())
            x = torch.stack(
                [x[i][idx[i]] for i in range(n)], dim=0
            )  # (N, K, D)
            assert x.shape == (n, k, d)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class _Predictor(nn.Module):
    """Small ViT predictor head for I-JEPA."""

    def __init__(
        self,
        embed_dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        num_patches: int,
        norm_layer=_DEFAULT_NORM,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_dim), requires_grad=False
        )
        pos = get_2d_sincos_pos_embed(
            predictor_dim, int(num_patches**0.5), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        self.blocks = nn.ModuleList(
            [
                _Block(predictor_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(predictor_dim)
        self.unembed = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context_feats: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target-token features from context-token features.

        Args:
            context_feats: (N, K_c, D) from context encoder.
            context_indices: (N, K_c) global token indices of the context.
            target_indices:  (N, K_t) global token indices of the target.
        Returns:
            (N, K_t, D) predicted target features.
        """
        n = context_feats.shape[0]
        k_c = context_feats.shape[1]
        k_t = target_indices.shape[1]

        ctx = self.embed(context_feats)
        ctx_pos = self.pos_embed.expand(n, -1, -1).gather(
            1, context_indices.unsqueeze(-1).expand(-1, -1, ctx.shape[-1])
        )
        ctx = ctx + ctx_pos

        mask_tokens = self.mask_token.expand(n, k_t, -1)
        tgt_pos = self.pos_embed.expand(n, -1, -1).gather(
            1, target_indices.unsqueeze(-1).expand(-1, -1, ctx.shape[-1])
        )
        tgt = mask_tokens + tgt_pos

        tokens = torch.cat([ctx, tgt], dim=1)  # (N, K_c + K_t, D)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        pred_tgt = tokens[:, k_c:, :]
        return self.unembed(pred_tgt)


class IJEPA(nn.Module):
    """Minimal I-JEPA implementation compatible with the AstroMAE pipeline."""

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        predictor_dim: int = 192,
        predictor_depth: int = 2,
        predictor_num_heads: int = 6,
        mlp_ratio: float = 4.0,
        ema_momentum: float = 0.996,
        context_ratio: float = 0.25,
        target_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.ema_momentum = ema_momentum
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio

        self.context_encoder = _ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.target_encoder = _ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        # Initialise target = context.
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        self.predictor = _Predictor(
            embed_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
            num_patches=self.context_encoder.num_patches,
        )

        self.num_patches = self.context_encoder.num_patches
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans

    @torch.no_grad()
    def _ema_update(self) -> None:
        m = self.ema_momentum
        for ctx_p, tgt_p in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            tgt_p.mul_(m).add_(ctx_p.detach(), alpha=1 - m)

    def _sample_masks(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample disjoint context / target token index sets.

        Returns ``(ctx_mask, tgt_mask, ctx_idx, tgt_idx)``:
          - ``ctx_mask``, ``tgt_mask``: ``(B, L)`` binary.
          - ``ctx_idx``, ``tgt_idx``: ``(B, K)`` gathered indices.
        """
        length = self.num_patches
        k_c = max(1, int(length * self.context_ratio))
        k_t = max(1, int(length * self.target_ratio))
        if k_c + k_t > length:
            k_t = max(1, length - k_c - 1)

        if generator is None:
            noise = torch.rand(batch_size, length, device=device)
        else:
            noise = torch.rand(batch_size, length, device=device, generator=generator)
        order = torch.argsort(noise, dim=1)
        ctx_idx = order[:, :k_c]
        tgt_idx = order[:, k_c : k_c + k_t]

        ctx_mask = torch.zeros(batch_size, length, device=device)
        ctx_mask.scatter_(1, ctx_idx, 1.0)
        tgt_mask = torch.zeros(batch_size, length, device=device)
        tgt_mask.scatter_(1, tgt_idx, 1.0)
        return ctx_mask, tgt_mask, ctx_idx, tgt_idx

    def forward(
        self,
        imgs: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> JEPAOutput:
        ctx_mask, tgt_mask, ctx_idx, tgt_idx = self._sample_masks(
            imgs.shape[0], imgs.device, generator=generator
        )

        ctx_feats = self.context_encoder(imgs, keep_mask=ctx_mask)

        with torch.no_grad():
            full_tgt_feats = self.target_encoder(imgs)  # (N, L, D)
            target_feats = torch.gather(
                full_tgt_feats,
                1,
                tgt_idx.unsqueeze(-1).expand(-1, -1, full_tgt_feats.shape[-1]),
            )  # (N, K_t, D)

        pred_feats = self.predictor(ctx_feats, ctx_idx, tgt_idx)
        loss = F.smooth_l1_loss(pred_feats, target_feats)
        return JEPAOutput(
            loss=loss,
            pred=pred_feats,
            target=target_feats,
            context_mask=ctx_mask,
            target_mask=tgt_mask,
        )

    @torch.no_grad()
    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled target-encoder features — the embedding
        used for downstream linear probes."""
        self.target_encoder.eval()
        feats = self.target_encoder(imgs)  # (N, L, D)
        return feats.mean(dim=1)


def ijepa_tiny(
    img_size: int = 128, patch_size: int = 16, in_chans: int = 3
) -> IJEPA:
    return IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=64,
        depth=2,
        num_heads=4,
        predictor_dim=32,
        predictor_depth=1,
        predictor_num_heads=4,
        mlp_ratio=2.0,
    )


def ijepa_small(
    img_size: int = 128, patch_size: int = 16, in_chans: int = 3
) -> IJEPA:
    return IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=384,
        depth=6,
        num_heads=6,
        predictor_dim=192,
        predictor_depth=2,
        predictor_num_heads=6,
        mlp_ratio=4.0,
    )
