"""Reconstruction losses for MAE variants.

This module lives on a research-track and is the implementation of
**Contribution C1** in ``aidlc-docs/research/novel-contributions.md``:
the *IVAR-weighted masked reconstruction loss*.

The standard MAE loss averages squared pixel error over masked patches,
implicitly assuming homoscedastic Gaussian noise. Astronomical pixels
are not homoscedastic: bright pixels have large Poisson shot noise,
faint (sky-dominated) pixels have small variance. FITS cutouts ship
with per-pixel inverse-variance (IVAR) maps. Using them as weights in
the reconstruction loss is the physically correct thing to do and is,
to our knowledge, not implemented in any published astronomical MAE
(Mohamadi 2024, Walmsley Euclid-MAE 2024-25, MMAE 2025 all use plain
MSE).

Two forms are provided:

``mse_masked_loss``:
    The classical MAE loss.  Patchified pred / target, mean squared error
    per pixel, averaged per patch, then averaged over *masked* patches only
    (with the MAE ``mask`` where 1 = masked-out, 0 = kept).

``ivar_weighted_masked_loss``:
    Same shape as above but weights the pixel-squared-error by
    per-pixel IVAR.  When ``ivar`` is uniform this is exactly equivalent
    to ``mse_masked_loss`` (up to a constant scale), which our tests
    explicitly verify.
"""

from __future__ import annotations

import torch


def patchify_with_channels(imgs: torch.Tensor, patch_size: int, in_chans: int) -> torch.Tensor:
    """(N, C, H, W) -> (N, L, p*p*C)."""
    p, c = patch_size, in_chans
    if imgs.shape[2] != imgs.shape[3] or imgs.shape[2] % p != 0:
        raise ValueError(f"Expected square input divisible by {p}, got {tuple(imgs.shape)}")
    h = w = imgs.shape[2] // p
    x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(imgs.shape[0], h * w, p * p * c)


def mse_masked_loss(
    pred: torch.Tensor,
    target_patches: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Vanilla MAE reconstruction loss.

    Args:
        pred: ``(N, L, P)`` predicted patch content.
        target_patches: ``(N, L, P)`` true patch content (already patchified).
        mask: ``(N, L)`` where 1 = patch was masked-out (loss applies),
            0 = kept (loss ignored).
    """
    per_patch = (pred - target_patches).pow(2).mean(dim=-1)  # (N, L)
    denom = mask.sum().clamp(min=1.0)
    return (per_patch * mask).sum() / denom


def ivar_weighted_masked_loss(
    pred: torch.Tensor,
    target_patches: torch.Tensor,
    mask: torch.Tensor,
    ivar_patches: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """IVAR-weighted MAE reconstruction loss (Contribution C1).

    Per pixel :math:`p`, with true value :math:`y_p`, prediction
    :math:`\\hat{y}_p` and inverse variance :math:`w_p = 1 / \\sigma_p^2`
    the per-patch loss is

    .. math::
        \\ell_\\text{patch} = \\frac{\\sum_p w_p \\, (\\hat{y}_p - y_p)^2}
                                   {\\sum_p w_p + \\varepsilon}

    and the total loss is the mean of :math:`\\ell_\\text{patch}` over
    masked patches. Setting all :math:`w_p \\equiv 1` recovers
    :func:`mse_masked_loss` up to a constant scale, which we verify in
    the unit tests.

    Args:
        pred: ``(N, L, P)``.
        target_patches: ``(N, L, P)``.
        mask: ``(N, L)`` — 1 = masked-out, 0 = kept.
        ivar_patches: ``(N, L, P)`` inverse-variance weights,
            patchified the same way as ``target_patches``. Must be
            non-negative.
        eps: numerical floor added to ``sum(w)`` to avoid division by
            zero on patches where every weight is zero (e.g. fully
            masked-out IVAR from a bad-pixel mask).
    """
    if ivar_patches.shape != target_patches.shape:
        raise ValueError(
            f"ivar_patches shape {ivar_patches.shape} must match "
            f"target_patches shape {target_patches.shape}"
        )
    if torch.any(ivar_patches < 0):
        raise ValueError("ivar_patches must be non-negative")

    sq_err = (pred - target_patches).pow(2)
    weighted = ivar_patches * sq_err
    numer = weighted.sum(dim=-1)
    denom = ivar_patches.sum(dim=-1).clamp(min=eps)
    per_patch = numer / denom  # (N, L)

    mask_sum = mask.sum().clamp(min=1.0)
    return (per_patch * mask).sum() / mask_sum


def chi_squared_outer_annulus(
    pred: torch.Tensor,
    target: torch.Tensor,
    ivar: torch.Tensor,
    inner_radius_frac: float = 0.7,
) -> torch.Tensor:
    """Low-surface-brightness chi-squared in the outer annulus.

    Used as the falsifiable-prediction metric for Contribution C1: does
    IVAR-weighted pretraining reconstruct the low-SB regime better than
    unweighted MSE?

    Args:
        pred / target / ivar: ``(N, C, H, W)`` tensors.
        inner_radius_frac: start of the outer annulus, as a fraction of
            ``min(H, W) / 2``. Pixels with radial distance below this
            fraction are excluded.
    """
    if pred.shape != target.shape or pred.shape != ivar.shape:
        raise ValueError("pred, target and ivar must share shape (N, C, H, W)")
    _, _, h, w = pred.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(h, device=pred.device).float(),
        torch.arange(w, device=pred.device).float(),
        indexing="ij",
    )
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_outer = inner_radius_frac * min(h, w) / 2.0
    mask = (r >= r_outer).float()  # (H, W)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    chi2 = ivar * (pred - target) ** 2 * mask
    n_pix = mask.sum() * pred.shape[0] * pred.shape[1]
    return chi2.sum() / n_pix.clamp(min=1.0)


def equivariance_regulariser(
    feats: torch.Tensor, feats_rotated: list[torch.Tensor]
) -> torch.Tensor:
    """C4-equivariance consistency loss.

    Given CLS-token embeddings of x and of rot90^n(x) for n ∈ {1, 2, 3},
    returns the mean squared distance. Contribution C2 auxiliary loss.
    """
    if not feats_rotated:
        return torch.zeros((), device=feats.device)
    losses = [(f - feats).pow(2).mean() for f in feats_rotated]
    return torch.stack(losses).mean()


def patchify_ivar(ivar: torch.Tensor, patch_size: int, in_chans: int | None = None) -> torch.Tensor:
    """Patchify an IVAR map with the same shape contract as the target patches."""
    c = ivar.shape[1] if in_chans is None else in_chans
    return patchify_with_channels(ivar, patch_size, c)
