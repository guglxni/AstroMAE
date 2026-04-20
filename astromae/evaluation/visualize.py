"""Reconstruction visualisation for MAE."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from astromae.models.mae import MaskedAutoencoder

logger = logging.getLogger(__name__)


@torch.no_grad()
def save_reconstruction_grid(
    model: MaskedAutoencoder,
    images: torch.Tensor,
    out_path: os.PathLike | str,
    mask_ratio: float = 0.75,
    device: torch.device | None = None,
) -> Path:
    """Save a ``[original | masked | reconstruction]`` grid PNG.

    Requires ``matplotlib``; raises ``ImportError`` otherwise.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for reconstruction visualisation. "
            "Install with `pip install matplotlib`."
        ) from e

    if device is None:
        device = next(model.parameters()).device
    model.eval().to(device)
    images = images.to(device)

    _, pred, mask = model(images, mask_ratio=mask_ratio)
    recon_patches = pred

    patches = model.patchify(images)
    mask_exp = mask.unsqueeze(-1).expand_as(patches)
    masked_patches = patches * (1.0 - mask_exp)
    masked = model.unpatchify(masked_patches)

    mix_patches = patches * (1.0 - mask_exp) + recon_patches * mask_exp
    mix = model.unpatchify(mix_patches)

    originals = images.cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    masks = masked.cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    mixes = mix.cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()

    n = originals.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        for j, (img, title) in enumerate(
            zip(
                [originals[i], masks[i], mixes[i]],
                ["original", "masked", "reconstructed"],
                strict=True,
            )
        ):
            axes[i, j].imshow(img.squeeze())
            axes[i, j].set_axis_off()
            if i == 0:
                axes[i, j].set_title(title)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved reconstruction grid: %s", out_path)
    return out_path
