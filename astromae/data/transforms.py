"""Astronomical image transforms (asinh stretch, per-band normalisation)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BandStats:
    """Per-band mean / std for normalisation."""

    mean: tuple[float, ...]
    std: tuple[float, ...]


class AsinhStretch:
    """Lupton-style asinh flux stretch.

    Maps raw flux values (which can span many orders of magnitude) into a
    bounded dynamic range without clipping faint structure. Commonly used
    when displaying galaxy images; also helps neural networks see the
    low-surface-brightness regime.

    Args:
        softening: the ``Q`` parameter in ``asinh(flux / softening)``.
            Small values emphasise faint pixels; large values keep the
            transform nearly linear. Default 0.1 suits the synthetic
            profiles.
    """

    def __init__(self, softening: float = 0.1) -> None:
        self.softening = softening

    def __call__(
        self, image: torch.Tensor, ivar: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        stretched = torch.asinh(image / max(self.softening, 1e-12))
        if ivar is None:
            return stretched, None
        # Variance propagation: Var(asinh(x/s)) ≈ (1/(s * sqrt(1+(x/s)^2)))² Var(x)
        s = self.softening
        df_dx = 1.0 / (s * torch.sqrt(1.0 + (image / s) ** 2))
        new_var = torch.where(ivar > 0, 1.0 / ivar, torch.full_like(ivar, float("inf")))
        new_var = df_dx**2 * new_var
        new_ivar = torch.where(
            torch.isfinite(new_var) & (new_var > 0),
            1.0 / new_var,
            torch.zeros_like(ivar),
        )
        return stretched, new_ivar


class BandNormalize:
    """Per-channel mean/std normalisation."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(
        self, image: torch.Tensor, ivar: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean = self.mean.view(-1, 1, 1).to(image)
        std = self.std.view(-1, 1, 1).to(image).clamp(min=1e-6)
        out = (image - mean) / std
        out_ivar = None if ivar is None else ivar * (std**2)
        return out, out_ivar


class Compose:
    """Compose (image, ivar) transforms in order."""

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(
        self, image: torch.Tensor, ivar: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for t in self.transforms:
            image, ivar = t(image, ivar)
        return image, ivar


class RandomFlipRotate90:
    """Random horizontal/vertical flip and C₄ rotation.

    Astronomy-appropriate augmentation: galaxies have no preferred
    orientation. Rotation is restricted to the discrete C₄ group so
    IVAR maps transform without resampling artefacts.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def __call__(
        self, image: torch.Tensor, ivar: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        k = int(torch.randint(0, 4, (1,), generator=self._rng).item())
        image = torch.rot90(image, k=k, dims=(-2, -1))
        if ivar is not None:
            ivar = torch.rot90(ivar, k=k, dims=(-2, -1))

        if torch.rand(1, generator=self._rng).item() < 0.5:
            image = torch.flip(image, dims=(-1,))
            if ivar is not None:
                ivar = torch.flip(ivar, dims=(-1,))
        if torch.rand(1, generator=self._rng).item() < 0.5:
            image = torch.flip(image, dims=(-2,))
            if ivar is not None:
                ivar = torch.flip(ivar, dims=(-2,))

        return image, ivar
