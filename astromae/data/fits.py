"""FITS cutout loading for AstroMAE.

Loads multi-band FITS cutouts with optional ``IVAR`` / ``WEIGHT`` / ``MASK``
planes. The returned dictionary always contains the three keys ``image``,
``ivar``, ``meta`` so downstream code can rely on a uniform interface
regardless of whether the underlying file ships a variance plane.

Notes on FITS conventions used here
-----------------------------------
- Multi-band images are expected as multi-extension FITS (MEF) where each
  SCIENCE HDU has an ``EXTNAME`` matching the band (``'g'``, ``'r'``,
  ``'z'``, ``'i'`` …) — this is the Legacy Survey cutout convention.
  Alternatively, a single primary HDU of shape ``(C, H, W)`` is accepted.
- IVAR planes are recognised by ``EXTNAME`` ∈ {``'IVAR'``, ``'WEIGHT'``,
  ``'VAR'``} — the last one is converted to inverse variance with a
  clamp.
- A MASK plane (``'MASK'`` / ``'FLAGS'``) is optional; non-zero values
  are treated as bad pixels and have their IVAR zeroed out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

_IVAR_EXTNAMES = {"IVAR", "WEIGHT", "IVAR_IMAGE"}
_VAR_EXTNAMES = {"VAR", "VARIANCE"}
_MASK_EXTNAMES = {"MASK", "FLAGS", "DQ"}


@dataclass(frozen=True)
class FITSCutout:
    """In-memory representation of a multi-band FITS cutout."""

    image: torch.Tensor  # (C, H, W), float32
    ivar: torch.Tensor  # (C, H, W), float32, non-negative
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        if self.image.shape != self.ivar.shape:
            raise ValueError(
                f"image and ivar shape mismatch: {self.image.shape} vs {self.ivar.shape}"
            )
        if self.image.ndim != 3:
            raise ValueError(f"expected (C, H, W), got {tuple(self.image.shape)}")


def _lazy_astropy():
    try:
        from astropy.io import fits  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "astropy is required for FITS I/O. Install with `pip install astropy`."
        ) from e
    return fits


def _stack_bands(
    hdul, bands: Sequence[str], data_key: str | None = None
) -> np.ndarray | None:
    """Return (C, H, W) for a list of band EXTNAMEs, or None if not found.

    ``data_key=None`` matches the plain band name (e.g. ``G``); otherwise only
    the suffixed name (e.g. ``G_IVAR``) is matched.
    """
    arrays: list[np.ndarray] = []
    for band in bands:
        target_name = band.upper() if data_key is None else f"{band.upper()}_{data_key.upper()}"
        hdu = None
        for h in hdul:
            if (h.name or "").upper() == target_name:
                hdu = h
                break
        if hdu is None or hdu.data is None:
            return None
        arrays.append(np.asarray(hdu.data, dtype=np.float32))
    if not arrays:
        return None
    return np.stack(arrays, axis=0)


def _load_from_primary(hdul, bands: Sequence[str]) -> np.ndarray:
    """Fallback: assume the primary HDU holds a (C, H, W) cube."""
    data = np.asarray(hdul[0].data)
    if data.ndim == 2:
        data = data[None, :, :]
    if data.ndim != 3:
        raise ValueError(f"Unsupported primary HDU shape: {data.shape}")
    if data.shape[0] != len(bands):
        # Tolerate mismatch: crop to requested bands if more are provided.
        logger.warning(
            "Primary HDU has %d channels but %d bands requested; truncating.",
            data.shape[0],
            len(bands),
        )
        data = data[: len(bands)]
    return data.astype(np.float32)


def _find_single_plane(hdul, names: set[str]) -> np.ndarray | None:
    for h in hdul:
        hname = (h.name or "").upper()
        if hname in names and h.data is not None:
            return np.asarray(h.data)
    return None


def load_fits_cutout(
    path: Path | str,
    bands: Sequence[str] = ("g", "r", "z"),
    ivar_floor: float = 1e-12,
) -> FITSCutout:
    """Load a FITS cutout as an :class:`FITSCutout`.

    Args:
        path: path to the FITS file.
        bands: band names in the order to stack along the channel axis.
        ivar_floor: clamp IVAR to ``max(ivar, 0)`` and replace NaN/Inf;
            this is the floor for numerical stability in the loss.
    """
    fits = _lazy_astropy()
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        image = _stack_bands(hdul, bands, data_key=None)
        if image is None:
            image = _load_from_primary(hdul, bands)

        ivar = _stack_bands(hdul, bands, data_key="IVAR")
        if ivar is None:
            ivar = _find_single_plane(hdul, _IVAR_EXTNAMES)
            if ivar is not None and ivar.ndim == 2:
                ivar = np.broadcast_to(ivar[None, :, :], image.shape).copy()

        if ivar is None:
            var = _find_single_plane(hdul, _VAR_EXTNAMES)
            if var is not None:
                var = np.where(var > 0, var, np.inf)
                ivar = (1.0 / var).astype(np.float32)
                if ivar.ndim == 2:
                    ivar = np.broadcast_to(ivar[None, :, :], image.shape).copy()

        mask = _find_single_plane(hdul, _MASK_EXTNAMES)

        meta: dict[str, Any] = {"path": str(path), "bands": list(bands)}
        header = hdul[0].header
        for key in ("TARGET", "OBJID", "RA", "DEC", "Z_SPEC", "ZSPEC"):
            if key in header:
                meta[key.lower()] = header[key]

    if ivar is None:
        ivar = np.ones_like(image, dtype=np.float32)
    else:
        ivar = ivar.astype(np.float32)

    if mask is not None:
        if mask.ndim == 2:
            mask = np.broadcast_to(mask[None, :, :], image.shape).copy()
        ivar = np.where(mask == 0, ivar, 0.0).astype(np.float32)

    ivar = np.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
    ivar = np.maximum(ivar, ivar_floor).astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return FITSCutout(
        image=torch.from_numpy(image),
        ivar=torch.from_numpy(ivar),
        meta=meta,
    )


def save_fits_cutout(
    path: Path | str,
    image: np.ndarray | torch.Tensor,
    ivar: Optional[np.ndarray | torch.Tensor] = None,
    bands: Sequence[str] = ("g", "r", "z"),
    header: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a (C, H, W) image (and optional IVAR) to a multi-extension FITS file."""
    fits = _lazy_astropy()
    image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if image_np.ndim != 3 or image_np.shape[0] != len(bands):
        raise ValueError(
            f"expected image shape (C={len(bands)}, H, W); got {image_np.shape}"
        )

    hdul = fits.HDUList([fits.PrimaryHDU()])
    if header:
        for k, v in header.items():
            hdul[0].header[str(k)[:8].upper()] = v

    for band, arr in zip(bands, image_np, strict=True):
        h = fits.ImageHDU(arr.astype(np.float32), name=band.upper())
        hdul.append(h)

    if ivar is not None:
        iv = ivar.cpu().numpy() if isinstance(ivar, torch.Tensor) else np.asarray(ivar)
        if iv.shape != image_np.shape:
            raise ValueError(f"ivar shape {iv.shape} != image shape {image_np.shape}")
        for band, arr in zip(bands, iv, strict=True):
            hdul.append(fits.ImageHDU(arr.astype(np.float32), name=f"{band.upper()}_IVAR"))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(path, overwrite=True)
    return path


class FITSCutoutDataset(torch.utils.data.Dataset):
    """Dataset that reads FITS cutouts and returns ``(image, ivar)`` pairs.

    Labels, when available, are inferred from the parent directory (matching
    the existing ``PreprocessedAstroDataset`` contract).
    """

    def __init__(
        self,
        root_dir: Path | str,
        bands: Sequence[str] = ("g", "r", "z"),
        pattern: str = "**/*.fits",
        return_ivar: bool = True,
        transform=None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.bands = tuple(bands)
        self.return_ivar = return_ivar
        self.transform = transform

        self.paths: list[Path] = sorted(self.root_dir.glob(pattern))
        if not self.paths:
            logger.warning("No FITS files found under %s with pattern %s", root_dir, pattern)

        class_names = sorted({p.parent.name for p in self.paths})
        self.classes = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        cutout = load_fits_cutout(path, bands=self.bands)
        image, ivar = cutout.image, cutout.ivar
        if self.transform is not None:
            image, ivar = self.transform(image, ivar)
        label = self.class_to_idx.get(path.parent.name, -1)
        if self.return_ivar:
            return {"image": image, "ivar": ivar, "label": label}
        return image, label
