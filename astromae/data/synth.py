"""Synthetic FITS galaxy cutouts for CI, smoke tests, and pilot experiments.

No external data download required: we generate Sersic-profile galaxies
with configurable PSF, noise model and survey-profile, and write them as
multi-extension FITS files (image + IVAR + MASK) so the rest of the
stack sees something indistinguishable from a real cutout.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from astromae.data.fits import save_fits_cutout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurveyProfile:
    """Parameters characterising a (synthetic) survey.

    These are the axes that actually matter for a cross-survey SSL benchmark:
    PSF FWHM, sky noise, pixel scale, band set, optional calibration offset.
    """

    name: str
    bands: tuple[str, ...] = ("g", "r", "z")
    psf_sigma_px: float = 1.0
    sky_sigma: float = 0.05  # Gaussian sky noise
    poisson_gain: float = 10.0  # counts-per-flux; higher -> more Poisson-ish
    pixel_scale_jitter: float = 0.0  # ignored in v1
    calibration_offset: tuple[float, ...] = (0.0, 0.0, 0.0)


SURVEY_A = SurveyProfile(
    name="synth_A",
    bands=("g", "r", "z"),
    psf_sigma_px=1.0,
    sky_sigma=0.05,
    poisson_gain=10.0,
)
SURVEY_B = SurveyProfile(
    name="synth_B",
    bands=("g", "r", "z"),
    psf_sigma_px=1.5,
    sky_sigma=0.15,
    poisson_gain=7.0,
    calibration_offset=(0.02, 0.0, -0.03),
)


def _sersic_2d(
    size: int,
    center: tuple[float, float],
    r_eff: float,
    n: float,
    flux: float,
    ellip: float = 0.0,
    pa: float = 0.0,
) -> np.ndarray:
    """Compute a Sersic profile on a grid. Minimal, not astropy-perfect."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    xc, yc = center
    dx = x - xc
    dy = y - yc

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    xr = dx * cos_pa + dy * sin_pa
    yr = -dx * sin_pa + dy * cos_pa

    q = max(1.0 - ellip, 1e-3)
    r = np.sqrt(xr**2 + (yr / q) ** 2)

    bn = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)  # Ciotti & Bertin 1999
    intensity = np.exp(-bn * ((r / max(r_eff, 1e-6)) ** (1.0 / max(n, 0.1)) - 1.0))
    total = intensity.sum()
    if total <= 0:
        return intensity.astype(np.float32)
    return (intensity * (flux / total)).astype(np.float32)


def _gaussian_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    if size is None:
        size = int(max(3, round(6 * sigma + 1)))
        if size % 2 == 0:
            size += 1
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax**2) / (2.0 * sigma**2))
    g /= g.sum()
    return np.outer(g, g).astype(np.float32)


def _convolve_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from numpy.fft import fft2, ifft2

    h, w = image.shape
    kh, kw = kernel.shape
    ph, pw = h + kh - 1, w + kw - 1
    ip = np.zeros((ph, pw), dtype=np.float32)
    kp = np.zeros((ph, pw), dtype=np.float32)
    ip[:h, :w] = image
    kp[:kh, :kw] = kernel
    conv = np.real(ifft2(fft2(ip) * fft2(kp)))
    y0 = (kh - 1) // 2
    x0 = (kw - 1) // 2
    return conv[y0 : y0 + h, x0 : x0 + w].astype(np.float32)


def make_synthetic_cutout(
    size: int = 64,
    bands: Sequence[str] = ("g", "r", "z"),
    survey: SurveyProfile = SURVEY_A,
    class_idx: int = 0,
    rng: np.random.Generator | None = None,
    include_lsb_arc: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(image, ivar, meta)`` arrays for a single synthetic cutout.

    The two-class synthetic distinction used in tests:
      - class_idx=0: elliptical-like (n=4, low ellip)
      - class_idx=1: disk-like (n=1, higher ellip)
      - class_idx>=2: perturbed variants (for k-class setups)

    If ``include_lsb_arc`` is True, a low-surface-brightness arc is added
    in the outer annulus. Used by the IVAR-loss physics test.
    """
    if rng is None:
        rng = np.random.default_rng()

    flux = rng.uniform(5.0, 15.0)
    r_eff = rng.uniform(size * 0.10, size * 0.20)
    center = (size / 2.0 + rng.normal(0, 0.5), size / 2.0 + rng.normal(0, 0.5))

    if class_idx == 0:
        n, ellip = 4.0, 0.1
    elif class_idx == 1:
        n, ellip = 1.0, 0.4
    else:
        n = float(rng.uniform(0.8, 4.5))
        ellip = float(rng.uniform(0.0, 0.5))

    pa = float(rng.uniform(0, np.pi))

    band_brightness = np.linspace(0.9, 1.1, len(bands))
    if rng.random() > 0.5:
        band_brightness = band_brightness[::-1]

    image = np.zeros((len(bands), size, size), dtype=np.float32)
    clean = np.zeros_like(image)

    for c, _band in enumerate(bands):
        galaxy = _sersic_2d(size, center, r_eff, n, flux * band_brightness[c], ellip, pa)
        if include_lsb_arc:
            # Add a faint arc at r_eff * 2.2
            yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
            r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
            theta = np.arctan2(yy - center[1], xx - center[0])
            arc_r = r_eff * 2.2
            arc = (
                0.1
                * flux
                * np.exp(-((r - arc_r) ** 2) / (2.0 * (size * 0.03) ** 2))
                * np.exp(-((theta - pa) ** 2) / (2.0 * 0.25**2))
            )
            galaxy = galaxy + arc.astype(np.float32)
        clean[c] = galaxy

    psf = _gaussian_kernel(survey.psf_sigma_px)
    for c in range(image.shape[0]):
        conv = _convolve_fft(clean[c], psf)
        # Poisson + Gaussian noise
        counts = np.maximum(conv * survey.poisson_gain, 0.0)
        poisson = rng.poisson(counts).astype(np.float32) / survey.poisson_gain
        gaussian = rng.normal(0.0, survey.sky_sigma, size=conv.shape).astype(np.float32)
        offset = survey.calibration_offset[c] if c < len(survey.calibration_offset) else 0.0
        image[c] = poisson + gaussian + offset

    # Per-pixel variance ~ shot noise + sky
    expected = np.maximum(clean, 0.0)
    var = expected / max(survey.poisson_gain, 1e-6) + survey.sky_sigma**2
    ivar = (1.0 / np.maximum(var, 1e-8)).astype(np.float32)

    meta = {
        "class_idx": class_idx,
        "flux": float(flux),
        "r_eff": float(r_eff),
        "n": float(n),
        "ellip": float(ellip),
        "survey": survey.name,
        "include_lsb_arc": bool(include_lsb_arc),
    }
    return image, ivar, meta


def make_synthetic_fits_dataset(
    output_dir: Path | str,
    num_classes: int = 2,
    per_class: int = 8,
    size: int = 64,
    bands: Sequence[str] = ("g", "r", "z"),
    survey: SurveyProfile = SURVEY_A,
    seed: int = 0,
    clean: bool = True,
) -> int:
    """Write ``num_classes * per_class`` synthetic FITS cutouts to disk."""
    output_dir = Path(output_dir)
    if clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    count = 0
    for c in range(num_classes):
        class_dir = output_dir / f"class_{c}"
        class_dir.mkdir(exist_ok=True)
        for i in range(per_class):
            image, ivar, _meta = make_synthetic_cutout(
                size=size,
                bands=bands,
                survey=survey,
                class_idx=c,
                rng=rng,
            )
            save_fits_cutout(
                class_dir / f"sample_{i:04d}.fits",
                image=image,
                ivar=ivar,
                bands=bands,
                header={"TARGET": f"synth_{c}_{i}", "SURVEY": survey.name},
            )
            count += 1
    logger.info("Wrote %d synthetic FITS cutouts to %s", count, output_dir)
    return count


def make_synthetic_pair_for_lsb_test(
    size: int = 64,
    bands: Sequence[str] = ("g", "r", "z"),
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Paired cutouts with and without a low-surface-brightness arc.

    Used by :func:`tests.test_ivar_loss_lsb` to verify that IVAR-weighted
    reconstruction recovers the arc better than unweighted MSE.
    """
    rng = np.random.default_rng(seed)
    image, ivar, _ = make_synthetic_cutout(
        size=size,
        bands=bands,
        survey=SURVEY_A,
        class_idx=0,
        rng=rng,
        include_lsb_arc=True,
    )
    return {
        "image": torch.from_numpy(image).float(),
        "ivar": torch.from_numpy(ivar).float(),
    }
