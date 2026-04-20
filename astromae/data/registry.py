"""Dataset / survey registry for cross-survey experiments.

A minimal in-memory registry that maps a survey name to its canonical
band list, pixel scale, normalisation statistics, and a loader function.
Used by the cross-survey evaluation protocol (aidlc-docs/research/
evaluation-protocol.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SurveySpec:
    name: str
    bands: tuple[str, ...]
    pixel_scale_arcsec: float
    normalisation: dict[str, tuple[float, ...]]  # {"mean": (...), "std": (...)}
    loader: Callable | None = None
    metadata: dict[str, str] = field(default_factory=dict)


_REGISTRY: dict[str, SurveySpec] = {}


def register(spec: SurveySpec) -> None:
    _REGISTRY[spec.name] = spec


def get(name: str) -> SurveySpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown survey '{name}'. Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_surveys() -> list[str]:
    return sorted(_REGISTRY.keys())


# --- built-in entries -------------------------------------------------------
register(
    SurveySpec(
        name="synth_A",
        bands=("g", "r", "z"),
        pixel_scale_arcsec=0.262,  # DECaLS-like
        normalisation={"mean": (0.0, 0.0, 0.0), "std": (0.3, 0.3, 0.3)},
        metadata={"source": "astromae.data.synth.SURVEY_A"},
    )
)
register(
    SurveySpec(
        name="synth_B",
        bands=("g", "r", "z"),
        pixel_scale_arcsec=0.168,  # HSC-like
        normalisation={"mean": (0.02, 0.0, -0.03), "std": (0.35, 0.35, 0.35)},
        metadata={"source": "astromae.data.synth.SURVEY_B"},
    )
)
register(
    SurveySpec(
        name="decals_dr10",
        bands=("g", "r", "z"),
        pixel_scale_arcsec=0.262,
        normalisation={"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
        metadata={
            "source": "legacysurvey.org cutout service",
            "note": "placeholder stats; compute from real pilot sample before use.",
        },
    )
)
register(
    SurveySpec(
        name="hsc_pdr3",
        bands=("g", "r", "i", "z", "y"),
        pixel_scale_arcsec=0.168,
        normalisation={
            "mean": (0.0, 0.0, 0.0, 0.0, 0.0),
            "std": (1.0, 1.0, 1.0, 1.0, 1.0),
        },
        metadata={"source": "HSC-SSP PDR3", "note": "placeholder"},
    )
)
