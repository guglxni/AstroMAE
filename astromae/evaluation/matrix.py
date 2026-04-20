"""Experiment matrix runner.

Produces the headline table from ``aidlc-docs/research/evaluation-
protocol.md`` §1-2 on a (small, CPU-friendly) synthetic dataset. The
full paper-grade matrix runs the same code on real-data loaders.

The matrix currently covers:
  - `mae` vs `mae-iv` vs `mae-c4` vs `mae-c4-iv`
  - `jepa` vs `mae`
  - Morphology linear probe as the primary downstream metric.
  - Rotation-equivariance error as an intrinsic diagnostic.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from astromae.data.synth import SURVEY_A, make_synthetic_cutout
from astromae.evaluation.heads import morphology_probe
from astromae.models import (
    c4_equivariance_error,
    mae_c4_tiny,
    mae_tiny,
)
# noqa: E501 — IJEPA is imported lazily inside _run_jepa_variant to avoid
# circular imports when this module is used as a smoke test.

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    tag: str
    morphology_accuracy: float
    morphology_macro_f1: float
    equivariance_error: float
    wall_clock_sec: float
    param_count: int
    seed: int


def _build_dataset(
    n_per_class: int = 32,
    num_classes: int = 2,
    size: int = 32,
    bands: tuple[str, ...] = ("g", "r", "z"),
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = []
    ivars = []
    labels = []
    for c in range(num_classes):
        for _ in range(n_per_class):
            img, iv, _ = make_synthetic_cutout(
                size=size, bands=bands, survey=SURVEY_A, class_idx=c, rng=rng
            )
            images.append(img)
            ivars.append(iv)
            labels.append(c)
    img_t = torch.from_numpy(np.stack(images, axis=0)).float()
    iv_t = torch.from_numpy(np.stack(ivars, axis=0)).float()
    y = np.asarray(labels, dtype=np.int64)
    return img_t, iv_t, y


def _train_mae(
    model, images: torch.Tensor, ivar: torch.Tensor | None, steps: int, lr: float
) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(7)
    t0 = time.time()
    for _ in range(steps):
        batch_idx = torch.randint(0, images.shape[0], (min(16, images.shape[0]),))
        xb = images[batch_idx]
        iv = ivar[batch_idx] if ivar is not None else None
        loss, _, _ = model(xb, mask_ratio=0.75, generator=g, ivar=iv)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return time.time() - t0


def _train_jepa(model, images: torch.Tensor, steps: int, lr: float) -> float:
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    g = torch.Generator().manual_seed(7)
    t0 = time.time()
    for _ in range(steps):
        batch_idx = torch.randint(0, images.shape[0], (min(16, images.shape[0]),))
        xb = images[batch_idx]
        out = model(xb, generator=g)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        model._ema_update()
    return time.time() - t0


def _encode(model, images: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model.forward_features(images).cpu().numpy()


def _param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _run_mae_variant(
    tag: str,
    factory: Callable,
    images: torch.Tensor,
    ivar: torch.Tensor,
    labels: np.ndarray,
    seed: int,
    steps: int,
    lr: float,
    use_ivar: bool,
) -> RunResult:
    torch.manual_seed(seed)
    model = factory()
    wall = _train_mae(model, images, ivar if use_ivar else None, steps=steps, lr=lr)
    emb = _encode(model, images)
    probe = morphology_probe(emb, labels, seed=seed)
    err = float(c4_equivariance_error(model, images[: min(8, images.shape[0])]))
    return RunResult(
        tag=tag,
        morphology_accuracy=probe.accuracy,
        morphology_macro_f1=probe.macro_f1,
        equivariance_error=err,
        wall_clock_sec=wall,
        param_count=_param_count(model),
        seed=seed,
    )


def _run_jepa_variant(
    tag: str,
    images: torch.Tensor,
    labels: np.ndarray,
    seed: int,
    steps: int,
    lr: float,
) -> RunResult:
    from astromae.models import ijepa_tiny

    torch.manual_seed(seed)
    model = ijepa_tiny(img_size=images.shape[-1], patch_size=16, in_chans=images.shape[1])
    wall = _train_jepa(model, images, steps=steps, lr=lr)
    emb = _encode(model, images)
    probe = morphology_probe(emb, labels, seed=seed)
    # Equivariance diagnostic via the target encoder's features — we
    # wrap it in a small adapter so c4_equivariance_error can call
    # forward_features.
    err = float(c4_equivariance_error(model, images[: min(8, images.shape[0])]))
    return RunResult(
        tag=tag,
        morphology_accuracy=probe.accuracy,
        morphology_macro_f1=probe.macro_f1,
        equivariance_error=err,
        wall_clock_sec=wall,
        param_count=_param_count(model),
        seed=seed,
    )


def run_experiment_matrix(
    output_dir: Path | str,
    n_per_class: int = 32,
    num_classes: int = 2,
    size: int = 32,
    seed: int = 0,
    steps: int = 40,
    lr: float = 3e-3,
) -> list[RunResult]:
    """Run the small-scale ablation matrix and write outputs.

    Produces ``results.json`` and ``results.md`` in ``output_dir``.
    Intended for CI smoke; for paper-grade numbers, run the same
    function with larger synthetic datasets or real data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, ivar, labels = _build_dataset(
        n_per_class=n_per_class, num_classes=num_classes, size=size, seed=seed
    )

    in_chans = images.shape[1]
    img_size = images.shape[-1]

    def _mae_tiny_factory():
        return mae_tiny(img_size=img_size, patch_size=16, in_chans=in_chans)

    def _mae_c4_factory():
        return mae_c4_tiny(img_size=img_size, patch_size=16, in_chans=in_chans)

    results: list[RunResult] = []
    results.append(
        _run_mae_variant(
            "mae",
            _mae_tiny_factory,
            images,
            ivar,
            labels,
            seed=seed,
            steps=steps,
            lr=lr,
            use_ivar=False,
        )
    )
    results.append(
        _run_mae_variant(
            "mae-iv",
            _mae_tiny_factory,
            images,
            ivar,
            labels,
            seed=seed,
            steps=steps,
            lr=lr,
            use_ivar=True,
        )
    )
    results.append(
        _run_mae_variant(
            "mae-c4",
            _mae_c4_factory,
            images,
            ivar,
            labels,
            seed=seed,
            steps=steps,
            lr=lr,
            use_ivar=False,
        )
    )
    results.append(
        _run_mae_variant(
            "mae-c4-iv",
            _mae_c4_factory,
            images,
            ivar,
            labels,
            seed=seed,
            steps=steps,
            lr=lr,
            use_ivar=True,
        )
    )
    results.append(
        _run_jepa_variant("jepa", images, labels, seed=seed, steps=steps, lr=lr)
    )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    md_lines = [
        "# Experiment matrix (synthetic pilot)",
        "",
        "| tag | acc | macro-F1 | C4 err | params | wall (s) |",
        "|-----|-----|----------|--------|--------|----------|",
    ]
    for r in results:
        md_lines.append(
            f"| {r.tag} | {r.morphology_accuracy:.3f} | {r.morphology_macro_f1:.3f} | "
            f"{r.equivariance_error:.3e} | {r.param_count} | {r.wall_clock_sec:.2f} |"
        )
    (output_dir / "results.md").write_text("\n".join(md_lines) + "\n")
    logger.info("Wrote results to %s", output_dir)
    return results
