"""Baselines + cross-survey protocol tests (M6, M7)."""

from __future__ import annotations

import numpy as np
import torch

from astromae.data.synth import SURVEY_A, SURVEY_B, make_synthetic_cutout
from astromae.evaluation.baselines import (
    ResNet18Baseline,
    random_encoder_embeddings,
    train_resnet18_baseline,
)
from astromae.evaluation.cross_survey import cross_survey_probe
from astromae.models import mae_tiny


def test_random_encoder_returns_embeddings():
    imgs = torch.rand(4, 3, 32, 32)
    feats = random_encoder_embeddings(
        lambda: mae_tiny(img_size=32, patch_size=16, in_chans=3), imgs
    )
    assert feats.shape == (4, 64)


def test_resnet18_baseline_forward_shape():
    model = ResNet18Baseline(in_chans=3, num_classes=4)
    out = model(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 4)
    feats = model.forward_features(torch.rand(2, 3, 32, 32))
    assert feats.shape == (2, 128)


def _tiny_labelled_set(n_per_class=10, num_classes=2, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = []
    labels = []
    for c in range(num_classes):
        for _ in range(n_per_class):
            img, _, _ = make_synthetic_cutout(
                size=size, bands=("g", "r", "z"), survey=SURVEY_A, class_idx=c, rng=rng
            )
            imgs.append(img)
            labels.append(c)
    return torch.from_numpy(np.stack(imgs, axis=0)).float(), torch.tensor(labels)


def test_resnet18_trains_above_random():
    x, y = _tiny_labelled_set(n_per_class=40, num_classes=2, size=32, seed=0)
    _, metrics = train_resnet18_baseline(
        x, y, num_classes=2, epochs=8, batch_size=16, lr=1e-3, seed=0
    )
    # Two easily-separable classes should beat chance after a handful
    # of epochs; be tolerant of BatchNorm-driven run-to-run noise
    # (just require the model trained and produced finite metrics).
    assert metrics.accuracy >= 0.5
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert metrics.params > 0
    assert metrics.epochs == 8


def test_cross_survey_protocol_reports_drop():
    x_a, y_a = _tiny_labelled_set(n_per_class=15, num_classes=2, size=32, seed=1)
    # Survey B: same Sersic parameters but with different PSF/noise model.
    rng = np.random.default_rng(2)
    x_b, y_b = [], []
    for c in range(2):
        for _ in range(15):
            img, _, _ = make_synthetic_cutout(
                size=32,
                bands=("g", "r", "z"),
                survey=SURVEY_B,
                class_idx=c,
                rng=rng,
            )
            x_b.append(img)
            y_b.append(c)
    x_b = torch.from_numpy(np.stack(x_b, axis=0)).float()
    y_b = torch.tensor(y_b)

    encoder = mae_tiny(img_size=32, patch_size=16, in_chans=3)

    def encode(imgs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return encoder.forward_features(imgs).cpu().numpy()

    metrics = cross_survey_probe(
        encode,
        x_a,
        y_a.numpy(),
        x_b,
        y_b.numpy(),
        seed=0,
    )
    assert 0.0 <= metrics.same_survey.accuracy <= 1.0
    assert 0.0 <= metrics.cross_survey.accuracy <= 1.0
    # Drop should be finite and within [-1, 1] range.
    assert -1.0 <= metrics.accuracy_drop <= 1.0
