"""Tests for the scientific downstream heads (Contribution C5, M5)."""

from __future__ import annotations

import numpy as np
import pytest

from astromae.evaluation.heads import (
    anomaly_probe,
    morphology_probe,
    train_photoz_mdn,
)


def _synthetic_emb_regression(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.05, 1.5, size=n).astype(np.float32)
    direction = rng.normal(size=d).astype(np.float32)
    emb = (
        np.outer(z, direction)
        + 0.2 * rng.normal(size=(n, d)).astype(np.float32)
    ).astype(np.float32)
    return emb, z


def test_morphology_probe_recovers_class_bias():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 8)).astype(np.float32)
    labels = rng.integers(0, 2, size=60)
    emb[labels == 1] += 1.5  # class-dependent offset
    metrics = morphology_probe(emb, labels, seed=0)
    assert 0.6 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_morphology_probe_rejects_single_class():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(40, 8)).astype(np.float32)
    labels = np.zeros(40, dtype=int)
    with pytest.raises(ValueError):
        morphology_probe(emb, labels, seed=0)


def test_photoz_mdn_trains_and_reports_finite_metrics():
    emb, z = _synthetic_emb_regression(n=160, d=8, seed=0)
    _, m = train_photoz_mdn(
        emb,
        z,
        hidden_dim=32,
        n_components=3,
        lr=5e-3,
        epochs=60,
        batch_size=32,
        seed=0,
    )
    assert np.isfinite(m.nmad)
    assert 0.0 <= m.outlier_fraction <= 1.0
    assert m.crps > 0
    assert 0.0 <= m.pit_mean <= 1.0
    assert m.pit_std >= 0
    # Weak bound — the synthetic dataset is deliberately easy, so the
    # σ_NMAD should be well under the raw redshift scatter.
    assert m.nmad < 0.4


def test_photoz_mdn_outperforms_random_embeddings():
    emb_good, z = _synthetic_emb_regression(n=160, d=8, seed=1)
    rng = np.random.default_rng(2)
    emb_random = rng.normal(size=emb_good.shape).astype(np.float32)

    _, m_good = train_photoz_mdn(
        emb_good,
        z,
        hidden_dim=32,
        n_components=3,
        lr=5e-3,
        epochs=60,
        batch_size=32,
        seed=0,
    )
    _, m_rand = train_photoz_mdn(
        emb_random,
        z,
        hidden_dim=32,
        n_components=3,
        lr=5e-3,
        epochs=60,
        batch_size=32,
        seed=0,
    )
    assert m_good.nmad <= m_rand.nmad + 1e-3


def test_anomaly_probe_detects_injected_anomalies():
    rng = np.random.default_rng(0)
    n = 200
    emb = rng.normal(size=(n, 8)).astype(np.float32)
    is_anomaly = np.zeros(n, dtype=int)
    # Inject 10 anomalies: large-norm vectors.
    anomaly_idx = rng.choice(n, size=10, replace=False)
    emb[anomaly_idx] = rng.normal(size=(10, 8)).astype(np.float32) * 10 + 15
    is_anomaly[anomaly_idx] = 1
    metrics = anomaly_probe(emb, is_anomaly, top_k=20, contamination=0.05, seed=0)
    assert metrics.auroc >= 0.8
    assert metrics.recall_at_top_k >= 0.5
    assert metrics.top_k == 20


def test_anomaly_probe_rejects_single_class():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 4)).astype(np.float32)
    is_anomaly = np.zeros(30, dtype=int)
    with pytest.raises(ValueError):
        anomaly_probe(emb, is_anomaly, seed=0)
