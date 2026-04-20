"""Downstream scientific heads on frozen SSL embeddings.

Each head takes ``(embeddings, targets)`` and returns a fitted predictor
plus metrics. All heads work on any encoder that exposes
``forward_features`` — vanilla MAE, C4-MAE and I-JEPA share the same
interface.

Heads
-----
- ``MorphologyProbe``: linear probe on classification labels (we already
  have this; thin wrapper for uniformity).
- ``PhotoZMDN``: photometric redshift via a mixture density network head.
- ``AnomalyProbe``: Isolation Forest on embeddings; reports AUROC
  against labelled anomalies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------


@dataclass
class ProbeMetrics:
    accuracy: float
    macro_f1: float


def morphology_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    max_iter: int = 2000,
) -> ProbeMetrics:
    """Thin wrapper around sklearn LogisticRegression for morphology
    classification."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(labels)) < 2:
        raise ValueError("Need at least 2 classes for a morphology probe.")

    x_tr, x_te, y_tr, y_te = train_test_split(
        embeddings, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(scaler.transform(x_tr), y_tr)
    y_pred = clf.predict(scaler.transform(x_te))
    return ProbeMetrics(
        accuracy=float((y_pred == y_te).mean()),
        macro_f1=float(f1_score(y_te, y_pred, average="macro")),
    )


# ---------------------------------------------------------------------------
# Photo-z — Mixture Density Network
# ---------------------------------------------------------------------------


@dataclass
class PhotoZMetrics:
    nmad: float
    outlier_fraction: float
    crps: float
    pit_mean: float
    pit_std: float


class PhotoZMDN(nn.Module):
    """Mixture density network head for photometric redshift.

    Predicts parameters of a K-Gaussian mixture over z from a frozen
    embedding. Reports σ_NMAD, outlier fraction (|Δz / (1+z)| > 0.15),
    CRPS, and PIT calibration statistics.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_components: int = 5,
        z_min: float = 0.0,
        z_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.z_min = z_min
        self.z_max = z_max
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_mu = nn.Linear(hidden_dim, n_components)
        self.head_sigma = nn.Linear(hidden_dim, n_components)
        self.head_pi = nn.Linear(hidden_dim, n_components)

    def forward(
        self, z_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(z_embedding)
        mu = torch.sigmoid(self.head_mu(h)) * (self.z_max - self.z_min) + self.z_min
        sigma = F.softplus(self.head_sigma(h)) + 1e-3
        pi_logits = self.head_pi(h)
        return mu, sigma, pi_logits

    @staticmethod
    def _mixture_nll(
        z_true: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        pi_logits: torch.Tensor,
    ) -> torch.Tensor:
        log_pi = F.log_softmax(pi_logits, dim=-1)
        # log N(z | mu, sigma^2)
        log_prob = -0.5 * ((z_true.unsqueeze(-1) - mu) / sigma) ** 2 - torch.log(
            sigma * math.sqrt(2 * math.pi)
        )
        return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()


def _crps_mixture_gaussian(
    z_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    pi: np.ndarray,
    n_grid: int = 256,
) -> float:
    """Numerical CRPS for a Gaussian mixture posterior.

    CRPS(F, y) = integral (F(z) - 1{y <= z})^2 dz. We evaluate on a
    z-grid spanning z_true.min-3 to z_true.max+3.
    """
    from scipy.stats import norm

    z_grid = np.linspace(z_true.min() - 1.0, z_true.max() + 1.0, n_grid)
    out = []
    for i in range(z_true.shape[0]):
        cdf = np.zeros_like(z_grid)
        for k in range(mu.shape[1]):
            cdf += pi[i, k] * norm.cdf(z_grid, loc=mu[i, k], scale=sigma[i, k])
        heaviside = (z_grid >= z_true[i]).astype(np.float32)
        out.append(np.trapezoid((cdf - heaviside) ** 2, z_grid))
    return float(np.mean(out))


def _pit_mixture_gaussian(
    z_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    pi: np.ndarray,
) -> np.ndarray:
    from scipy.stats import norm

    pit = np.zeros(z_true.shape[0])
    for i in range(z_true.shape[0]):
        for k in range(mu.shape[1]):
            pit[i] += pi[i, k] * norm.cdf(z_true[i], loc=mu[i, k], scale=sigma[i, k])
    return pit


def train_photoz_mdn(
    embeddings: np.ndarray,
    redshifts: np.ndarray,
    n_components: int = 5,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
    device: Optional[str] = None,
    z_range: Optional[tuple[float, float]] = None,
) -> tuple[PhotoZMDN, PhotoZMetrics]:
    """Fit a :class:`PhotoZMDN` and return it with held-out metrics."""
    from sklearn.model_selection import train_test_split

    device_ = torch.device(device) if device else torch.device("cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    x_tr, x_te, y_tr, y_te = train_test_split(
        embeddings.astype(np.float32),
        redshifts.astype(np.float32),
        test_size=test_size,
        random_state=seed,
    )
    zmin, zmax = z_range or (float(y_tr.min()), float(y_tr.max()) + 1e-3)

    model = PhotoZMDN(
        input_dim=x_tr.shape[1],
        hidden_dim=hidden_dim,
        n_components=n_components,
        z_min=zmin,
        z_max=zmax,
    ).to(device_)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_tr_t = torch.from_numpy(x_tr).to(device_)
    y_tr_t = torch.from_numpy(y_tr).to(device_)

    n = x_tr_t.shape[0]
    for _ in range(epochs):
        idx = torch.from_numpy(rng.permutation(n)).long().to(device_)
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            mu, sigma, pi_logits = model(x_tr_t[batch])
            loss = PhotoZMDN._mixture_nll(y_tr_t[batch], mu, sigma, pi_logits)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        mu, sigma, pi_logits = model(torch.from_numpy(x_te).to(device_))
        pi = torch.softmax(pi_logits, dim=-1)
        z_point = (pi * mu).sum(dim=-1).cpu().numpy()
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
        pi_np = pi.cpu().numpy()

    delta = (z_point - y_te) / (1.0 + y_te)
    nmad = 1.4826 * float(np.median(np.abs(delta - np.median(delta))))
    outlier = float(np.mean(np.abs(delta) > 0.15))
    crps = _crps_mixture_gaussian(y_te, mu_np, sigma_np, pi_np)
    pit = _pit_mixture_gaussian(y_te, mu_np, sigma_np, pi_np)

    metrics = PhotoZMetrics(
        nmad=nmad,
        outlier_fraction=outlier,
        crps=crps,
        pit_mean=float(np.mean(pit)),
        pit_std=float(np.std(pit)),
    )
    logger.info(
        "PhotoZ MDN: nmad=%.4f, outliers=%.3f, CRPS=%.4f, PIT=%.3f+-%.3f",
        metrics.nmad,
        metrics.outlier_fraction,
        metrics.crps,
        metrics.pit_mean,
        metrics.pit_std,
    )
    return model, metrics


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


@dataclass
class AnomalyMetrics:
    auroc: float
    recall_at_top_k: float
    top_k: int


def anomaly_probe(
    embeddings: np.ndarray,
    is_anomaly: np.ndarray,
    top_k: int = 10,
    contamination: float = 0.1,
    seed: int = 42,
) -> AnomalyMetrics:
    """Train IsolationForest and evaluate against labelled anomalies.

    Args:
        embeddings: (N, D) frozen embeddings.
        is_anomaly: (N,) binary labels — 1 for anomaly, 0 for normal.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score

    if len(np.unique(is_anomaly)) < 2:
        raise ValueError("Need at least one anomaly and one normal sample.")

    clf = IsolationForest(
        contamination=contamination, random_state=seed, n_estimators=100
    )
    clf.fit(embeddings)
    scores = -clf.score_samples(embeddings)
    auroc = float(roc_auc_score(is_anomaly, scores))

    order = np.argsort(-scores)
    top_k_eff = min(top_k, len(order))
    top_idx = order[:top_k_eff]
    recall_top = float(is_anomaly[top_idx].sum() / is_anomaly.sum())
    return AnomalyMetrics(auroc=auroc, recall_at_top_k=recall_top, top_k=top_k_eff)
