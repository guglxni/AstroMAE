"""Downstream evaluation: linear probe and KMeans clustering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    accuracy: float
    macro_f1: float
    report: dict[str, dict[str, float]]


def linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    max_iter: int = 2000,
) -> ProbeResult:
    """Train logistic regression on embeddings and return a :class:`ProbeResult`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(labels)) < 2:
        raise ValueError("Need at least 2 classes for a linear probe.")

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=max_iter, n_jobs=None)
    clf.fit(x_train_s, y_train)
    y_pred = clf.predict(x_test_s)

    acc = float((y_pred == y_test).mean())
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    logger.info("Linear probe: acc=%.4f macro_f1=%.4f", acc, f1)
    return ProbeResult(accuracy=acc, macro_f1=f1, report=report)  # type: ignore[arg-type]


def cluster_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Run KMeans and return cluster purity + adjusted Rand index."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    preds = km.fit_predict(embeddings)

    purity = 0.0
    for c in np.unique(preds):
        members = labels[preds == c]
        if members.size == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        purity += counts.max()
    purity /= len(labels)

    ari = float(adjusted_rand_score(labels, preds))
    logger.info("KMeans (k=%d): purity=%.4f ARI=%.4f", n_clusters, purity, ari)
    return {"purity": float(purity), "ari": ari, "n_clusters": float(n_clusters)}
