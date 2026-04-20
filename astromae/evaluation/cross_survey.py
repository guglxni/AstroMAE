"""Cross-survey evaluation protocol (Contribution C4, M7).

Protocol:
  1. Pretrain encoder on Survey A.
  2. Train a morphology probe on Survey A embeddings.
  3. Evaluate the probe on Survey B images encoded by the same encoder.
  4. Report the accuracy drop vs. a Survey-B-only probe.

Shipped as a utility function that drives the already-existing
morphology_probe and extract_embeddings paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from astromae.evaluation.heads import ProbeMetrics, morphology_probe

logger = logging.getLogger(__name__)


@dataclass
class CrossSurveyMetrics:
    same_survey: ProbeMetrics
    cross_survey: ProbeMetrics
    accuracy_drop: float


def cross_survey_probe(
    encoder_fn: Callable[[torch.Tensor], np.ndarray],
    images_a: torch.Tensor,
    labels_a: np.ndarray,
    images_b: torch.Tensor,
    labels_b: np.ndarray,
    seed: int = 42,
) -> CrossSurveyMetrics:
    """Train a probe on survey A, evaluate on survey B.

    Args:
        encoder_fn: callable taking ``(N, C, H, W)`` and returning
            ``(N, D)`` float32 embeddings. Typically ``lambda x:
            encoder.forward_features(x).cpu().numpy()`` for a frozen
            encoder.
        images_a, labels_a: survey A images and integer labels (train).
        images_b, labels_b: survey B images and integer labels (eval).

    Returns:
        :class:`CrossSurveyMetrics` with same-survey and cross-survey
        accuracy and the gap between them.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    emb_a = encoder_fn(images_a)
    emb_b = encoder_fn(images_b)

    same = morphology_probe(emb_b, labels_b, seed=seed)

    scaler = StandardScaler().fit(emb_a)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(emb_a), labels_a)
    preds = clf.predict(scaler.transform(emb_b))
    cross_acc = float((preds == labels_b).mean())
    cross_f1 = float(f1_score(labels_b, preds, average="macro"))
    cross = ProbeMetrics(accuracy=cross_acc, macro_f1=cross_f1)

    drop = same.accuracy - cross.accuracy
    logger.info(
        "Cross-survey probe: same=%.4f, cross=%.4f, drop=%.4f",
        same.accuracy,
        cross.accuracy,
        drop,
    )
    return CrossSurveyMetrics(
        same_survey=same, cross_survey=cross, accuracy_drop=drop
    )
