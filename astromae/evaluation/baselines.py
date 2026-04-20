"""Honest baselines for the research-track comparison matrix (M6).

Without matched baselines, any SSL gains are unanchored. We provide:

- ``random_encoder_embeddings``: run an **untrained** MAE over the data
  and extract CLS embeddings. Upper bound on "bag of random features".
- ``ResNet18Baseline``: a supervised ResNet-18 trained end-to-end. The
  natural "is SSL actually worth it?" comparison point.

Both are kept tiny enough to run on CPU for CI smoke experiments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from astromae.models.mae import MaskedAutoencoder
from astromae.utils.device import select_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Random-init MAE embeddings
# ---------------------------------------------------------------------------


def random_encoder_embeddings(
    encoder_factory,
    images: torch.Tensor,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Return CLS embeddings from an *untrained* encoder.

    Args:
        encoder_factory: zero-arg callable returning a MaskedAutoencoder.
        images: (N, C, H, W) tensor on CPU (will be moved to device).
    """
    device = device or select_device()
    model: MaskedAutoencoder = encoder_factory()
    model.eval().to(device)
    with torch.no_grad():
        feats = model.forward_features(images.to(device)).cpu().numpy()
    return feats


# ---------------------------------------------------------------------------
# Supervised ResNet-18 (tiny variant for CPU CI)
# ---------------------------------------------------------------------------


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y = y + self.down(x)
        return F.relu(y, inplace=True)


class ResNet18Baseline(nn.Module):
    """Small ResNet-18 style classifier.

    Not a re-implementation of the canonical torchvision ResNet-18 at
    224x224 ImageNet resolution; this is a sized-down variant for
    galaxy-cutout inputs (64-128 px). Matches the ViT-tiny baseline's
    parameter budget to within a factor of a few so the comparison with
    SSL encoders is at least within the spirit of RESEARCH-03.
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(_BasicBlock(32, 64, 2), _BasicBlock(64, 64))
        self.layer2 = nn.Sequential(_BasicBlock(64, 128, 2), _BasicBlock(128, 128))
        self.layer3 = nn.Sequential(_BasicBlock(128, 128, 2), _BasicBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.pool(x).flatten(1)


@dataclass
class SupervisedMetrics:
    accuracy: float
    macro_f1: float
    epochs: int
    params: int


class _TensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def train_resnet18_baseline(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    device: Optional[torch.device] = None,
    val_fraction: float = 0.2,
) -> tuple[ResNet18Baseline, SupervisedMetrics]:
    """Train a ResNet-18 baseline and report held-out metrics."""
    from sklearn.metrics import f1_score

    device = device or select_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = images.shape[0]
    perm = rng.permutation(n)
    split = int(n * (1.0 - val_fraction))
    tr_idx = perm[:split]
    va_idx = perm[split:]

    x_tr = images[tr_idx].to(device)
    y_tr = labels[tr_idx].to(device)
    x_va = images[va_idx].to(device)
    y_va = labels[va_idx].to(device)

    model = ResNet18Baseline(in_chans=images.shape[1], num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        _TensorDataset(x_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(x_va).argmax(dim=-1).cpu().numpy()
        y_np = y_va.cpu().numpy()

    accuracy = float((preds == y_np).mean())
    macro_f1 = float(f1_score(y_np, preds, average="macro"))
    params = sum(p.numel() for p in model.parameters())
    metrics = SupervisedMetrics(
        accuracy=accuracy, macro_f1=macro_f1, epochs=epochs, params=params
    )
    logger.info(
        "ResNet-18 baseline: acc=%.4f macro_f1=%.4f params=%d epochs=%d",
        accuracy,
        macro_f1,
        params,
        epochs,
    )
    return model, metrics
