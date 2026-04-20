"""Frozen-encoder embedding extraction."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from astromae.data.dataset import PreprocessedAstroDataset
from astromae.models.mae import MaskedAutoencoder
from astromae.training.trainer import TrainerConfig
from astromae.utils.device import select_device

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    ckpt_path: os.PathLike | str,
    device: torch.device | None = None,
) -> tuple[MaskedAutoencoder, TrainerConfig]:
    """Instantiate a ``MaskedAutoencoder`` and load weights from ``ckpt_path``."""
    device = device or select_device()
    state = torch.load(ckpt_path, map_location=device)
    cfg_dict: dict = state.get("config") or {}
    known = {f.name for f in TrainerConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    cfg = TrainerConfig(**{k: v for k, v in cfg_dict.items() if k in known})

    model = MaskedAutoencoder(
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_chans=cfg.in_chans,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_depth=cfg.decoder_depth,
        decoder_num_heads=cfg.decoder_num_heads,
        mlp_ratio=cfg.mlp_ratio,
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    logger.info("Loaded MAE checkpoint from %s", ckpt_path)
    logger.debug("Config: %s", asdict(cfg))
    return model, cfg


@torch.no_grad()
def extract_embeddings(
    model: MaskedAutoencoder,
    dataset_dir: os.PathLike | str,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the frozen encoder and return ``(embeddings, labels, class_names)``.

    ``mask_ratio`` is implicitly zero — we use ``forward_features``.
    """
    device = device or select_device()
    dataset = PreprocessedAstroDataset(root_dir=dataset_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {dataset_dir}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval().to(device)
    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = model.forward_features(imgs).cpu().numpy()
        all_feats.append(feats)
        all_labels.append(np.asarray(labels))

    embeddings = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    logger.info(
        "Extracted embeddings: %s (dtype=%s) with %d classes",
        embeddings.shape,
        embeddings.dtype,
        len(dataset.classes),
    )
    return embeddings, y, dataset.classes


def save_embeddings(
    out_dir: os.PathLike | str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)
    (out_dir / "class_names.txt").write_text("\n".join(class_names))
    return out_dir
