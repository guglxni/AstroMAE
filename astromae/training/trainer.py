"""MAE trainer with AMP, cosine-warmup LR, checkpointing, and resume."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from astromae.data.dataset import create_train_val_dataloaders
from astromae.models.mae import MaskedAutoencoder
from astromae.training.scheduler import CosineWarmupScheduler
from astromae.utils.device import select_device
from astromae.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    # data
    dataset_dir: str = "./data/processed"
    image_size: int = 128
    batch_size: int = 8
    val_ratio: float = 0.1
    num_workers: int = 0
    in_memory: bool = False

    # model
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: float = 4.0

    # optimization
    mask_ratio: float = 0.75
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.01
    grad_clip: float | None = 1.0

    # runtime
    checkpoint_dir: str = "./outputs/checkpoints"
    log_dir: str = "./outputs"
    save_every: int = 5
    amp: bool = True
    device: str | None = None
    seed: int | None = 42
    tensorboard: bool = True
    log_interval: int = 50

    # misc
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: os.PathLike | str) -> TrainerConfig:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        if extra:
            kwargs["extra"] = extra
        return cls(**kwargs)


class Trainer:
    """High-level MAE trainer."""

    def __init__(self, config: TrainerConfig) -> None:
        self.cfg = config
        seed_everything(self.cfg.seed)

        self.device = select_device(self.cfg.device)
        logger.info("Using device: %s", self.device)

        self.ckpt_dir = Path(self.cfg.checkpoint_dir)
        self.log_dir = Path(self.cfg.log_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "train_log.csv"

        self.train_loader, self.val_loader = create_train_val_dataloaders(
            dataset_dir=self.cfg.dataset_dir,
            batch_size=self.cfg.batch_size,
            val_ratio=self.cfg.val_ratio,
            num_workers=self.cfg.num_workers,
            image_size=self.cfg.image_size,
            seed=self.cfg.seed or 42,
            in_memory=self.cfg.in_memory,
            return_labels=False,
        )

        self.model = MaskedAutoencoder(
            img_size=self.cfg.image_size,
            patch_size=self.cfg.patch_size,
            in_chans=self.cfg.in_chans,
            embed_dim=self.cfg.embed_dim,
            depth=self.cfg.depth,
            num_heads=self.cfg.num_heads,
            decoder_embed_dim=self.cfg.decoder_embed_dim,
            decoder_depth=self.cfg.decoder_depth,
            decoder_num_heads=self.cfg.decoder_num_heads,
            mlp_ratio=self.cfg.mlp_ratio,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95),
        )

        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = steps_per_epoch * self.cfg.epochs
        warmup_steps = steps_per_epoch * self.cfg.warmup_epochs
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )

        self.amp_enabled = bool(self.cfg.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.tb_writer = None
        if self.cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
            except ImportError:  # pragma: no cover
                logger.warning("TensorBoard not installed; skipping TB logs.")

        self.start_epoch = 0
        self.best_val_loss = math.inf
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # checkpoint I/O
    # ------------------------------------------------------------------
    def _checkpoint_state(self, epoch: int, avg_loss: float) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": avg_loss,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.cfg),
        }

    def _save(self, path: Path, state: dict[str, Any]) -> None:
        torch.save(state, path)
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: os.PathLike | str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.start_epoch = state["epoch"]
        self.best_val_loss = state.get("best_val_loss", math.inf)
        logger.info("Resumed from %s at epoch %d", path, self.start_epoch)

    # ------------------------------------------------------------------
    # train / validate
    # ------------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        logger.info(
            "Starting training: %d epochs, %d train batches, %d val batches",
            self.cfg.epochs,
            len(self.train_loader),
            len(self.val_loader) if self.val_loader else 0,
        )

        for epoch in range(self.start_epoch, self.cfg.epochs):
            train_loss = self._run_epoch(epoch, self.train_loader, train=True)
            val_loss = (
                self._run_epoch(epoch, self.val_loader, train=False)
                if self.val_loader is not None
                else float("nan")
            )
            lr = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
            }
            self.history.append(row)
            pd.DataFrame(self.history).to_csv(self.log_file, index=False)
            logger.info(
                "Epoch %d | train %.4f | val %.4f | lr %.2e",
                epoch + 1,
                train_loss,
                val_loss,
                lr,
            )
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("loss/train_epoch", train_loss, epoch + 1)
                if not math.isnan(val_loss):
                    self.tb_writer.add_scalar("loss/val_epoch", val_loss, epoch + 1)
                self.tb_writer.add_scalar("lr", lr, epoch + 1)

            state = self._checkpoint_state(epoch + 1, train_loss)
            self._save(self.ckpt_dir / "last.pt", state)

            effective_val = val_loss if not math.isnan(val_loss) else train_loss
            if effective_val < self.best_val_loss:
                self.best_val_loss = effective_val
                state["best_val_loss"] = self.best_val_loss
                self._save(self.ckpt_dir / "best.pt", state)

            if self.cfg.save_every > 0 and (epoch + 1) % self.cfg.save_every == 0:
                self._save(self.ckpt_dir / f"epoch_{epoch + 1}.pt", state)

        summary = {
            "epochs_completed": self.cfg.epochs,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.history[-1]["train_loss"] if self.history else None,
        }
        summary_path = self.log_dir / "train_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Training complete. Summary: %s", summary)
        if self.tb_writer is not None:
            self.tb_writer.close()
        return summary

    def _run_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        train: bool,
    ) -> float:
        mode = "train" if train else "val"
        self.model.train(train)
        start = time.time()
        total_loss = 0.0
        n_batches = 0

        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=self.amp_enabled)
            if self.device.type == "cuda"
            else torch.amp.autocast("cpu", enabled=False)
        )

        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(self.device, non_blocking=True)
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train), autocast_ctx:
                loss, _, _ = self.model(imgs, mask_ratio=self.cfg.mask_ratio)

            if train:
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

            loss_val = float(loss.detach().item())
            if math.isfinite(loss_val):
                total_loss += loss_val
                n_batches += 1

            if train and self.cfg.log_interval > 0 and batch_idx % self.cfg.log_interval == 0:
                logger.info(
                    "  [%s] epoch %d batch %d/%d loss %.4f",
                    mode,
                    epoch + 1,
                    batch_idx + 1,
                    len(loader),
                    loss_val,
                )
                if self.tb_writer is not None:
                    global_step = epoch * len(loader) + batch_idx
                    self.tb_writer.add_scalar("loss/train_step", loss_val, global_step)

        elapsed = time.time() - start
        avg = total_loss / max(1, n_batches)
        logger.debug("%s epoch %d done in %.1fs (avg %.4f)", mode, epoch + 1, elapsed, avg)
        return avg
