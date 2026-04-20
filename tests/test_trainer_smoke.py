"""CPU-only end-to-end smoke test for the trainer + evaluation pipeline."""

from pathlib import Path

import numpy as np
import pytest
import torch

from astromae.data.preprocess import generate_synthetic_dataset
from astromae.evaluation.embed import (
    extract_embeddings,
    load_model_from_checkpoint,
    save_embeddings,
)
from astromae.evaluation.probe import cluster_embeddings, linear_probe
from astromae.training.trainer import Trainer, TrainerConfig


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    processed = root / "processed"
    generate_synthetic_dataset(
        processed,
        num_classes=2,
        per_class=8,
        image_size=32,
        in_chans=3,
        seed=7,
    )
    return processed


def _tiny_config(processed: Path, output: Path) -> TrainerConfig:
    return TrainerConfig(
        dataset_dir=str(processed),
        image_size=32,
        patch_size=16,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2.0,
        batch_size=4,
        val_ratio=0.25,
        num_workers=0,
        epochs=1,
        mask_ratio=0.5,
        learning_rate=1e-3,
        warmup_epochs=0,
        grad_clip=1.0,
        checkpoint_dir=str(output / "ckpts"),
        log_dir=str(output),
        save_every=1,
        amp=False,
        tensorboard=False,
        log_interval=1,
        seed=0,
    )


def test_trainer_runs_on_cpu_and_saves_checkpoints(tiny_dataset, tmp_path):
    cfg = _tiny_config(tiny_dataset, tmp_path / "out")
    trainer = Trainer(cfg)
    summary = trainer.train()

    assert (Path(cfg.checkpoint_dir) / "last.pt").exists()
    assert (Path(cfg.checkpoint_dir) / "best.pt").exists()
    assert (Path(cfg.log_dir) / "train_log.csv").exists()
    assert summary["epochs_completed"] == 1
    assert np.isfinite(summary["best_val_loss"])


def test_resume_restores_epoch(tiny_dataset, tmp_path):
    out = tmp_path / "out"
    cfg = _tiny_config(tiny_dataset, out)
    Trainer(cfg).train()

    # Resume from last and verify start_epoch is loaded.
    cfg2 = _tiny_config(tiny_dataset, out)
    cfg2.epochs = 2
    t2 = Trainer(cfg2)
    t2.load_checkpoint(Path(cfg.checkpoint_dir) / "last.pt")
    assert t2.start_epoch == 1


def test_embeddings_and_linear_probe(tiny_dataset, tmp_path):
    cfg = _tiny_config(tiny_dataset, tmp_path / "out")
    Trainer(cfg).train()

    ckpt = Path(cfg.checkpoint_dir) / "last.pt"
    model, _ = load_model_from_checkpoint(ckpt, device=torch.device("cpu"))
    emb, labels, classes = extract_embeddings(model, dataset_dir=str(tiny_dataset), batch_size=4)

    assert emb.shape[0] == 16
    assert emb.shape[1] == cfg.embed_dim
    assert set(labels.tolist()) == {0, 1}
    assert classes == ["class_0", "class_1"]

    save_embeddings(tmp_path / "emb", emb, labels, classes)
    assert (tmp_path / "emb" / "embeddings.npy").exists()

    # The synthetic dataset has a class-mean bias so even an untrained encoder
    # should beat random.
    probe = linear_probe(emb, labels, test_size=0.5, seed=0)
    assert 0.0 <= probe.accuracy <= 1.0
    assert probe.accuracy >= 0.5

    clusters = cluster_embeddings(emb, labels, seed=0)
    assert 0.0 <= clusters["purity"] <= 1.0
