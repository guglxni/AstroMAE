"""``astromae`` unified command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from astromae import __version__

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ---------------------------------------------------------------------------
# subcommand handlers
# ---------------------------------------------------------------------------
def _cmd_data_download(args: argparse.Namespace) -> int:
    from astromae.data.download import download_and_save_galaxy10

    download_and_save_galaxy10(num_images=args.num_images, save_dir=args.output_dir)
    return 0


def _cmd_data_preprocess(args: argparse.Namespace) -> int:
    from astromae.data.preprocess import (
        generate_synthetic_dataset,
        preprocess_images,
    )

    if args.synthetic > 0:
        generate_synthetic_dataset(
            args.processed_dir,
            num_classes=args.synthetic_classes,
            per_class=args.synthetic,
            image_size=args.image_size,
        )
    else:
        preprocess_images(
            args.raw_dir,
            args.processed_dir,
            image_size=args.image_size,
            normalize=args.normalize,
        )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from astromae.training.trainer import Trainer, TrainerConfig

    cfg = TrainerConfig.from_yaml(args.config) if args.config else TrainerConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.dataset_dir is not None:
        cfg.dataset_dir = args.dataset_dir
    if args.output_dir is not None:
        cfg.log_dir = args.output_dir
        cfg.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")

    trainer = Trainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()
    return 0


def _cmd_eval_embed(args: argparse.Namespace) -> int:
    from astromae.evaluation.embed import (
        extract_embeddings,
        load_model_from_checkpoint,
        save_embeddings,
    )

    model, cfg = load_model_from_checkpoint(args.checkpoint)
    dataset_dir = args.dataset_dir or cfg.dataset_dir
    embeddings, labels, classes = extract_embeddings(
        model,
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
    )
    out_dir = save_embeddings(args.output_dir, embeddings, labels, classes)
    logger.info("Saved embeddings to %s", out_dir)
    return 0


def _cmd_eval_probe(args: argparse.Namespace) -> int:
    import numpy as np

    from astromae.evaluation.probe import cluster_embeddings, linear_probe

    embeddings = np.load(Path(args.embeddings_dir) / "embeddings.npy")
    labels = np.load(Path(args.embeddings_dir) / "labels.npy")

    probe = linear_probe(embeddings, labels, seed=args.seed)
    clusters = cluster_embeddings(embeddings, labels, seed=args.seed)

    report = {
        "linear_probe": {
            "accuracy": probe.accuracy,
            "macro_f1": probe.macro_f1,
            "per_class": probe.report,
        },
        "clustering": clusters,
    }
    out_path = Path(args.embeddings_dir) / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        "Probe: acc=%.4f, macro_f1=%.4f | KMeans purity=%.4f ARI=%.4f",
        probe.accuracy,
        probe.macro_f1,
        clusters["purity"],
        clusters["ari"],
    )
    logger.info("Report written to %s", out_path)
    return 0


def _cmd_eval_reconstruct(args: argparse.Namespace) -> int:
    import torch

    from astromae.data.dataset import PreprocessedAstroDataset
    from astromae.evaluation.embed import load_model_from_checkpoint
    from astromae.evaluation.visualize import save_reconstruction_grid

    model, cfg = load_model_from_checkpoint(args.checkpoint)
    dataset = PreprocessedAstroDataset(args.dataset_dir or cfg.dataset_dir)
    if len(dataset) == 0:
        logger.error("Empty dataset.")
        return 1
    n = min(args.num_samples, len(dataset))
    imgs = torch.stack([dataset[i][0] for i in range(n)], dim=0)
    save_reconstruction_grid(model, imgs, out_path=args.output, mask_ratio=cfg.mask_ratio)
    return 0


def _cmd_experiment_matrix(args: argparse.Namespace) -> int:
    from astromae.evaluation.matrix import run_experiment_matrix

    results = run_experiment_matrix(
        args.output_dir,
        n_per_class=args.n_per_class,
        num_classes=args.num_classes,
        size=args.image_size,
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
    )
    logger.info("Matrix complete: %d rows written to %s", len(results), args.output_dir)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from astromae.training.trainer import TrainerConfig
    from astromae.utils.device import select_device

    print(f"astromae v{__version__}")
    print(f"device: {select_device()}")
    if args.config:
        cfg = TrainerConfig.from_yaml(args.config)
        print("config:")
        print(json.dumps(asdict(cfg), indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astromae",
        description="AstroMAE: Masked Autoencoders for astrophysical images.",
    )
    parser.add_argument("--version", action="version", version=f"astromae {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- data ---
    p_data = sub.add_parser("data", help="Dataset utilities").add_subparsers(
        dest="data_command", required=True
    )

    p_dl = p_data.add_parser("download", help="Download Galaxy10 (SDSS)")
    p_dl.add_argument("--num-images", type=int, default=1000)
    p_dl.add_argument("--output-dir", type=Path, default=Path("./data/raw"))
    p_dl.set_defaults(func=_cmd_data_download)

    p_pp = p_data.add_parser("preprocess", help="Tensorise raw images")
    p_pp.add_argument("--raw-dir", type=Path, default=Path("./data/raw"))
    p_pp.add_argument("--processed-dir", type=Path, default=Path("./data/processed"))
    p_pp.add_argument("--image-size", type=int, default=128)
    p_pp.add_argument("--normalize", action="store_true")
    p_pp.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="If > 0, write synthetic tensors instead of processing raw images.",
    )
    p_pp.add_argument("--synthetic-classes", type=int, default=2)
    p_pp.set_defaults(func=_cmd_data_preprocess)

    # --- train ---
    p_train = sub.add_parser("train", help="Run MAE pretraining")
    p_train.add_argument("--config", type=Path, default=Path("config.yaml"))
    p_train.add_argument("--dataset-dir", type=str, default=None)
    p_train.add_argument("--output-dir", type=str, default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--resume", type=Path, default=None)
    p_train.set_defaults(func=_cmd_train)

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluation utilities").add_subparsers(
        dest="eval_command", required=True
    )

    p_embed = p_eval.add_parser("embed", help="Extract encoder embeddings")
    p_embed.add_argument("--checkpoint", type=Path, required=True)
    p_embed.add_argument("--dataset-dir", type=str, default=None)
    p_embed.add_argument("--output-dir", type=Path, default=Path("./outputs/embeddings"))
    p_embed.add_argument("--batch-size", type=int, default=64)
    p_embed.set_defaults(func=_cmd_eval_embed)

    p_probe = p_eval.add_parser("probe", help="Linear probe + KMeans on embeddings")
    p_probe.add_argument("--embeddings-dir", type=Path, required=True)
    p_probe.add_argument("--seed", type=int, default=42)
    p_probe.set_defaults(func=_cmd_eval_probe)

    p_recon = p_eval.add_parser("reconstruct", help="Save a reconstruction grid")
    p_recon.add_argument("--checkpoint", type=Path, required=True)
    p_recon.add_argument("--dataset-dir", type=str, default=None)
    p_recon.add_argument("--num-samples", type=int, default=6)
    p_recon.add_argument("--output", type=Path, default=Path("./outputs/reconstruction.png"))
    p_recon.set_defaults(func=_cmd_eval_reconstruct)

    # --- experiment matrix (research-track) ---
    p_matrix = sub.add_parser(
        "experiment-matrix",
        help=(
            "Run the research-track ablation matrix (mae / mae-iv / mae-c4 / "
            "mae-c4-iv / jepa) on synthetic Sersic cutouts; "
            "produces results.json + results.md"
        ),
    )
    p_matrix.add_argument("--output-dir", type=Path, default=Path("./outputs/experiment_matrix"))
    p_matrix.add_argument("--n-per-class", type=int, default=32)
    p_matrix.add_argument("--num-classes", type=int, default=2)
    p_matrix.add_argument("--image-size", type=int, default=32)
    p_matrix.add_argument("--steps", type=int, default=40)
    p_matrix.add_argument("--lr", type=float, default=3e-3)
    p_matrix.add_argument("--seed", type=int, default=0)
    p_matrix.set_defaults(func=_cmd_experiment_matrix)

    # --- info ---
    p_info = sub.add_parser("info", help="Print environment / config info")
    p_info.add_argument("--config", type=Path, default=None)
    p_info.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
