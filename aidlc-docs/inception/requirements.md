# AstroMAE — Requirements

## Functional requirements

- FR-01 **Data acquisition**: `astromae data download` downloads the
  Galaxy10 SDSS dataset and persists raw PNGs organised by class.
- FR-02 **Preprocessing**: `astromae data preprocess` resizes to a
  configurable square size, converts to tensor in `[0, 1]`, optionally
  applies per-channel mean/std normalisation computed from a subset of
  the data, and persists `.pt` tensors under `data/processed/<class>/`.
- FR-03 **Train / Val split**: the dataset loader supports a
  deterministic train/validation split controlled by `val_ratio` and
  a random seed.
- FR-04 **MAE pretraining**: `astromae train --config config.yaml`
  pretrains a ViT-MAE model with masked reconstruction and logs
  per-batch and per-epoch metrics.
- FR-05 **Checkpointing**: the trainer writes an `epoch_<N>.pt`
  checkpoint every `save_every` epochs, always writes `last.pt`, and
  writes `best.pt` whenever a new lowest validation loss is observed.
- FR-06 **Resume**: `astromae train --resume <ckpt>` restores model,
  optimizer, scheduler, AMP scaler and epoch count.
- FR-07 **Embedding extraction**: `astromae eval embed` runs the
  frozen encoder with `mask_ratio=0` over a dataset and writes
  `embeddings.npy`, `labels.npy`, `paths.txt`.
- FR-08 **Downstream eval**: `astromae eval probe` trains a logistic
  regression probe on the embeddings and reports accuracy and F1;
  `astromae eval cluster` runs KMeans and reports cluster purity.
- FR-09 **Reconstruction viz**: `astromae eval reconstruct` produces a
  PNG grid of (original, masked, reconstructed) samples.

## Non-functional requirements

- NFR-01 **Device portability**: automatically selects CUDA → MPS →
  CPU in that order; all code paths must work on CPU with a downscaled
  model so that CI can run them.
- NFR-02 **Reproducibility**: a single `--seed` flag seeds Python,
  NumPy and torch RNGs; masking uses a per-sample RNG derived from
  that seed.
- NFR-03 **Configurability**: all knobs (model shape, optimizer,
  schedule, data) live in a single YAML file; CLI overrides via
  `--set key=value` (optional, stretch).
- NFR-04 **Installability**: the project is an installable Python
  package (`pip install -e .`) with a declared entry point
  `astromae = astromae.cli:main`.
- NFR-05 **Quality gates**: CI runs `ruff` (lint + format),
  `pytest` (unit + smoke), and uploads coverage. No deprecated GitHub
  Actions.
- NFR-06 **No unnecessary network in CI**: tests must not download
  real datasets; they synthesise tiny random tensors on the fly.

## Constraints / assumptions

- Python 3.11, PyTorch 2.x, timm ≥ 1.0.
- Dataset assumption: images are square-ish, RGB; `preprocess.py`
  centre-crops/resizes to a configurable size.
- Running on Apple Silicon: mixed precision on MPS is limited; AMP is
  only enabled on CUDA.
