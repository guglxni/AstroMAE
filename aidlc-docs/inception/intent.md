# AstroMAE — Intent

Authored following the [AI-DLC](https://github.com/awslabs/aidlc-workflows)
methodology during the refinement pass.

## One-sentence description

AstroMAE is a reproducible, configuration-driven implementation of a
**Masked Autoencoder (MAE)** Vision Transformer specialised for
self-supervised pre-training on astrophysical images (e.g. SDSS
Galaxy10, Kepler/TESS cutouts), together with tools to extract the
learned representations and evaluate them on downstream tasks such as
morphological classification and clustering.

## Why it exists

Self-supervised learning is a natural fit for astrophysics: modern
surveys produce orders of magnitude more unlabelled images than
labelled ones. MAE (He et al., 2021) has shown that masking-then-
reconstructing pixel patches yields transferable visual features at
low compute cost. AstroMAE's goal is to make it trivial for an
astrophysics researcher to:

1. Download a public galaxy-image dataset.
2. Pretrain a ViT encoder on it with masked reconstruction.
3. Extract embeddings and run a lightweight downstream evaluation
   (linear probe accuracy, KMeans purity, t-SNE/UMAP visualisation).

## In scope

- ViT-B/16 and ViT-S/8 MAE backbones with configurable depth/dims.
- 1–3 channel (RGB / narrow-band) image input at arbitrary square size.
- Galaxy10 (SDSS) reference dataset pipeline (download + preprocess +
  per-channel stats).
- Training loop with: CUDA / Apple MPS / CPU auto-selection, mixed
  precision (AMP), cosine LR schedule with linear warmup, gradient
  clipping, resume-from-checkpoint, best-validation-loss checkpointing,
  CSV + TensorBoard logging.
- Evaluation: frozen-encoder embedding extraction, linear probe (logistic
  regression), KMeans clustering with cluster purity, reconstruction
  visualisation.
- CPU-runnable smoke tests and CI.
- Installable Python package (`astromae`) with a small CLI.

## Explicitly out of scope (for this iteration)

- Multi-node distributed training (single-node multi-GPU via
  `torchrun` is out of scope too — to be revisited).
- Kepler/TESS light-curve 1-D support (README mentions it as a future
  direction only).
- Hyperparameter sweeping, experiment tracking via W&B, model export
  (ONNX / TorchScript).
- Pretrained weight releases.

## Primary users

- ML-literate astrophysics researchers who want a clean starting point.
- Students reproducing MAE-style self-supervised pretraining on
  scientific images.

## Success criteria

- `pip install -e .` succeeds and `pytest` passes on CPU-only CI.
- A full end-to-end smoke run (download → preprocess → 1 epoch train
  → embedding extraction → linear probe) completes in under 5 minutes
  on CPU with a downsized ViT.
- The model's encoder produces embeddings whose linear-probe accuracy
  on Galaxy10 classes is meaningfully above random (baseline: >25% on
  10 classes; target after proper pretrain on GPU: >60%).
