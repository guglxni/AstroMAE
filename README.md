# AstroMAE

[![CI](https://github.com/guglxni/astromae/actions/workflows/ci.yml/badge.svg)](https://github.com/guglxni/astromae/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**AstroMAE** is a reproducible, configuration-driven implementation of a
**Masked Autoencoder** (He et al., 2021) Vision Transformer specialised for
self-supervised pre-training on astrophysical images such as the SDSS
Galaxy10 dataset. It ships with a command-line interface, a downstream
evaluation pipeline (embedding extraction → linear probe / KMeans) and
reconstruction visualisation — all runnable on CPU for smoke tests and on
GPU for real training.

This repository follows the
[AI-DLC](https://github.com/awslabs/aidlc-workflows) (AI-Driven Development
Life Cycle) methodology. The intent, requirements, design and execution
plan that shaped the current iteration live under `aidlc-docs/`.

---

## Highlights

- **ViT-MAE** with configurable depth/dims; `tiny`, `small`, and `base`
  profiles built in.
- **Training loop** with CUDA / Apple MPS / CPU auto-selection, mixed
  precision (AMP) on CUDA, linear-warmup + cosine LR schedule, gradient
  clipping, best/last/periodic checkpointing, and full resume.
- **TensorBoard + CSV logging** out of the box.
- **Downstream evaluation**: frozen-encoder embedding extractor, linear
  logistic-regression probe with macro-F1, KMeans cluster purity + ARI,
  and reconstruction-grid PNGs.
- **Single CLI** (`astromae data|train|eval|info`) and a minimal Python
  API (`from astromae import ...`).
- **CPU-only smoke tests** in CI (pytest + ruff on Python 3.10/3.11/3.12).

---

## Installation

```bash
# Clone and install the package (editable) with dev extras.
git clone https://github.com/guglxni/astromae.git
cd astromae
pip install -e ".[dev,viz,tb]"
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate astromae
```

> ℹ️ `astroNN` is an optional dependency only needed for the Galaxy10
> downloader. Install it with `pip install -e ".[astro]"` or skip it and
> use the synthetic-data path described below.

---

## Quickstart (synthetic CPU run)

```bash
# 1. Generate a tiny synthetic labelled dataset.
astromae data preprocess --synthetic 16 --synthetic-classes 3 \
    --image-size 32 --processed-dir ./data/processed

# 2. Pretrain a tiny MAE on it.
astromae train --config configs/tiny.yaml --epochs 1

# 3. Extract encoder embeddings.
astromae eval embed \
    --checkpoint ./outputs/checkpoints/last.pt \
    --output-dir ./outputs/embeddings

# 4. Run a linear probe + KMeans on the embeddings.
astromae eval probe --embeddings-dir ./outputs/embeddings

# 5. (optional) Save a reconstruction grid.
astromae eval reconstruct \
    --checkpoint ./outputs/checkpoints/last.pt \
    --output ./outputs/reconstruction.png
```

`make smoke` runs all of the above end-to-end.

---

## Real training on Galaxy10

```bash
astromae data download --num-images 10000
astromae data preprocess --image-size 128 --normalize
astromae train --config configs/default.yaml
```

The default config is tuned for a single modern GPU. Tweak `batch_size`,
`epochs`, `image_size`, `embed_dim/depth/...` in `configs/default.yaml`
or pass CLI overrides (`--epochs`, `--batch-size`, `--device`).

---

## Project layout

```text
astromae/
├── __init__.py
├── cli.py                # `astromae ...` entry point
├── data/                 # download + preprocess + dataset
├── models/
│   ├── mae.py            # MaskedAutoencoder + tiny/small/base profiles
│   └── pos_embed.py      # 2D sin-cos positional embeddings
├── training/
│   ├── scheduler.py      # CosineWarmupScheduler
│   └── trainer.py        # Trainer + TrainerConfig
├── evaluation/
│   ├── embed.py          # frozen-encoder feature extraction
│   ├── probe.py          # linear probe + KMeans
│   └── visualize.py      # reconstruction PNG
└── utils/                # seeding, device selection

configs/                  # YAML configs (default + tiny)
aidlc-docs/               # AI-DLC inception artefacts
tests/                    # pytest suite (CPU-only, tiny model)
```

---

## Python API

```python
from astromae import models
from astromae.training import Trainer, TrainerConfig
from astromae.evaluation import (
    extract_embeddings,
    linear_probe,
    load_model_from_checkpoint,
)

# Build and train.
cfg = TrainerConfig.from_yaml("configs/tiny.yaml")
trainer = Trainer(cfg)
trainer.train()

# Evaluate.
model, _ = load_model_from_checkpoint("outputs/checkpoints/last.pt")
emb, y, classes = extract_embeddings(model, "data/processed")
print(linear_probe(emb, y).accuracy)
```

---

## Development

```bash
pip install -e ".[dev]"
pre-commit install
make lint
make test
```

CI runs the full suite on Python 3.10, 3.11 and 3.12 with CPU-only
PyTorch; no real data is downloaded in CI.

---

## AI-DLC

The refinement/upgrade roadmap that produced the current state of this
repo was authored using the
[AI-DLC methodology](https://github.com/awslabs/aidlc-workflows). The
driving documents are:

- [`aidlc-docs/inception/intent.md`](aidlc-docs/inception/intent.md)
- [`aidlc-docs/inception/requirements.md`](aidlc-docs/inception/requirements.md)
- [`aidlc-docs/inception/design.md`](aidlc-docs/inception/design.md)
- [`aidlc-docs/inception/execution-plan.md`](aidlc-docs/inception/execution-plan.md)

---

## References

- He, K., Chen, X., Xie, S., Li, Y., Dollár, P. & Girshick, R. (2021).
  [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).
- Dosovitskiy, A. et al. (2021).
  [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929).
- Leung, H. W. & Bovy, J. (2019).
  [Galaxy10 SDSS dataset via `astroNN`](https://astronn.readthedocs.io/en/latest/galaxy10.html).

## License

MIT. See [LICENSE](LICENSE) if present; otherwise this project is
distributed under the MIT License as declared in `pyproject.toml`.
