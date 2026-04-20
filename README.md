# AstroMAE

[![CI](https://github.com/guglxni/astromae/actions/workflows/ci.yml/badge.svg)](https://github.com/guglxni/astromae/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**AstroMAE** is a research-track self-supervised learning library for
galaxy imaging. It ships:

- **Vanilla ViT-MAE** (He et al. 2021) with a clean CPU-runnable
  reference implementation.
- **IVAR-weighted MAE** — a new, physically correct reconstruction loss
  that uses the per-pixel inverse-variance maps shipped in every FITS
  cutout (novel; no published astronomical MAE does this).
- **C₄-equivariant ViT-MAE** — approximately rotation-equivariant patch
  embedding + polar positional embedding for galaxies (first
  equivariant ViT-MAE for astronomy; E(2)-CNNs exist, equivariant
  ViT-SSL does not).
- **I-JEPA** — a feature-space alternative to pixel reconstruction, for
  the head-to-head MAE vs JEPA benchmark on noisy astronomical images
  that is missing from the literature.
- **Scientific downstream heads** — photometric-redshift Mixture
  Density Network with σ_NMAD / CRPS / PIT, morphology linear probe,
  IsolationForest anomaly detection.
- **Honest baselines + cross-survey evaluation protocol** —
  random-init encoder, supervised ResNet-18, and a
  pretrain-on-A-probe-on-B protocol for foundation-model-grade
  transfer reporting.

The repository follows the
[AI-DLC](https://github.com/awslabs/aidlc-workflows) methodology. Two
tracks of documentation live under `aidlc-docs/`:

1. `aidlc-docs/inception/` — the software engineering scaffold.
2. `aidlc-docs/research/` — the research-track: intent, lit review,
   falsifiable novel contributions, execution plan, and a full
   evaluation protocol.
3. `aidlc-docs/research-extension/` — a new AI-DLC extension
   (`research-rigor`) that adds 10 blocking rules enforcing lit-review
   gates, matched-compute baselines, ablation matrices,
   calibration-aware metrics, seed stability, data provenance,
   reproducibility bundles, and negative-result honesty.

---

## Install

```bash
git clone https://github.com/guglxni/astromae.git
cd astromae
pip install -e ".[dev,viz,tb,fits]"
```

Conda:

```bash
conda env create -f environment.yml
conda activate astromae
```

---

## Quickstart — the research-track matrix in 10 seconds

```bash
astromae experiment-matrix --output-dir ./outputs/matrix \
  --n-per-class 16 --num-classes 2 --image-size 32 \
  --steps 8 --seed 0
cat ./outputs/matrix/results.md
```

This runs five pretraining objectives on synthetic Sérsic-profile
galaxy cutouts (including IVAR noise model), trains a morphology
linear probe on each, and measures intrinsic rotation equivariance:

```
| tag        | acc   | macro-F1 | C4 err      | params | wall (s) |
| mae        | 0.429 | 0.417    | 2.575e-02   | 152896 | 0.04     |
| mae-iv     | 0.429 | 0.417    | 2.472e-02   | 152896 | 0.14     |
| mae-c4     | 0.429 | 0.417    | 1.588e-07   | 152896 | 0.04     |
| mae-c4-iv  | 0.429 | 0.417    | 1.443e-07   | 152896 | 0.04     |
| jepa       | 0.429 | 0.300    | 5.186e-02   | 246048 | 0.04     |
```

At only 8 training steps the downstream accuracy is flat (as expected);
the intrinsic equivariance diagnostic nevertheless shows the C₄-MAE
variants are ~5 orders of magnitude more rotation-equivariant than the
vanilla baseline. Running the same matrix with longer training,
multiple seeds and real FITS data is what produces paper-grade numbers.

---

## Quickstart — classical pretrain / eval flow

```bash
# 1. Synthetic data (no network, no astroNN).
astromae data preprocess --synthetic 16 --synthetic-classes 3 \
    --image-size 64 --processed-dir ./data/processed

# 2. MAE pretraining with the vanilla loss.
astromae train --config configs/tiny.yaml --epochs 1

# 3. Frozen-encoder feature extraction.
astromae eval embed \
    --checkpoint ./outputs/checkpoints/last.pt \
    --output-dir ./outputs/embeddings

# 4. Linear probe + KMeans.
astromae eval probe --embeddings-dir ./outputs/embeddings

# 5. Reconstruction grid.
astromae eval reconstruct \
    --checkpoint ./outputs/checkpoints/last.pt \
    --output ./outputs/reconstruction.png
```

---

## Architecture

```
astromae/
├── data/
│   ├── fits.py              # multi-extension FITS reader/writer with IVAR/MASK
│   ├── synth.py             # Sérsic galaxy generator with Poisson+Gaussian noise
│   ├── transforms.py        # asinh stretch, normalisation, C₄ rot/flip
│   ├── registry.py          # SurveySpec registry (synth_A, synth_B, decals_dr10, hsc_pdr3)
│   ├── download.py          # optional Galaxy10 via astroNN
│   ├── preprocess.py        # PNG -> .pt preprocessing
│   └── dataset.py           # PreprocessedAstroDataset + split utils
├── models/
│   ├── mae.py               # MaskedAutoencoder (+ IVAR-weighted forward)
│   ├── equivariant.py       # MaskedAutoencoderC4 + rot-equiv patch embed + polar posemb
│   ├── jepa.py              # I-JEPA context/target/predictor
│   └── pos_embed.py         # 2D sin-cos pos embeddings
├── training/
│   ├── losses.py            # mse_masked_loss, ivar_weighted_masked_loss,
│   │                        # chi_squared_outer_annulus, equivariance_regulariser
│   ├── scheduler.py         # CosineWarmupScheduler
│   └── trainer.py           # Trainer + TrainerConfig
├── evaluation/
│   ├── embed.py             # frozen encoder feature extraction
│   ├── heads.py             # morphology probe, PhotoZ MDN (CRPS/PIT), anomaly probe
│   ├── baselines.py         # random MAE, ResNet18 supervised baseline
│   ├── cross_survey.py      # pretrain-on-A-probe-on-B protocol
│   ├── matrix.py            # experiment-matrix runner
│   ├── probe.py             # linear probe + KMeans
│   └── visualize.py         # reconstruction grid
└── utils/                   # seeding, device selection
```

---

## Research track

The five pinned novel contributions live in
[`aidlc-docs/research/novel-contributions.md`](aidlc-docs/research/novel-contributions.md):

| # | Contribution | What's new | Where it lives |
|---|---|---|---|
| C1 | **IVAR-weighted MAE loss** | First astronomical MAE to use per-pixel FITS IVAR in the reconstruction loss. | `astromae.training.losses.ivar_weighted_masked_loss` |
| C2 | **C₄-equivariant ViT-MAE** | First equivariant ViT-SSL for astronomy. | `astromae.models.equivariant.MaskedAutoencoderC4` |
| C3 | **MAE vs I-JEPA on noisy astro images** | First rigorous head-to-head benchmark on astronomical data with calibration-aware metrics. | `astromae.models.jepa.IJEPA` + `astromae.evaluation.matrix` |
| C4 | **Cross-survey foundation-model protocol** | First SSL-foundation-model cross-survey probe protocol. | `astromae.evaluation.cross_survey_probe` |
| C5 | **Flow-matching photo-z on SSL embeddings** | MDN head with CRPS/PIT; flow-matching variant is the next step. | `astromae.evaluation.heads.train_photoz_mdn` |

Every contribution has a **falsifiable prediction** (see the research
docs). Every prediction has at least a CPU-runnable smoke test in the
`tests/` suite. Paper-grade numbers require GPU-scale training on real
FITS cutouts; the evaluation protocol for that is pinned in
[`aidlc-docs/research/evaluation-protocol.md`](aidlc-docs/research/evaluation-protocol.md).

### AI-DLC Research-Rigor extension

A new AI-DLC extension is authored in
[`aidlc-docs/research-extension/research-rigor/`](aidlc-docs/research-extension/research-rigor/).
It adds 10 blocking rules (RESEARCH-01 … RESEARCH-10) covering:

1. Literature-review gate.
2. Falsifiable prediction per contribution.
3. Matched-compute baseline.
4. Ablation matrix mandatory.
5. Uncertainty and calibration reporting.
6. Seed stability.
7. Data provenance.
8. Reproducibility bundle.
9. Scope-creep brake.
10. Negative-result honesty.

Ship-ready for upstream inclusion at
[awslabs/aidlc-workflows](https://github.com/awslabs/aidlc-workflows).

---

## Python API

```python
import torch
from astromae.data import make_synthetic_fits_dataset
from astromae.models import mae_c4_tiny, ijepa_tiny
from astromae.evaluation import (
    extract_embeddings, morphology_probe, train_photoz_mdn, cross_survey_probe,
)

# Build a C4-equivariant MAE.
model = mae_c4_tiny(img_size=64, patch_size=16, in_chans=3)

# Forward with the IVAR-weighted loss.
imgs = torch.rand(4, 3, 64, 64)
ivar = torch.rand_like(imgs) + 0.5
loss, pred, mask = model(imgs, mask_ratio=0.75, ivar=ivar)

# Alternatively, pretrain with I-JEPA.
jepa = ijepa_tiny(img_size=64, patch_size=16, in_chans=3)
out = jepa(imgs)
out.loss.backward()
jepa._ema_update()
```

---

## Development

```bash
pip install -e ".[dev]"
pre-commit install
make lint
make test
make smoke
```

CI runs on Python 3.10 / 3.11 / 3.12 with CPU PyTorch wheels; no real
data is downloaded in CI.

---

## References

- He, K. et al. (2021). [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).
- Assran, M. et al. (2023). [I-JEPA](https://arxiv.org/abs/2301.08243).
- Mohamadi, S. & Saee, H. (2024). AstroMAE: Redshift Prediction Using a Masked Autoencoder.
- Walmsley, M. et al. (2024). Euclid DR1 MAE. ([HuggingFace](https://huggingface.co/mwalmsley/euclid-dr1-mae))
- Lanusse, F. et al. (2023). [AstroCLIP](https://arxiv.org/abs/2310.03024).
- Himes, J. et al. (2025). [Multi-Modal Masked Autoencoders](https://arxiv.org/abs/2510.22527).
- Dillmann, S. & Heinrich, J. (2023). [E(2) Equivariant NNs for Galaxy Morphology](https://arxiv.org/abs/2311.01500).
- Mirzaei, B. et al. (2021). [Batch Inverse-Variance weighting](https://arxiv.org/abs/2107.04497).
- Zhou, X. et al. (2024). [Photometric redshifts for DESI-LS with BNNs](https://arxiv.org/abs/2412.02390).

See [`aidlc-docs/research/literature-review.md`](aidlc-docs/research/literature-review.md)
for the full annotated bibliography and ownership matrix.

## License

MIT. See [LICENSE](LICENSE).
