# Novel Contributions — AstroMAE Research Track

This document pins down exactly what we claim is new. Every claim is
anchored to a gap identified in `literature-review.md`.

## C1. IVAR-Weighted Masked Reconstruction (IW-MAE)

### What it is

Replace MAE's plain `(pred - target)²` loss on masked patches with

```
L_patch(i) = sum_pixels( w_p · (pred_p - target_p)² ) / sum_pixels(w_p)
```

where `w_p = 1 / σ²_p` is the per-pixel inverse variance drawn directly
from the survey's `IVAR` / `WEIGHT` extension. Patches are aggregated
with the MAE mask as usual.

### Why it's novel

No published MAE for astronomy uses per-pixel IVAR in the loss. The
idea is a direct transplant of Mirzaei et al. (2021) "Batch-Inverse
Variance" from supervised regression to self-supervised
reconstruction, and it is the physically correct loss for Poisson +
Gaussian noise astronomical imaging where noise is known pixel-by-pixel.

### Why it should help

- Sky-dominated pixels carry high `w_p` → the model is forced to learn
  the low-surface-brightness structure (tidal features, arcs).
- Bright-core pixels with near-saturated Poisson noise have low `w_p` →
  the model stops overfitting readout/photon noise near galaxy centres.
- The loss is numerically well-behaved and a drop-in replacement:
  setting `w_p ≡ 1` recovers vanilla MAE exactly.

### Falsifiable prediction

On DECaLS-EDR cutouts with simulated low-surface-brightness injections,
IW-MAE reduces reconstruction χ² in the outer annulus (r > 0.7·r_50) by
at least 15% relative to vanilla MAE trained for equal steps.

## C2. Rotation-Equivariant ViT-MAE (C₄-MAE)

### What it is

Modify the patch embedding and the positional embedding of the ViT-MAE
so that the CLS-token embedding is approximately equivariant under the
cyclic rotation group C₄ (90° / 180° / 270°). Implementation:

1. **Shared rotated convolution stems**: the patch-embed `Conv2d` is
   replaced with a stem that applies the same 3×3 filter at 0°, 90°,
   180°, 270° rotations and averages over the group. This makes the
   patch features C₄-equivariant.
2. **Rotation-aware positional embedding**: instead of fixed sin-cos,
   we use an embedding that is compatible with 90° patch-grid
   permutations, so that rotating the image by 90° permutes the tokens
   consistently with the position codes.
3. **Equivariance regularisation**: a cheap consistency term
   `‖f(rot90ⁿ(x)) − f(x)‖` is added with small weight to drive
   remaining non-equivariance to zero.

### Why it's novel

No published rotation-equivariant ViT-MAE (or ViT-JEPA) for galaxies.
Equivariant CNNs for galaxies exist (Dillmann & Heinrich 2023) and
equivariant SSL for natural images exists (Dangovski 2022), but the
intersection is empty.

### Why it should help

- Galaxies have no preferred orientation → orientation is a pure
  nuisance variable. Every non-equivariant model is wasting capacity
  learning orientation invariance from data.
- The inductive bias should show up as: (a) better low-label accuracy,
  (b) lower variance of embeddings under test-time rotation,
  (c) more stable cross-survey transfer (PSF orientation is
  survey-specific).

### Falsifiable prediction

C₄-MAE's CLS embeddings satisfy `‖f(rot(x)) − f(x)‖₂ / ‖f(x)‖₂ ≤ 0.01`
for `rot ∈ C₄`, vs. 0.1–0.3 for vanilla MAE. In the Galaxy10 linear
probe at 1% of labels, C₄-MAE beats vanilla MAE by ≥ 3 pp top-1.

## C3. Head-to-Head MAE vs I-JEPA on Noisy Astronomical Images

### What it is

A single pretraining harness that can train either:
- Vanilla MAE (baseline)
- IW-MAE (C1)
- C₄-MAE (C2)
- IW + C₄-MAE (C1 ∪ C2)
- I-JEPA
- IW-JEPA (noise-aware I-JEPA — novel in its own right)

All with matched:
- backbone (same ViT)
- compute (matched FLOPs / wall-clock)
- downstream evaluation protocol

### Why it's novel

No published paper makes this comparison on an astronomical benchmark
with noise-aware and calibration-aware downstream metrics. The closest
is MMAE which only compares to its own ablations.

### Why it should help

I-JEPA operates in feature space so it **doesn't waste capacity
reconstructing sky noise**. On astronomical images where a large
fraction of pixels are empty sky, this should give JEPA a structural
advantage over MAE.

### Falsifiable prediction

On photometric redshift (linear probe on embeddings, Mixture Density
Network head), I-JEPA beats MAE in CRPS by ≥ 5%. IW-IJEPA beats
everything.

## C4. Cross-Survey Transfer as a Foundation-Model Benchmark

### What it is

A protocol, not an architecture:

1. Pretrain on Survey A (e.g. HSC cutouts).
2. Freeze encoder.
3. Linear-probe / MLP-probe on Survey B (e.g. DECaLS, SDSS).
4. Report accuracy drop vs. a same-survey probe.

Implemented as a `DatasetRegistry` abstraction so any number of surveys
can be slotted in.

### Why it's novel

DeepAstroUDA does cross-survey domain adaptation for supervised
learning, but no published astro-SSL foundation model paper reports
cross-survey probe numbers with matched baselines.

### Why it should help

Real surveys (LSST, Euclid, Roman) will be deployed on data the model
didn't see. A foundation model that *doesn't* generalise across surveys
is not actually a foundation model.

### Falsifiable prediction

C₄-MAE's accuracy drop from same-survey to cross-survey probe is at
most 8 pp; vanilla MAE's drop is ≥ 15 pp.

## C5. Flow-Matching Photometric Redshift Head (PhotoZ-FM)

### What it is

A lightweight conditional flow-matching (Lipman 2023) head that
produces a full posterior `p(z | embedding)` from the frozen encoder.
Reports σ_NMAD, outlier fraction, CRPS and PIT for calibration.

### Why it's novel

Flow-matching photo-z on SSL embeddings is new. Current state of the
art uses either (a) BNNs on raw photometry, (b) CNN+MDN on raw pixels,
or (c) flow matching on photometry alone. Combining SSL embeddings
with flow-matching posterior inference has not been published.

### Why it should help

- SSL embeddings condense survey-invariant morphology information.
- Flow matching handles multi-modal posteriors (photo-z degeneracies)
  that point estimators (MSE regression) cannot.

### Falsifiable prediction

At equal parameter count, PhotoZ-FM on C₄-MAE embeddings achieves CRPS
within 5% of fully-supervised Bayesian-NN (Zhou 2024 DESI-LS) while
using only labels for the probe head, not for the encoder.

## Matrix of ownership vs. novelty

| Idea | Who owns it | We contribute |
|------|-------------|----------------|
| ViT-MAE for astronomy | Mohamadi 2024, Walmsley 2024–25 | Nothing new in the architecture. Baseline only. |
| MAE on 8×8 patches at 224² | Walmsley | Same. |
| Sparse-AE interpretability | Wu & Walmsley 2025 | Out of scope for this PR (future extension hook). |
| CLIP cross-modal | AstroCLIP, MMAE, AstroM³ | Out of scope (pixel-only model). |
| Equivariant CNN | Dillmann & Heinrich, Scaife | We lift the idea into a ViT-MAE. |
| BIV regression | Mirzaei 2021 | We lift the idea into MAE reconstruction loss. |
| JEPA / I-JEPA | Assran 2023, Chen 2024 | We port it to noisy astro data. |
| Photo-z BNN / MDN / CFM | Zhou 2024, Tseng 2024 | We combine CFM with SSL embeddings. |
| Cross-survey transfer | DeepAstroUDA 2023 | We make it a foundation-model SSL benchmark. |

## What we explicitly decline to claim

- That any of these individual ideas is a magic bullet.
- That a 1-GPU training run proves a scientific effect. Scientific
  claims require the pretraining compute budget spelled out in the
  evaluation protocol; this PR delivers the **machinery and the
  experiment design**, plus pilot results on public subsets.
- That we've found new strong lenses / anomalies. Discovery claims
  require downstream pipeline work beyond this PR.
