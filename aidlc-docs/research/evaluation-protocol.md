# Evaluation Protocol

Every quantitative claim in `novel-contributions.md` must be reproduced
by one of the commands in this document. No hand-tuning, no cherry-picked
seeds, at least 3 seeds per reported number.

## 0. Environment and seeding

- `python -c "from astromae.utils import seed_everything; seed_everything(S)"`
  with `S ∈ {42, 43, 44}` for every reported metric.
- Report mean ± std.
- Pin dependencies via `uv pip compile` output committed to
  `requirements-lock.txt` before any publication run.

## 1. Pretraining configurations under test

| Tag | Backbone | Objective | Loss | Equivariance |
|-----|----------|-----------|------|---------------|
| `mae` | ViT | MAE | MSE | — |
| `mae-iv` | ViT | MAE | **IVAR-MSE** | — |
| `mae-c4` | ViT | MAE | MSE | **C₄** |
| `mae-c4-iv` | ViT | MAE | **IVAR-MSE** | **C₄** |
| `jepa` | ViT | I-JEPA | MSE in feature space | — |
| `jepa-c4` | ViT | I-JEPA | MSE in feature space | **C₄** |
| `jepa-iv-c4` | ViT | I-JEPA | **IVAR-weighted target reg.** | **C₄** |

All trained with matched wall-clock and matched parameter count.
The `-iv` variants require FITS data; on synthetic data we emulate IVAR.

## 2. Downstream evaluation heads

All heads are trained on top of *frozen* encoders.

### 2.1 Linear probe / k-NN (morphology)

- Metric: top-1 accuracy, macro-F1 on Galaxy10 (and DECaLS-subset once
  available).
- Low-label regime: 1%, 5%, 10%, 100% of labels.
- Hyperparameters fixed: `LogisticRegression(max_iter=2000,
  StandardScaler)`.

### 2.2 Photometric redshift — MDN head

- Metric: σ_NMAD, outlier fraction (|z_pred − z_true|/(1+z_true) > 0.15),
  CRPS.
- 5-Gaussian mixture density network, fixed architecture.

### 2.3 Photometric redshift — Flow-matching head

- Conditional flow matching (Lipman 2023) on the embedding.
- Metrics: CRPS, PIT histogram (ideal flat), 68 / 95% coverage,
  σ_NMAD via argmax of integrated posterior.

### 2.4 Anomaly detection

- Inject simulated strong-lens arcs / jellyfish features into a
  labelled subset; hold out.
- Train IsolationForest on embeddings.
- Metric: AUROC, recall at top-k.

## 3. Intrinsic encoder diagnostics

### 3.1 Rotation-equivariance error

For each model and each test image x:
```
err(x) = mean_{θ ∈ {90°,180°,270°}} ‖f(R_θ x) − f(x)‖₂ / ‖f(x)‖₂
```
Report median, 90th percentile across 1000 test images.

### 3.2 Low-surface-brightness reconstruction χ²

On held-out FITS cutouts with known IVAR:
```
χ² = sum_pixels_in_outer_annulus( IVAR · (pred − target)² )
```
Normalised by number of pixels. Compare `mae` vs `mae-iv`.

### 3.3 Rank-based retrieval quality

k-NN retrieval in embedding space on a labelled set; metric is
precision@10 using Galaxy10 class.

## 4. Cross-survey protocol

For each pretrained model `M` from section 1:
- Train morphology probe on Survey A; evaluate on A → `acc_AA`.
- Train morphology probe on Survey A embeddings of Survey B images;
  evaluate on B → `acc_AB`.
- Report `Δ = acc_AA − acc_AB` (smaller is better).

In this PR, surveys A and B are two synthetic survey profiles differing
in (i) effective PSF (1.0 vs 1.5 px Gaussian blur), (ii) additive noise
σ (0.05 vs 0.15), (iii) pixel scale (jitter). This is a pilot of the
real-data protocol.

## 5. Compute honesty

Every reported number is annotated with:
- GPU / CPU used.
- Wall-clock seconds.
- Parameter count.
- Number of seeds.

If a result was obtained on synthetic data, that must be stated *in the
table caption*, not in a footnote.

## 6. Pass/fail criteria for the thesis

The thesis in `research-intent.md` is **accepted** if ≥ 4 of the
following 5 hold:

1. `mae-iv` beats `mae` in low-SB χ² by ≥ 15%.
2. `mae-c4` beats `mae` in rotation equivariance by ≥ 10× and in
   Galaxy10 @ 1% labels by ≥ 3 pp top-1.
3. `jepa` beats `mae` in CRPS on photo-z by ≥ 5%.
4. `mae-c4-iv` (or `jepa-c4-iv`) has cross-survey Δ ≤ 8 pp and beats
   vanilla `mae` by ≥ 7 pp.
5. PhotoZ-FM on `mae-c4-iv` achieves CRPS within 5% of the supervised
   DESI-BNN baseline at matched label counts.

Fewer than 4 → null result. Null results are publishable at ML4PS /
AAS-AI provided the experiment design is honest.

## 7. Reproducibility bundle

For every paper-grade number we must be able to run:
```
astromae experiment run --config experiments/<name>.yaml --seed S
```
and have it write:
- `results.json` — metrics, params, wall-clock, git SHA.
- `config_resolved.yaml` — fully resolved config.
- `checkpoint_last.pt` — model state.
- `embeddings.npy`, `labels.npy` — for re-eval without retraining.
