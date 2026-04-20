# Research-Track Execution Plan

Each unit of work is independently shippable and leaves the repo with
green tests and lint.

## Milestone 0 — Research scaffolding (this PR, documentation)

- **M0.1** Research-track inception docs (this directory).
- **M0.2** AI-DLC research extension rules
  (`aidlc-docs/research-extension/`) that add:
  - Mandatory literature-review gate before construction.
  - Mandatory falsifiable-prediction declaration per contribution.
  - Mandatory baseline + ablation matrix before claiming novelty.
- **M0.3** Dependency additions (`astropy`, optional `e2cnn`).

## Milestone 1 — FITS / multi-band data infrastructure

- **M1.1** `astromae.data.fits` module: read `.fits` cutouts with
  optional `IVAR` / `WEIGHT` / `MASK` planes using `astropy.io.fits`.
- **M1.2** `FITSCutoutDataset` — returns `{image, ivar, meta}` tuples.
- **M1.3** `astromae.data.transforms` — asinh stretch
  (Lupton-style), per-channel normalisation with survey-level stats,
  optional `DataMask` handling.
- **M1.4** Synthetic FITS generator for unit tests (no astroNN
  download required in CI).
- **M1.5** Dataset registry (`astromae.data.registry`) mapping survey
  names → default band lists, pixel scales, stats.

## Milestone 2 — IVAR-weighted MAE loss (C1)

- **M2.1** `astromae.training.losses.ivar_weighted_mse(pred, target, ivar, mask)`.
- **M2.2** Integrate into `MaskedAutoencoder.forward_loss` behind a
  flag so backwards-compatibility is preserved.
- **M2.3** Tests: (a) `ivar ≡ 1` reproduces vanilla MSE;
  (b) extremely low IVAR pixels don't dominate; (c) gradient correctness.
- **M2.4** Synthetic physics test: injected low-surface-brightness
  arc must be reconstructed better than with unweighted MSE.

## Milestone 3 — Rotation-equivariant ViT-MAE (C2)

- **M3.1** `astromae.models.equivariant.RotEquivPatchEmbed` — shared
  weights applied at 4 rotations, mean-pooled (exact C₄ equivariance
  for the patch embedding).
- **M3.2** `astromae.models.equivariant.C4PositionalEmbedding` — use
  sin-cos with (r, θ)-polar coordinates so 90° rotations permute
  tokens consistently.
- **M3.3** Optional equivariance regulariser in the trainer:
  `L_eq = mean ‖f(rot90ⁿ(x)) − f(x)‖²` for `n ∈ {1,2,3}`.
- **M3.4** Unit tests: measured equivariance error under all 4 C₄
  rotations ≤ 1e-4 at init, ≤ 1e-2 after training.
- **M3.5** Factory: `mae_c4_tiny`, `mae_c4_small`, `mae_c4_base`.

## Milestone 4 — I-JEPA objective (C3)

- **M4.1** `astromae.models.jepa.JEPAHead` — context encoder + target
  encoder (EMA of context) + small predictor.
- **M4.2** `astromae.training.jepa_trainer.JEPATrainer` reusing the
  same checkpoint / logging plumbing.
- **M4.3** Config profile `configs/jepa_tiny.yaml`.
- **M4.4** Extend evaluation to accept JEPA-style encoders
  (`forward_features` interface).
- **M4.5** Unit tests: loss decreases on a synthetic dataset; EMA
  updates are gradient-free; mask-block-sampler produces valid blocks.

## Milestone 5 — Scientific downstream heads (C5)

- **M5.1** `astromae.evaluation.heads.morphology` — already-tested linear
  probe; ensure it works with any encoder (MAE / JEPA / equivariant).
- **M5.2** `astromae.evaluation.heads.photoz_mdn` — Mixture Density
  Network head on embeddings with σ_NMAD / outlier-fraction metrics.
- **M5.3** `astromae.evaluation.heads.photoz_fm` — Conditional
  Flow-Matching head. CRPS + PIT diagnostics.
- **M5.4** `astromae.evaluation.heads.anomaly` — Isolation Forest in
  embedding space; rank-based evaluation using labelled synthetic
  anomalies.

## Milestone 6 — Baselines & honest comparison

- **M6.1** `astromae.evaluation.baselines.ResNet18Baseline` — supervised
  end-to-end baseline. Trained on Galaxy10 with the same data pipeline.
- **M6.2** Random-encoder baseline (untrained MAE).
- **M6.3** "Same architecture, different loss" matrix runner:
  `astromae.evaluation.matrix.BaselineMatrix` that produces the table
  reported in the paper draft.

## Milestone 7 — Cross-survey protocol (C4)

- **M7.1** `astromae.evaluation.cross_survey` — take a pretrained
  encoder, a labelled evaluation dataset from *another* survey, run
  the linear probe, return accuracy-drop.
- **M7.2** Registry support for at least two mock surveys in the
  synthetic-data path; one real-data recipe (DECaLS → HSC once
  compute exists).

## Milestone 8 — CLI, CI, docs, notebook

- **M8.1** Extend CLI: `astromae train --objective {mae,jepa}`,
  `--loss {mse,ivar}`, `--equivariant {none,c4}`; `astromae eval
  photoz {mdn,fm}`; `astromae eval baselines`; `astromae eval
  cross-survey`.
- **M8.2** Add an experiment matrix runner invokable via `make
  experiment-matrix`.
- **M8.3** Update README with a research-grade quickstart.
- **M8.4** Populate a second notebook: `notebooks/research_experiments.ipynb`
  that runs the full C1–C3 ablation on synthetic FITS data in < 2 min
  on CPU.

## Definition of done for this PR

- All new modules have unit tests; the CPU smoke suite still passes.
- `ruff check` and `ruff format --check` pass.
- Experiment-matrix runner produces a CSV + markdown report on
  synthetic data so every claim has at least a *smoke-level* empirical
  check.
- Research docs are complete and cross-reference the code paths that
  implement each contribution.

## What this PR intentionally defers

| Deferred item | Why |
|---------------|-----|
| Pretraining on full DECaLS / HSC / Euclid | Needs GPU-weeks. |
| Publishing a HuggingFace weight release | Follows from a real pretrain. |
| Strong-lens / anomaly discovery campaign | Follow-up after a real pretrain. |
| Sparse-autoencoder interpretability layer | Nice follow-up; already owned by Wu&Walmsley for Euclid. |
| Spectroscopic modality | Separate cross-modal extension. |
