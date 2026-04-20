# AstroMAE — Research Intent

> Authored under the AI-DLC *Research-Track* extension (see
> `aidlc-docs/research-extension/`). This supersedes the generic intent in
> `aidlc-docs/inception/intent.md` when the goal shifts from "clean scaffold"
> to "produce a defensible research contribution".

## The problem we want to make a real dent in

Modern photometric surveys (LSST/Rubin, Euclid, HSC, DES/DESI-LS) will
produce tens of billions of galaxy cutouts over the next decade. The
volume forces a choice:

1. Train **per-task supervised models** — expensive to label, brittle to
   survey domain shift, one model per science question.
2. Train **foundation-model style self-supervised encoders** — reused
   across tasks, but (so far) mostly ported from natural-image recipes.

Option (2) is clearly the winning strategy (AstroCLIP, Euclid-MAE, MMAE,
AstroM³, Mantis-Shrimp all validate this). The open problems inside
option (2) — the problems we want to attack — are:

**RP-1. Noise-aware pretraining.** Every FITS cutout ships with a
per-pixel inverse-variance map (`IVAR` / `WEIGHT` plane). Existing
astronomical MAEs (Mohamadi 2024, Walmsley's Euclid-MAE 2024–25, MMAE
2025) all optimise *unweighted* MSE pixel reconstruction — the same loss
as the ImageNet MAE. This is physically wrong: bright pixels have
Poisson shot-noise orders of magnitude larger than sky-dominated pixels,
so unweighted MSE over-fits noise in the brightest pixels and
under-fits the low-surface-brightness tails where most of the
scientifically interesting structure (tidal streams, low-mass
satellites, strong-lens arcs) lives.

**RP-2. Symmetry-aware pretraining.** Galaxies have no preferred
orientation. E(2)-equivariant *CNNs* for galaxies exist (Dillmann & Heinrich
2023; Scaife & Porter 2021 for radio galaxies), but no published work
has built an E(2)- or C₄-equivariant *ViT-MAE/JEPA* for astronomy.
Natural-image E-SSL exists (Dangovski 2022, Garrido 2023) but has not
been cross-pollinated into astro.

**RP-3. Pretraining-objective comparison.** Is MAE actually the right
objective for noisy, low-information-density scientific images, or
should we use a latent-predictive objective (I-JEPA) that doesn't
waste capacity reconstructing sky noise? No rigorous head-to-head
comparison exists on an astronomical benchmark with proper
uncertainty-aware downstream eval.

**RP-4. Cross-survey generalisation.** A model trained on HSC should
usefully transfer to DECaLS or SDSS even though the PSF, depth, and
filter set differ. Walmsley's Zoobot and DeepAstroUDA address pieces of
this; a foundation-model-grade answer does not yet exist.

## Research thesis

> **An MAE/JEPA that (a) learns from noise-weighted reconstructions
> using IVAR maps as a first-class input, (b) is approximately
> C₄-equivariant by construction, and (c) is trained on multi-band
> FITS cutouts, produces embeddings that (i) beat unweighted MAE on
> photometric-redshift CRPS, (ii) beat supervised ResNet/ViT on
> low-label galaxy morphology, and (iii) transfer across surveys
> better than any published astro-SSL baseline.**

If that thesis survives contact with data, that is a genuinely novel,
publishable contribution. If it doesn't, the negative result (with
well-designed experiments and uncertainty-aware metrics) is also
publishable at a NeurIPS ML4PS / AAS-AI workshop.

## Claims we deliberately do not make

- We do not claim to *discover* specific objects (strong lenses, jellyfish
  galaxies, mergers) in this PR. Discovery requires a finished
  pretrain and pipeline time — the right follow-up.
- We do not aim to beat AstroCLIP in absolute cross-modal accuracy;
  AstroCLIP uses spectra, and pixel-only models are structurally weaker.
- We do not claim our implementation is the largest or fastest. We aim
  for clarity, reproducibility and physical correctness.

## Success metrics (publication-grade)

| Metric | Target | Baseline |
|---|---|---|
| Photo-z σ_NMAD (linear probe on embeddings) | ≤ 0.03 on Galaxy10 proxy / DECaLS-EDR sample | DESI-BNN 0.018 (full supervised) |
| Photo-z CRPS, PIT calibration | CRPS ≤ baseline, PIT deviation from uniform ≤ 0.02 | — |
| Galaxy10 linear probe @ 1% labels | ≥ 80% top-1 | ResNet-18 supervised ≈ 65–70% @ 1% |
| Cross-survey probe (pretrain HSC → probe DECaLS) | accuracy drop ≤ 8 pp | ≥ 15 pp for non-equivariant unweighted MAE |
| Low-SB reconstruction χ² | ≥ 15% better than unweighted MAE | unweighted MAE |
| Rotation equivariance error (CLS embedding under C₄) | ≤ 1e-2 cosine distance | vanilla MAE ~ 0.1–0.3 |

## Out of scope even for the research track

- Training on the full LSST / Euclid corpus (that's compute we don't
  have). We validate on public DECaLS-SV / HSC-PDR cutouts.
- Spectroscopic modality. Out of scope; the design leaves room for a
  future cross-modal extension without refactoring.
- Video / time-series. Separate foundation-model problem.
