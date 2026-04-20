# Literature Review — Astro Self-Supervised Learning (2021 → 2026)

Cut-off: April 2026. Sources: arXiv, ADS, published conference venues
(NeurIPS ML4PS, ICML AI4Science, AAS journals). Only papers we can
directly verify are cited. This document is used to establish what is
already owned by others and where the unclaimed ground lies.

## 1. Foundation models for galaxy images

| Model | Year | Architecture | Data | Notes |
|-------|------|--------------|------|-------|
| Hayat et al. — contrastive galaxy encoder | 2021 | ResNet + MoCo | SDSS 1.3M | First self-supervised galaxy encoder. |
| Stein et al. — similarity search for lenses | 2022 | MoCo-v2 ResNet-50 | DESI-LS 76M | Single-label k-NN found new lenses. |
| Walmsley et al. — **Zoobot** | 2022–25 | CNN (EffNet/ConvNeXt), supervised | GZ / DECaLS / Euclid | De facto morphology baseline. |
| **AstroCLIP** (Lanusse, Parker) | 2023 | Dual-transformer, CLIP-style | DESI spec + LS img | Cross-modal foundation model. |
| **AstroMAE** (Mohamadi, Saee) | 2024 | ViT + MAE | SDSS | MAE pretraining + photo-z fine-tune. Single-survey, unweighted loss. |
| **Euclid-MAE** (Walmsley et al.) | 2024–25 | ViT, 30M params, 8×8 patches, 90% mask | Euclid DR1 13.6M | State of the art single-survey MAE. Unweighted reconstruction. |
| **MMAE** (Himes et al.) | NeurIPS ML4PS 2025 | Multi-modal MAE | HSC-PDR2 + DESI-DR1 (134k) | Images + spectra joint, unweighted. |
| **AstroM³** (Rizhko, Bloom) | AJ 2025 | CLIP trimodal | Time-series + spectra + metadata | Not images — complementary. |
| Mantis-Shrimp | IoP 2025 | Multi-band CNN | UV/Optical/IR | Photo-z; not SSL. |
| Euclid-SAE (Wu, Walmsley) | NeurIPS workshop 2025 | Sparse autoencoder on MAE features | Euclid Q1 | Interpretability layer. |

**Takeaway.** The "pretrain-on-a-single-survey-with-unweighted-MAE" cell
is saturated (Mohamadi, Euclid-MAE). The next rung — **noise-aware,
symmetry-aware, multi-survey** SSL — is empty.

## 2. Self-supervised objectives (2021–2026)

| Objective | Paper | Contribution |
|-----------|-------|--------------|
| Contrastive (SimCLR, MoCo) | Chen 2020, He 2020 | Augmentation-invariant. |
| MAE | He et al. 2022 | 75% mask, pixel reconstruction. |
| SimMIM | Xie 2021 | Random masking, light decoder. |
| BEiT | Bao 2021 | Tokenised target. |
| **DINO / DINOv2 / DINOv3** | Caron 2021; Oquab 2024; 2025 | Self-distillation, ViT, Gram anchoring in v3. |
| **I-JEPA** | Assran 2023 | Predicts *features* of target blocks, not pixels. |
| **V-JEPA** | Bardes 2024 | Video version. |
| **C-JEPA** | Chen 2024 | I-JEPA + VICReg; fixes representation collapse. |
| **D-JEPA** | Chen et al. ICLR 2025 | JEPA + diffusion/flow matching; generative + SSL. |
| Equivariant SSL | Dangovski 2022, Garrido 2023, Park 2025 | Augmentation-aware, preserves transformation info. |
| **Equivariance-coherent SSL** | Gupta et al. Mar 2025 (arXiv:2503.18753) | Intermediate-transformation reconstruction; plugs into iBOT/DINOv2. |

**Takeaway.** I-JEPA is better suited than MAE for images where
pixel-level reconstruction wastes capacity on noise — exactly the astro
regime. No paper compares MAE vs I-JEPA on astronomical data with
noise-aware evaluation.

## 3. Equivariant vision for astronomy

| Paper | Architecture | Symmetry | Domain |
|-------|--------------|----------|--------|
| Scaife & Porter 2021 (arXiv:2102.08252) | GCNN | D₁₆ | Radio galaxy Fanaroff-Riley |
| Dillmann & Heinrich 2023 (arXiv:2311.01500) | E(2)-Steerable CNN | D₁₆ | Galaxy10 DECals — 95.5% |
| Various — steerable CNN library (e2cnn) | — | E(2) | General |

**Takeaway.** All published astro equivariance is CNN-based. There is
**no published rotation-equivariant ViT-MAE or JEPA for galaxies**.
Building one is genuinely novel.

## 4. Noise-weighted / heteroscedastic deep learning

| Paper | Idea | Domain |
|-------|------|--------|
| Kendall & Gal 2017 | Heteroscedastic aleatoric uncertainty | General vision. |
| Mirzaei et al. 2021 — Batch-Inverse-Variance (arXiv:2107.04497) | IVAR weighted regression | Robotics/regression. |
| Farrens et al. 2020 (A&A 638, A18) | U-Net deconvolution with noise | Galaxy imaging. |
| PINN heteroscedastic losses | 2024–25 cosmology PINNs | Not SSL. |

**Takeaway.** Inverse-variance-weighted *regression* is a known idea;
no one has lifted it into a self-supervised *reconstruction* objective
for scientific imaging. The MAE/JEPA literature assumes homoscedastic
Gaussian noise by default (through plain MSE).

## 5. Photometric redshift with uncertainty

- Zhou et al. Dec 2024 (arXiv:2412.02390) — Bayesian NNs on DESI-LS, σ_NMAD 0.018 (BGS).
- ViT-MDNz 2024 — single-band ViT + mixture density network, σ_NMAD 0.034.
- Probabilistic Autoencoder (Tseng 2024), Conditional Flow Matching (Dec 2025 arXiv:2512.05078).

**Takeaway.** Calibration (PIT / CRPS) and mixture/flow density heads
are the current standard. Using MAE embeddings with a flow-matching
photo-z head is a natural combination that nobody has published.

## 6. Cross-survey domain adaptation

- DeepAstroUDA (Ciprijanović 2023) — universal domain adaptation across SDSS → DECaLS.
- LSBG-DETR / LSBG-ViT 2025 — DES → HSC transfer.
- Pix2WGAN 2025 — SDSS/DECaLS → HSC image enhancement.

**Takeaway.** No SSL foundation-model paper explicitly evaluates
pretraining on survey A and probing on survey B with the same
architecture and no fine-tuning.

## 7. Mapping our contributions against the literature

| Contribution | Prior art | Novelty |
|--------------|-----------|---------|
| **IVAR-weighted MAE loss** | Unweighted MAE everywhere (Mohamadi, Walmsley, MMAE); BIV regression (Mirzaei 2021) | First SSL reconstruction loss using FITS IVAR maps. |
| **C₄-equivariant ViT-MAE / JEPA for galaxies** | E(2)-CNNs exist; equivariant ViT-SSL is natural-image only | First equivariant ViT-SSL for astronomy. |
| **Head-to-head MAE vs I-JEPA on noisy astro images** | JEPA/MAE comparisons only on ImageNet | First astro benchmark with noise-aware and calibration-aware eval. |
| **Pretrain-on-A, probe-on-B protocol** | DeepAstroUDA (not SSL), Zoobot (supervised) | First SSL foundation-model-style cross-survey eval. |
| **Flow-matching photo-z head on SSL embeddings** | BNN / MDN / CFM heads on raw pixels | Combines modern photo-z PDF head with SSL encoder. |

All five rows are individually plausible and together form a coherent
thesis that was not previously available.
