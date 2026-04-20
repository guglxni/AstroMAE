# AstroMAE — High-Level Design

## Package layout (target)

```text
astromae/
├── __init__.py
├── cli.py                # `astromae ...` entry point
├── config.py             # dataclass config loader (YAML → Config)
├── data/
│   ├── __init__.py
│   ├── download.py       # Galaxy10 downloader (astroNN)
│   ├── preprocess.py     # resize → tensor → per-channel norm
│   └── dataset.py        # PreprocessedAstroDataset + split utils
├── models/
│   ├── __init__.py
│   ├── mae.py            # MaskedAutoencoder (ViT) — fixed & hardened
│   └── pos_embed.py      # 2D sin-cos positional embeddings
├── training/
│   ├── __init__.py
│   ├── trainer.py        # device/AMP/scheduler/checkpoint/resume
│   └── scheduler.py      # warmup + cosine LR
├── evaluation/
│   ├── __init__.py
│   ├── embed.py          # frozen-encoder feature extraction
│   ├── probe.py          # linear probe + KMeans purity
│   └── visualize.py      # reconstruction grid
└── utils/
    ├── __init__.py
    ├── logging.py
    └── seed.py
```

`src/` is deleted; legacy thin shims are kept only if tests still
reference them. `tests/` is updated to use `from astromae...`.

## Key design decisions

1. **Proper 2-D sin-cos positional embeddings**.  The original
   `model.py` defines helpers (`get_2d_sincos_pos_embed`) at module
   scope but calls them as `self.get_2d_sincos_pos_embed(...)` inside
   `initialize_weights`, which throws `AttributeError`.  Fix by moving
   them into `models/pos_embed.py` and calling the module-level
   functions.

2. **Channel-count correctness**.  `patchify`/`unpatchify` currently
   hardcode `3`.  Replace with `self.in_chans` so the model works for
   1 or arbitrary channel counts (common in astrophysics).

3. **timm API stability**.  `timm.models.vision_transformer.PatchEmbed`
   and `Block` are internal and have churned across versions; pin to
   public `timm.layers.PatchEmbed` + re-implement `Block` from
   `timm.models.vision_transformer` via a safe wrapper or, simpler,
   drop a tiny local `Block` using `torch.nn.MultiheadAttention`. We
   take the simpler path of importing from `timm.layers` +
   `timm.models.vision_transformer.Block` *defensively*: try the new
   path first, fall back to the old one.

4. **Trainer separation of concerns**.  The `Trainer` owns device
   placement, optimizer, scheduler, AMP scaler, checkpoint I/O and
   logging; the `MaskedAutoencoder` owns the forward/loss only.

5. **Config as dataclass**.  `Config.from_yaml(path)` returns a frozen
   dataclass; defaults live in `config.py`, not scattered in trainer
   code.  This makes CI and unit tests easy.

6. **Tiny ViT profile**.  Provide `tiny` profile (patch 16, 128×128,
   embed_dim=64, depth=2, decoder_depth=1) for CPU smoke tests.

## Data flow

```text
download.py ──► data/raw/<class>/*.png
     │
     ▼
preprocess.py ──► data/processed/<class>/*.pt   (C, H, W) float32 in [0,1]
     │
     ▼
PreprocessedAstroDataset ──► (optional) train/val random_split
     │
     ▼
Trainer.train()   ──► outputs/checkpoints/{last,best,epoch_N}.pt
                  └► outputs/train_log.csv + TensorBoard events
     │
     ▼
evaluation.embed  ──► embeddings.npy, labels.npy
evaluation.probe  ──► probe_accuracy, classification_report.json
evaluation.visualize ──► reconstruction_grid.png
```

## Risk register

- **R-01 timm version drift** — mitigated by defensive imports + a
  pinned floor in `pyproject.toml`.
- **R-02 AMP on MPS** — mitigated by only enabling AMP for CUDA.
- **R-03 Galaxy10 download flakiness (astroNN hits a third-party
  mirror)** — mitigated by making download a separate step and by
  having unit tests that synthesise tensors instead of calling out.
- **R-04 Reproducibility on GPU** — not a goal; we only seed and set
  `torch.use_deterministic_algorithms=False` to keep speed.
