# AstroMAE — Execution Plan

Each unit of work is independently committable and leaves the repo in a
working state (CI green, `pytest` passing).

| # | Unit of Work | Key changes | Risk |
|---|--------------|-------------|------|
| 1 | AI-DLC scaffolding | `.aidlc/`, `.cursor/rules/ai-dlc.mdc`, `aidlc-docs/*` | none |
| 2 | Package skeleton | Move `src/*.py` to `astromae/`, add `pyproject.toml`, `requirements.txt`, entry points, `__init__.py`s | low |
| 3 | Model hardening | Fix sin-cos pos embed bug, generalise channel count, defensive timm imports, add unit tests | low |
| 4 | Dataset upgrades | Train/val split, optional in-memory caching, `weights_only=True` torch.load, synthetic dataset for tests | low |
| 5 | Trainer upgrades | Device auto-select, AMP, cosine+warmup LR, grad clip, best/last checkpoints, resume, TensorBoard | medium |
| 6 | Evaluation module | Embedding extractor, linear probe, KMeans purity, reconstruction visualisation | low |
| 7 | CLI | `astromae train`, `astromae data download/preprocess`, `astromae eval embed/probe/reconstruct` | low |
| 8 | Tests | Update imports, add trainer smoke test, add embedding/probe test, ensure everything runs CPU-only with tiny model | medium |
| 9 | CI & DX | Ruff config, pre-commit, modernise workflow, Makefile, Dockerfile | low |
| 10 | README + notebook | Rewrite README, populate `data_explore.ipynb` with actual exploration code | low |

## Definition of done

- `ruff check .` and `ruff format --check .` pass.
- `pytest -q` passes on CPU.
- A manual CPU smoke run:
  1. `python -m astromae.data.preprocess --synthetic 64` creates fake data.
  2. `astromae train --config configs/tiny.yaml --epochs 1` completes.
  3. `astromae eval embed --ckpt outputs/checkpoints/last.pt` writes embeddings.
  4. `astromae eval probe --embeddings outputs/embeddings.npy` prints an accuracy number.
- README reflects the new commands and architecture.
