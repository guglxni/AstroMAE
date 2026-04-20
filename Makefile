.PHONY: help install dev lint format test test-cov clean download preprocess smoke train

PY ?= python

help:
	@echo "Common targets:"
	@echo "  install     Install package (pip install -e .)"
	@echo "  dev         Install package with dev extras"
	@echo "  lint        ruff check + format check"
	@echo "  format      ruff format"
	@echo "  test        pytest"
	@echo "  test-cov    pytest with coverage"
	@echo "  smoke       End-to-end synthetic-data smoke run (CPU)"
	@echo "  download    Download Galaxy10 via astroNN"
	@echo "  preprocess  Preprocess raw PNGs to .pt tensors"
	@echo "  train       Run training with configs/default.yaml"
	@echo "  clean       Remove build artefacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev,viz,tb]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest -q

test-cov:
	pytest --cov=astromae --cov-report=term --cov-report=html -q

download:
	$(PY) -m astromae.cli data download --num-images 1000

preprocess:
	$(PY) -m astromae.cli data preprocess --image-size 128

train:
	$(PY) -m astromae.cli train --config configs/default.yaml

smoke:
	$(PY) -m astromae.cli data preprocess --synthetic 16 --synthetic-classes 3 \
		--image-size 64 --processed-dir ./data/processed
	$(PY) -m astromae.cli train --config configs/tiny.yaml --epochs 1
	$(PY) -m astromae.cli eval embed \
		--checkpoint ./outputs/checkpoints/last.pt \
		--output-dir ./outputs/embeddings
	$(PY) -m astromae.cli eval probe \
		--embeddings-dir ./outputs/embeddings

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -name __pycache__ -type d -exec rm -rf {} +
