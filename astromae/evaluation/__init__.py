"""Downstream evaluation: embeddings, linear probes, reconstruction."""

from astromae.evaluation.embed import extract_embeddings, load_model_from_checkpoint
from astromae.evaluation.probe import cluster_embeddings, linear_probe
from astromae.evaluation.visualize import save_reconstruction_grid

__all__ = [
    "cluster_embeddings",
    "extract_embeddings",
    "linear_probe",
    "load_model_from_checkpoint",
    "save_reconstruction_grid",
]
