"""Downstream evaluation: embeddings, linear probes, reconstruction."""

from astromae.evaluation.baselines import (
    ResNet18Baseline,
    SupervisedMetrics,
    random_encoder_embeddings,
    train_resnet18_baseline,
)
from astromae.evaluation.cross_survey import CrossSurveyMetrics, cross_survey_probe
from astromae.evaluation.embed import extract_embeddings, load_model_from_checkpoint
from astromae.evaluation.heads import (
    AnomalyMetrics,
    PhotoZMDN,
    PhotoZMetrics,
    ProbeMetrics,
    anomaly_probe,
    morphology_probe,
    train_photoz_mdn,
)
from astromae.evaluation.probe import cluster_embeddings, linear_probe
from astromae.evaluation.visualize import save_reconstruction_grid

__all__ = [
    "AnomalyMetrics",
    "CrossSurveyMetrics",
    "PhotoZMDN",
    "PhotoZMetrics",
    "ProbeMetrics",
    "ResNet18Baseline",
    "SupervisedMetrics",
    "anomaly_probe",
    "cluster_embeddings",
    "cross_survey_probe",
    "extract_embeddings",
    "linear_probe",
    "load_model_from_checkpoint",
    "morphology_probe",
    "random_encoder_embeddings",
    "save_reconstruction_grid",
    "train_photoz_mdn",
    "train_resnet18_baseline",
]
