"""Training utilities."""

from astromae.training.losses import (
    chi_squared_outer_annulus,
    equivariance_regulariser,
    ivar_weighted_masked_loss,
    mse_masked_loss,
    patchify_ivar,
)
from astromae.training.scheduler import CosineWarmupScheduler
from astromae.training.trainer import Trainer, TrainerConfig

__all__ = [
    "CosineWarmupScheduler",
    "Trainer",
    "TrainerConfig",
    "chi_squared_outer_annulus",
    "equivariance_regulariser",
    "ivar_weighted_masked_loss",
    "mse_masked_loss",
    "patchify_ivar",
]
