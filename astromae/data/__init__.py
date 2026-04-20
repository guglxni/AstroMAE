"""Dataset / preprocessing utilities."""

from astromae.data.dataset import (
    PreprocessedAstroDataset,
    create_dataloader,
    create_train_val_dataloaders,
)
from astromae.data.fits import (
    FITSCutout,
    FITSCutoutDataset,
    load_fits_cutout,
    save_fits_cutout,
)
from astromae.data.registry import SurveySpec, get, list_surveys, register
from astromae.data.synth import (
    SURVEY_A,
    SURVEY_B,
    SurveyProfile,
    make_synthetic_cutout,
    make_synthetic_fits_dataset,
    make_synthetic_pair_for_lsb_test,
)
from astromae.data.transforms import (
    AsinhStretch,
    BandNormalize,
    Compose,
    RandomFlipRotate90,
)

__all__ = [
    "SURVEY_A",
    "SURVEY_B",
    "AsinhStretch",
    "BandNormalize",
    "Compose",
    "FITSCutout",
    "FITSCutoutDataset",
    "PreprocessedAstroDataset",
    "RandomFlipRotate90",
    "SurveyProfile",
    "SurveySpec",
    "create_dataloader",
    "create_train_val_dataloaders",
    "get",
    "list_surveys",
    "load_fits_cutout",
    "make_synthetic_cutout",
    "make_synthetic_fits_dataset",
    "make_synthetic_pair_for_lsb_test",
    "register",
    "save_fits_cutout",
]
