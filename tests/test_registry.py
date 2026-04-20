"""Survey registry tests."""

import pytest

from astromae.data.registry import SurveySpec, get, list_surveys, register


def test_builtin_entries_exist():
    surveys = list_surveys()
    for expected in ("synth_A", "synth_B", "decals_dr10", "hsc_pdr3"):
        assert expected in surveys


def test_register_round_trip():
    spec = SurveySpec(
        name="custom_unit_test",
        bands=("u",),
        pixel_scale_arcsec=0.5,
        normalisation={"mean": (0.0,), "std": (1.0,)},
    )
    register(spec)
    assert get("custom_unit_test") is spec


def test_unknown_survey_raises():
    with pytest.raises(KeyError):
        get("does_not_exist_xyz")
