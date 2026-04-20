"""Test the experiment-matrix runner end-to-end on synthetic data."""

import json

from astromae.evaluation.matrix import run_experiment_matrix


def test_matrix_runs_and_writes_artifacts(tmp_path):
    out = tmp_path / "matrix_out"
    results = run_experiment_matrix(
        out,
        n_per_class=16,
        num_classes=2,
        size=32,
        seed=0,
        steps=8,
        lr=3e-3,
    )
    tags = {r.tag for r in results}
    assert tags == {"mae", "mae-iv", "mae-c4", "mae-c4-iv", "jepa"}

    for r in results:
        assert 0.0 <= r.morphology_accuracy <= 1.0
        assert r.param_count > 0
        assert r.wall_clock_sec >= 0
        assert r.equivariance_error >= 0

    data = json.loads((out / "results.json").read_text())
    assert len(data) == 5
    md = (out / "results.md").read_text()
    assert "# Experiment matrix" in md


def test_matrix_c4_has_smaller_equivariance_error_than_vanilla_mae(tmp_path):
    out = tmp_path / "matrix_out"
    results = run_experiment_matrix(
        out,
        n_per_class=16,
        num_classes=2,
        size=32,
        seed=0,
        steps=4,
        lr=3e-3,
    )
    by_tag = {r.tag: r for r in results}
    # C4 variants should have lower equivariance error than vanilla MAE.
    assert by_tag["mae-c4"].equivariance_error <= by_tag["mae"].equivariance_error
    assert by_tag["mae-c4-iv"].equivariance_error <= by_tag["mae"].equivariance_error
