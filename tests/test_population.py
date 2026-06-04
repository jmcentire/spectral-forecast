"""Tests for robust population nominals."""

import numpy as np

from spectral_forecast.population import (
    fit_population_nominal,
    population_deviance_matrix,
    population_feature_deviance_tensor,
    population_signed_deviance_tensor,
)


def test_population_deviance_surfaces_persistent_heldout_difference() -> None:
    rng = np.random.default_rng(10)
    train = [
        rng.normal(0.0, 0.2, (100, 3, 4)),
        rng.normal(0.0, 0.2, (100, 3, 4)),
    ]
    nominal = fit_population_nominal(
        train,
        entity_names=("a", "b", "c"),
        feature_names=("f0", "f1", "f2", "f3"),
    )
    ordinary = rng.normal(0.0, 0.2, (40, 3, 4))
    shifted = ordinary.copy()
    shifted[:, 1, :] += 3.0

    ordinary_score = population_deviance_matrix(ordinary, nominal)
    shifted_score = population_deviance_matrix(shifted, nominal)

    assert float(np.median(shifted_score[:, 1])) > float(np.median(ordinary_score[:, 1])) + 5.0
    feature_z = population_feature_deviance_tensor(shifted, nominal)
    assert feature_z.shape == shifted.shape
    signed_z = population_signed_deviance_tensor(shifted, nominal)
    assert float(np.median(signed_z[:, 1])) > 5.0
    assert np.allclose(np.abs(signed_z), feature_z)


def test_population_nominal_requires_matching_geometry() -> None:
    nominal = fit_population_nominal(
        [np.zeros((10, 2, 3))],
        entity_names=("a", "b"),
        feature_names=("x", "y", "z"),
    )

    try:
        population_deviance_matrix(np.zeros((10, 3, 3)), nominal)
    except ValueError as exc:
        assert "population tensors" in str(exc)
    else:
        raise AssertionError("expected a geometry mismatch")
