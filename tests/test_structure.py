"""Tests for multichannel structure-readiness diagnostics."""

import numpy as np

from spectral_forecast.structure import covariance_entropy, structure_readiness


def test_correlated_channels_have_lower_covariance_entropy_than_independent_noise() -> None:
    rng = np.random.default_rng(12)
    t = np.arange(512, dtype=np.float64)
    latent = np.sin(2 * np.pi * 0.04 * t)
    correlated = {
        f"c{i}": latent + rng.normal(0.0, 0.12, len(t))
        for i in range(5)
    }
    independent = {
        f"n{i}": rng.normal(0.0, 1.0, len(t))
        for i in range(5)
    }

    corr_matrix = np.column_stack(list(correlated.values()))
    noise_matrix = np.column_stack(list(independent.values()))

    assert covariance_entropy(corr_matrix) < covariance_entropy(noise_matrix)


def test_structure_readiness_accepts_shared_latent_structure() -> None:
    rng = np.random.default_rng(23)
    t = np.arange(640, dtype=np.float64)
    latent = np.sin(2 * np.pi * 0.03 * t)
    series = {
        f"s{i}": latent + 0.3 * np.sin(2 * np.pi * (0.05 + i * 0.005) * t) + rng.normal(0.0, 0.1, len(t))
        for i in range(6)
    }

    readiness = structure_readiness(series, null_repeats=24, seed=5)

    assert readiness.ready
    assert readiness.structure_score > 0.35
    assert readiness.covariance_entropy_deficit > 0.0
    assert readiness.covariance_z_effect is not None
    assert readiness.covariance_z_effect > 2.0


def test_structure_readiness_rejects_independent_noise() -> None:
    rng = np.random.default_rng(31)
    series = {
        f"n{i}": rng.normal(0.0, 1.0, 640)
        for i in range(6)
    }

    readiness = structure_readiness(series, null_repeats=24, seed=6)

    assert not readiness.ready
    assert readiness.reason in {"no_null_separation", "weak_structure"}
