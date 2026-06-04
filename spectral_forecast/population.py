"""Robust population nominals for generic entity-feature tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PopulationNominal:
    """Robust per-entity, per-feature reference fitted across observations."""

    entity_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    centers: NDArray[np.float64]
    scales: NDArray[np.float64]
    observations: int
    source_groups: int

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["centers_shape"] = list(self.centers.shape)
        row["scales_shape"] = list(self.scales.shape)
        row.pop("centers")
        row.pop("scales")
        return row


def _validate_tensors(
    tensors: Sequence[NDArray[np.floating]],
    *,
    entity_names: Sequence[str],
    feature_names: Sequence[str],
) -> list[NDArray[np.float64]]:
    if not tensors:
        raise ValueError("at least one population tensor is required")
    expected = (len(entity_names), len(feature_names))
    if expected[0] < 1 or expected[1] < 1:
        raise ValueError("entity_names and feature_names must be non-empty")
    out = []
    for tensor in tensors:
        values = np.asarray(tensor, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != expected:
            raise ValueError(
                "population tensors must have shape "
                f"(observations, {expected[0]}, {expected[1]})"
            )
        if len(values) < 1:
            raise ValueError("population tensors must contain observations")
        if not np.all(np.isfinite(values)):
            raise ValueError("population tensors must contain only finite values")
        out.append(values)
    return out


def fit_population_nominal(
    tensors: Sequence[NDArray[np.floating]],
    *,
    entity_names: Sequence[str],
    feature_names: Sequence[str],
    min_scale: float = 1e-6,
) -> PopulationNominal:
    """Fit a robust population reference across grouped observations."""

    if min_scale <= 0:
        raise ValueError("min_scale must be positive")
    validated = _validate_tensors(
        tensors,
        entity_names=entity_names,
        feature_names=feature_names,
    )
    pooled = np.concatenate(validated, axis=0)
    centers = np.median(pooled, axis=0)
    mad = np.median(np.abs(pooled - centers[None, :, :]), axis=0)
    scales = 1.4826 * mad
    fallback = np.std(pooled, axis=0)
    scales = np.where(scales >= min_scale, scales, fallback)
    scales = np.maximum(scales, min_scale)
    return PopulationNominal(
        entity_names=tuple(entity_names),
        feature_names=tuple(feature_names),
        centers=np.asarray(centers, dtype=np.float64),
        scales=np.asarray(scales, dtype=np.float64),
        observations=int(len(pooled)),
        source_groups=len(validated),
    )


def population_deviance_matrix(
    tensor: NDArray[np.floating],
    nominal: PopulationNominal,
    *,
    feature_quantile: float = 0.9,
) -> NDArray[np.float64]:
    """Score each observation/entity against a frozen population nominal."""

    if not 0.5 <= feature_quantile <= 1.0:
        raise ValueError("feature_quantile must be in [0.5, 1.0]")
    values = _validate_tensors(
        [tensor],
        entity_names=nominal.entity_names,
        feature_names=nominal.feature_names,
    )[0]
    z = population_feature_deviance_tensor(values, nominal)
    if feature_quantile == 1.0:
        return np.max(z, axis=2).astype(np.float64)
    return np.quantile(z, feature_quantile, axis=2).astype(np.float64)


def population_feature_deviance_tensor(
    tensor: NDArray[np.floating],
    nominal: PopulationNominal,
) -> NDArray[np.float64]:
    """Return absolute robust population deviation per entity and feature."""

    return np.abs(population_signed_deviance_tensor(tensor, nominal)).astype(np.float64)


def population_signed_deviance_tensor(
    tensor: NDArray[np.floating],
    nominal: PopulationNominal,
) -> NDArray[np.float64]:
    """Return signed robust population deviation per entity and feature."""

    values = _validate_tensors(
        [tensor],
        entity_names=nominal.entity_names,
        feature_names=nominal.feature_names,
    )[0]
    return (
        (values - nominal.centers[None, :, :])
        / nominal.scales[None, :, :]
    ).astype(np.float64)
