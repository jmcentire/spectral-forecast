"""Identity-aware and permutation-invariant relationship dynamics."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from spectral_forecast.relationships import (
    _effective_nperseg,
    _lagged_dependence_metric,
    _robust_standardize_matrix,
    _spectral_alignment_metric,
)

RelationshipFamily = Literal["correlation", "spectral_alignment", "lagged_dependence"]
DynamicsMode = Literal["step", "drift"]
DynamicsMetric = Literal["exact_identity", "structural_role", "global_motif"]


@dataclass(frozen=True)
class RelationshipGraphSequence:
    """Windowed relationship graphs on a canonical entity order."""

    entities: tuple[str, ...]
    anchors: tuple[int, ...]
    family: RelationshipFamily
    window_size: int
    stride: int
    graphs: NDArray[np.float64]


@dataclass(frozen=True)
class RelationshipDynamicsEvidence:
    """Observed-versus-null evidence for one dynamics mode and graph metric."""

    mode: DynamicsMode
    metric: DynamicsMetric
    observed: float
    null_mean: float
    null_std: float
    observed_minus_null: float
    z_effect: float | None
    null_exceedances: int
    null_repeats: int
    empirical_p_ge_observed: float
    empirical_p_floor: float
    fdr_by_q_value: float
    detected_fdr_by: bool
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationshipDynamicsResult:
    """Step and drift evidence for exact identities, roles, and motifs."""

    entities: tuple[str, ...]
    graph_windows: int
    family: RelationshipFamily
    window_size: int
    stride: int
    min_segment_windows: int
    null_block_size: int
    null_block_estimated: bool
    requested_null_repeats: int
    null_repeats: int
    null_permutation_space: int
    null_exhaustive: bool
    evidence: tuple[RelationshipDynamicsEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": list(self.entities),
            "graph_windows": self.graph_windows,
            "family": self.family,
            "window_size": self.window_size,
            "stride": self.stride,
            "min_segment_windows": self.min_segment_windows,
            "null_block_size": self.null_block_size,
            "null_block_estimated": self.null_block_estimated,
            "requested_null_repeats": self.requested_null_repeats,
            "null_repeats": self.null_repeats,
            "null_permutation_space": self.null_permutation_space,
            "null_exhaustive": self.null_exhaustive,
            "evidence": [row.to_dict() for row in self.evidence],
        }


def relationship_adjacency(
    matrix: NDArray[np.floating],
    *,
    family: RelationshipFamily = "correlation",
    sample_rate: float = 1.0,
    nperseg: int = 64,
    max_lag: int = 16,
) -> NDArray[np.float64]:
    """Construct one symmetric weighted relationship graph."""

    x = _robust_standardize_matrix(matrix)
    count = x.shape[1]
    adjacency = np.zeros((count, count), dtype=np.float64)
    if family == "correlation":
        adjacency = np.asarray(np.corrcoef(x, rowvar=False), dtype=np.float64)
        adjacency = np.where(np.isfinite(adjacency), adjacency, 0.0)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency
    effective_nperseg = _effective_nperseg(len(x), nperseg)
    for left in range(count):
        for right in range(left + 1, count):
            if family == "spectral_alignment":
                value = _spectral_alignment_metric(
                    x[:, left],
                    x[:, right],
                    sample_rate=sample_rate,
                    nperseg=effective_nperseg,
                )[0]
            elif family == "lagged_dependence":
                _, attributes = _lagged_dependence_metric(
                    x[:, left],
                    x[:, right],
                    max_lag=max_lag,
                    sample_rate=sample_rate,
                )
                value = abs(float(attributes["best_lag_correlation"]))
            else:
                raise ValueError(f"unknown relationship family: {family}")
            adjacency[left, right] = value
            adjacency[right, left] = value
    return adjacency


def window_relationship_graphs(
    matrix: NDArray[np.floating],
    entity_names: Sequence[str],
    *,
    family: RelationshipFamily = "correlation",
    window_size: int = 12,
    stride: int = 3,
    sample_rate: float = 1.0,
    nperseg: int = 64,
    max_lag: int = 16,
) -> RelationshipGraphSequence:
    """Build past-window relationship graphs without using labels."""

    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2D")
    if len(entity_names) != x.shape[1]:
        raise ValueError("entity_names must match matrix columns")
    if window_size < 32:
        raise ValueError("relationship graph windows need at least 32 rows")
    if stride < 1:
        raise ValueError("stride must be positive")
    if len(x) < window_size:
        raise ValueError("matrix is shorter than the relationship window")
    anchors = tuple(range(window_size, len(x) + 1, stride))
    graphs = np.asarray(
        [
            relationship_adjacency(
                x[anchor - window_size : anchor],
                family=family,
                sample_rate=sample_rate,
                nperseg=nperseg,
                max_lag=max_lag,
            )
            for anchor in anchors
        ],
        dtype=np.float64,
    )
    return RelationshipGraphSequence(
        entities=tuple(str(name) for name in entity_names),
        anchors=anchors,
        family=family,
        window_size=window_size,
        stride=stride,
        graphs=graphs,
    )


def exact_identity_distance(
    left: NDArray[np.floating], right: NDArray[np.floating]
) -> float:
    """RMS edge change with entity identities held fixed."""

    a, b = _validated_graph_pair(left, right)
    upper = np.triu_indices(len(a), k=1)
    return float(np.sqrt(np.mean((a[upper] - b[upper]) ** 2)))


def _role_signatures(graph: NDArray[np.float64]) -> NDArray[np.float64]:
    rows = []
    for index in range(len(graph)):
        incident = np.delete(graph[index], index)
        ordered = np.sort(incident)
        rows.append(
            np.concatenate(
                [
                    ordered,
                    np.asarray(
                        [
                            np.mean(incident),
                            np.std(incident),
                            np.sum(np.abs(incident)),
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
        )
    return np.asarray(rows, dtype=np.float64)


def structural_role_distance(
    left: NDArray[np.floating], right: NDArray[np.floating]
) -> tuple[float, tuple[int, ...]]:
    """Minimum node-role distance after allowing entity substitution."""

    a, b = _validated_graph_pair(left, right)
    left_roles = _role_signatures(a)
    right_roles = _role_signatures(b)
    scale = max(1.0, float(np.sqrt(left_roles.shape[1])))
    costs = np.linalg.norm(
        left_roles[:, None, :] - right_roles[None, :, :], axis=2
    ) / scale
    left_indices, right_indices = linear_sum_assignment(costs)
    assignment = np.full(len(a), -1, dtype=np.int64)
    assignment[left_indices] = right_indices
    return float(np.mean(costs[left_indices, right_indices])), tuple(int(x) for x in assignment)


def _motif_signature(graph: NDArray[np.float64]) -> NDArray[np.float64]:
    symmetric = (graph + graph.T) / 2.0
    strengths = np.sort(np.sum(symmetric, axis=1))
    absolute_strengths = np.sort(np.sum(np.abs(symmetric), axis=1))
    eigenvalues = np.sort(np.linalg.eigvalsh(symmetric))
    triangle = float(np.trace(symmetric @ symmetric @ symmetric) / 6.0)
    return np.concatenate(
        [strengths, absolute_strengths, eigenvalues, np.asarray([triangle])]
    )


def global_motif_distance(
    left: NDArray[np.floating], right: NDArray[np.floating]
) -> float:
    """Permutation-invariant distance between global graph signatures."""

    a, b = _validated_graph_pair(left, right)
    delta = _motif_signature(a) - _motif_signature(b)
    return float(np.linalg.norm(delta) / max(1.0, np.sqrt(len(delta))))


def describe_step_change(
    sequence: RelationshipGraphSequence,
    split_graph_index: int,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Describe the edges and entities carrying a frozen graph split."""

    graphs = np.asarray(sequence.graphs, dtype=np.float64)
    if not 0 < split_graph_index < len(graphs):
        raise ValueError("split_graph_index must divide the graph sequence")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    left = np.mean(graphs[:split_graph_index], axis=0)
    right = np.mean(graphs[split_graph_index:], axis=0)
    delta = right - left
    upper = np.triu_indices(len(sequence.entities), k=1)
    motif_delta = _motif_signature(right) - _motif_signature(left)
    order = np.argsort(np.abs(delta[upper]))[::-1]
    edges = []
    for position in order[:top_k]:
        left_index = int(upper[0][position])
        right_index = int(upper[1][position])
        edges.append(
            {
                "left_entity": sequence.entities[left_index],
                "right_entity": sequence.entities[right_index],
                "before": float(left[left_index, right_index]),
                "after": float(right[left_index, right_index]),
                "delta": float(delta[left_index, right_index]),
                "absolute_delta": float(abs(delta[left_index, right_index])),
            }
        )
    entity_change = np.sum(np.abs(delta), axis=1)
    entity_order = np.argsort(entity_change)[::-1]
    total = float(np.sum(entity_change))
    entities = [
        {
            "entity": sequence.entities[int(index)],
            "absolute_change": float(entity_change[int(index)]),
            "fraction": float(entity_change[int(index)] / total) if total else 0.0,
        }
        for index in entity_order[:top_k]
    ]
    role_distance, role_assignment = structural_role_distance(left, right)
    return {
        "split_graph_index": split_graph_index,
        "split_anchor": sequence.anchors[split_graph_index],
        "top_changed_edges": edges,
        "top_changed_entities": entities,
        "top_entity_fraction": entities[0]["fraction"] if entities else 0.0,
        "top_three_entity_fraction": float(
            sum(row["fraction"] for row in entities[:3])
        ),
        "edge_delta_upper": [float(value) for value in delta[upper]],
        "motif_signature_delta": [float(value) for value in motif_delta],
        "role_distance": role_distance,
        "role_assignment": list(role_assignment),
        "global_motif_distance": global_motif_distance(left, right),
    }


def _validated_graph_pair(
    left: NDArray[np.floating], right: NDArray[np.floating]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape != b.shape:
        raise ValueError("graphs must be equal square matrices")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("graphs must contain only finite values")
    return a, b


def _distance(metric: DynamicsMetric, left: NDArray, right: NDArray) -> float:
    if metric == "exact_identity":
        return exact_identity_distance(left, right)
    if metric == "structural_role":
        return structural_role_distance(left, right)[0]
    if metric == "global_motif":
        return global_motif_distance(left, right)
    raise ValueError(f"unknown dynamics metric: {metric}")


def _step_statistics(
    graphs: NDArray[np.float64], min_segment_windows: int
) -> dict[DynamicsMetric, tuple[float, int]]:
    metrics: tuple[DynamicsMetric, ...] = (
        "exact_identity",
        "structural_role",
        "global_motif",
    )
    best_exact = -np.inf
    best_split = -1
    for split in range(min_segment_windows, len(graphs) - min_segment_windows + 1):
        left = np.mean(graphs[:split], axis=0)
        right = np.mean(graphs[split:], axis=0)
        scale = float(np.sqrt(split * (len(graphs) - split) / len(graphs)))
        value = _distance("exact_identity", left, right) * scale
        if value > best_exact:
            best_exact = value
            best_split = split
    left = np.mean(graphs[:best_split], axis=0)
    right = np.mean(graphs[best_split:], axis=0)
    scale = float(
        np.sqrt(best_split * (len(graphs) - best_split) / len(graphs))
    )
    return {
        metric: (_distance(metric, left, right) * scale, best_split)
        for metric in metrics
    }


def _drift_statistics(
    graphs: NDArray[np.float64], min_segment_windows: int
) -> dict[DynamicsMetric, tuple[float, float]]:
    metrics: tuple[DynamicsMetric, ...] = (
        "exact_identity",
        "structural_role",
        "global_motif",
    )
    baseline = np.mean(graphs[:min_segment_windows], axis=0)
    times = np.arange(len(graphs), dtype=np.float64)
    out = {}
    for metric in metrics:
        distances = np.asarray(
            [_distance(metric, baseline, graph) for graph in graphs], dtype=np.float64
        )
        distances[np.abs(distances) < 1e-12] = 0.0
        if float(np.ptp(distances)) < 1e-12:
            correlation = 0.0
        else:
            correlation = float(spearmanr(times, distances).statistic)
        if not np.isfinite(correlation):
            correlation = 0.0
        out[metric] = (abs(correlation), correlation)
    return out


def _unique_block_permutation_orders(
    length: int,
    block_size: int,
    requested: int,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    block_count = math.ceil(length / block_size)
    identity = tuple(range(block_count))
    permutation_space = math.factorial(block_count) - 1
    if permutation_space < 1:
        raise ValueError("null block size leaves fewer than two graph blocks")
    target = min(requested, permutation_space)
    if target == permutation_space and block_count <= 9:
        orders = [
            order
            for order in itertools.permutations(range(block_count))
            if order != identity
        ]
        rng.shuffle(orders)
        return tuple(orders), permutation_space

    seen = {identity}
    orders = []
    while len(orders) < target:
        order = tuple(int(index) for index in rng.permutation(block_count))
        if order in seen:
            continue
        seen.add(order)
        orders.append(order)
    return tuple(orders), permutation_space


def _block_permutation_indices(
    length: int, block_size: int, order: Sequence[int]
) -> NDArray[np.int64]:
    blocks = [
        np.arange(start, min(start + block_size, length), dtype=np.int64)
        for start in range(0, length, block_size)
    ]
    return np.concatenate([blocks[int(index)] for index in order])


def _apply_by_fdr(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        (float(row["empirical_p_ge_observed"]), index)
        for index, row in enumerate(rows)
    )
    total = len(ordered)
    harmonic = sum(1.0 / rank for rank in range(1, total + 1))
    running = 1.0
    adjusted: dict[int, float] = {}
    for rank in range(total, 0, -1):
        p_value, index = ordered[rank - 1]
        running = min(running, p_value * total / rank)
        adjusted[index] = min(1.0, running * harmonic)
    for index, row in enumerate(rows):
        row["fdr_by_q_value"] = adjusted[index]
        row["detected_fdr_by"] = bool(
            adjusted[index] <= 0.05 and row["observed_minus_null"] > 0.0
        )


def estimate_relationship_block_size(
    sequence: RelationshipGraphSequence,
    *,
    correlation_threshold: float = 0.2,
    minimum: int = 2,
    maximum_fraction: float = 0.25,
) -> int:
    """Estimate a conservative graph-window dependence block length."""

    graphs = np.asarray(sequence.graphs, dtype=np.float64)
    if graphs.ndim != 3 or graphs.shape[1] != graphs.shape[2]:
        raise ValueError("sequence graphs must have shape windows x entities x entities")
    if not 0.0 <= correlation_threshold < 1.0:
        raise ValueError("correlation_threshold must be in [0, 1)")
    if minimum < 1 or not 0.0 < maximum_fraction <= 0.5:
        raise ValueError("invalid block-size bounds")
    upper = np.triu_indices(graphs.shape[1], k=1)
    vectors = graphs[:, upper[0], upper[1]]
    centered = vectors - np.mean(vectors, axis=0, keepdims=True)
    varying = np.std(centered, axis=0) > 1e-12
    if not np.any(varying):
        return min(minimum, max(1, len(graphs) // 2))
    centered = centered[:, varying]
    max_lag = max(minimum, min(len(graphs) // 4, 32))
    max_lag = min(max_lag, max(1, len(graphs) // 2))
    for lag in range(1, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = float(np.sum(left * right) / denominator) if denominator else 0.0
        if abs(correlation) <= correlation_threshold:
            return max(minimum, lag)
    fraction_cap = max(minimum, int(np.floor(len(graphs) * maximum_fraction)))
    return min(max_lag, fraction_cap)


def analyze_relationship_dynamics(
    sequence: RelationshipGraphSequence,
    *,
    min_segment_windows: int = 4,
    null_repeats: int = 999,
    null_block_size: int | None = None,
    seed: int = 20260605,
) -> RelationshipDynamicsResult:
    """Test step and drift structure with scan-aware block-permutation nulls."""

    graphs = np.asarray(sequence.graphs, dtype=np.float64)
    if graphs.ndim != 3 or graphs.shape[1] != graphs.shape[2]:
        raise ValueError("sequence graphs must have shape windows x entities x entities")
    if len(graphs) < 2 * min_segment_windows:
        raise ValueError("not enough graph windows for the requested segment gate")
    if null_repeats < 1:
        raise ValueError("null repeats must be positive")
    block_estimated = null_block_size is None
    if null_block_size is None:
        null_block_size = estimate_relationship_block_size(sequence)
    if null_block_size < 1:
        raise ValueError("null block size must be positive")

    step = _step_statistics(graphs, min_segment_windows)
    drift = _drift_statistics(graphs, min_segment_windows)
    nulls: dict[tuple[DynamicsMode, DynamicsMetric], list[float]] = {
        (mode, metric): []
        for mode in ("step", "drift")
        for metric in ("exact_identity", "structural_role", "global_motif")
    }
    rng = np.random.default_rng(seed)
    null_orders, permutation_space = _unique_block_permutation_orders(
        len(graphs), null_block_size, null_repeats, rng
    )
    for order in null_orders:
        permuted = graphs[
            _block_permutation_indices(len(graphs), null_block_size, order)
        ]
        null_step = _step_statistics(permuted, min_segment_windows)
        null_drift = _drift_statistics(permuted, min_segment_windows)
        for metric in ("exact_identity", "structural_role", "global_motif"):
            nulls[("step", metric)].append(null_step[metric][0])
            nulls[("drift", metric)].append(null_drift[metric][0])

    rows: list[dict[str, Any]] = []
    for mode in ("step", "drift"):
        for metric in ("exact_identity", "structural_role", "global_motif"):
            if mode == "step":
                observed, split = step[metric]
                attributes = {
                    "best_split_graph_index": split,
                    "best_split_anchor": sequence.anchors[split],
                    "scan_corrected": True,
                }
            else:
                observed, signed_correlation = drift[metric]
                attributes = {
                    "signed_spearman_correlation": signed_correlation,
                    "baseline_graph_windows": min_segment_windows,
                }
            null = np.asarray(nulls[(mode, metric)], dtype=np.float64)
            mean = float(np.mean(null))
            std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
            delta = float(observed - mean)
            exceedances = int(np.sum(null >= observed))
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "observed": float(observed),
                    "null_mean": mean,
                    "null_std": std,
                    "observed_minus_null": delta,
                    "z_effect": delta / std if std > 1e-12 else None,
                    "null_exceedances": exceedances,
                    "null_repeats": len(null),
                    "empirical_p_ge_observed": float(
                        (exceedances + 1) / (len(null) + 1)
                    ),
                    "empirical_p_floor": float(1.0 / (len(null) + 1)),
                    "attributes": attributes,
                }
            )
    _apply_by_fdr(rows)
    evidence = tuple(RelationshipDynamicsEvidence(**row) for row in rows)
    return RelationshipDynamicsResult(
        entities=sequence.entities,
        graph_windows=len(graphs),
        family=sequence.family,
        window_size=sequence.window_size,
        stride=sequence.stride,
        min_segment_windows=min_segment_windows,
        null_block_size=null_block_size,
        null_block_estimated=block_estimated,
        requested_null_repeats=null_repeats,
        null_repeats=len(null_orders),
        null_permutation_space=permutation_space,
        null_exhaustive=len(null_orders) == permutation_space,
        evidence=evidence,
    )
