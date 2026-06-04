"""Label-free autotune adapter for temporal organizational networks.

This converts timestamped edge streams into synchronized numeric time series,
then delegates discovery to the existing spectral+stigmergy autotune core.
Known organizational labels, when provided, describe components or support
post-hoc attribution. They are not event labels and are not used as positives.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numpy.typing import NDArray

from spectral_forecast.autotune import (
    AutoTuneConfig,
    AutoTuneObservation,
    AutoTuneScore,
    build_autotune_observation,
    observation_null_totals_for_config,
    score_autotune_observation,
)
from spectral_forecast.structure import structure_readiness


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    edge_url: str
    label_url: str | None
    edge_file: str
    label_file: str | None
    data_format: str
    source_col: int
    target_col: int
    time_col: int
    directed: bool
    notes: str


@dataclass(frozen=True)
class TemporalEdge:
    source: str
    target: str
    timestamp: float


def _edge_pair(edge: TemporalEdge, *, directed: bool) -> tuple[str, str]:
    return (edge.source, edge.target) if directed else tuple(sorted((edge.source, edge.target)))


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def remap_edges_by_order_with_mapping(
    edges: Sequence[TemporalEdge],
    *,
    order: str,
    labels: Mapping[str, str],
    directed: bool,
    seed: int,
) -> tuple[list[TemporalEdge], list[int]]:
    """Replace timestamps with event ranks and retain original edge indices."""

    indexed = list(enumerate(edges))
    if order == "random_order":
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(indexed))
        ordered = [indexed[int(index)] for index in permutation]
    else:
        degrees: dict[str, int] = {}
        first_seen: dict[tuple[str, str], float] = {}
        for edge in edges:
            degrees[edge.source] = degrees.get(edge.source, 0) + 1
            degrees[edge.target] = degrees.get(edge.target, 0) + 1
            key = _edge_pair(edge, directed=directed)
            first_seen[key] = min(first_seen.get(key, float("inf")), float(edge.timestamp))

        def sort_key(item: tuple[int, TemporalEdge]) -> tuple[object, ...]:
            index, edge = item
            pair = _edge_pair(edge, directed=directed)
            pair_text = "|".join(pair)
            source_group = labels.get(edge.source, "unknown")
            target_group = labels.get(edge.target, "unknown")
            if order in {"timestamp", "timestamp_bucket_partial_order"}:
                return (float(edge.timestamp), index)
            if order == "first_seen_pair":
                return (first_seen[pair], float(edge.timestamp), pair_text, index)
            if order == "degree_descending":
                degree = degrees.get(edge.source, 0) + degrees.get(edge.target, 0)
                return (-degree, float(edge.timestamp), pair_text, index)
            if order == "group_then_time":
                group_pair = tuple(sorted((source_group, target_group)))
                return (group_pair, float(edge.timestamp), pair_text, index)
            if order == "stable_id_or_alphabetic":
                return (pair_text, float(edge.timestamp), index)
            if order == "hash_order":
                return (_stable_hash(pair_text), float(edge.timestamp), index)
            if order == "silly_proxy_order":
                digit_sevens = pair_text.count("7")
                return (len(pair_text), -digit_sevens, pair_text[::-1], float(edge.timestamp), index)
            raise ValueError(f"unknown enforced order: {order}")

        ordered = sorted(indexed, key=sort_key)

    remapped = [
        TemporalEdge(source=edge.source, target=edge.target, timestamp=float(rank))
        for rank, (_, edge) in enumerate(ordered)
    ]
    return remapped, [index for index, _ in ordered]


def remap_edges_by_order(
    edges: Sequence[TemporalEdge],
    *,
    order: str,
    labels: Mapping[str, str],
    directed: bool,
    seed: int,
) -> list[TemporalEdge]:
    """Replace timestamps with event ranks after sorting by an enforced order."""

    remapped, _ = remap_edges_by_order_with_mapping(
        edges,
        order=order,
        labels=labels,
        directed=directed,
        seed=seed,
    )
    return remapped


@dataclass(frozen=True)
class NetworkSeries:
    series: dict[str, NDArray[np.float64]]
    raw_series: dict[str, NDArray[np.float64]]
    dropped_series: list[str]
    metadata: dict[str, Any]


class ObservationCache:
    """Cache observer matrices across threshold/decay/min-active candidates."""

    def __init__(self) -> None:
        self.entries: dict[tuple[object, ...], AutoTuneObservation] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[object, ...]) -> AutoTuneObservation | None:
        value = self.entries.get(key)
        if value is not None:
            self.hits += 1
            return value
        self.misses += 1
        return None

    def put(self, key: tuple[object, ...], observation: AutoTuneObservation) -> None:
        self.entries[key] = observation

    def to_dict(self) -> dict[str, int]:
        return {"entries": len(self.entries), "hits": self.hits, "misses": self.misses}


DATASETS: dict[str, DatasetSpec] = {
    "email-eu": DatasetSpec(
        name="email-eu",
        edge_url="https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz",
        label_url=None,
        edge_file="email-Eu-core-temporal.txt.gz",
        label_file=None,
        data_format="edges",
        source_col=0,
        target_col=1,
        time_col=2,
        directed=True,
        notes=(
            "SNAP temporal Email-Eu full graph. Static department labels do not "
            "share node IDs with this temporal graph; do not use them here."
        ),
    ),
    "email-eu-dept1": DatasetSpec(
        name="email-eu-dept1",
        edge_url="https://snap.stanford.edu/data/email-Eu-core-temporal-Dept1.txt.gz",
        label_url=None,
        edge_file="email-Eu-core-temporal-Dept1.txt.gz",
        label_file=None,
        data_format="edges",
        source_col=0,
        target_col=1,
        time_col=2,
        directed=True,
        notes="SNAP temporal Email-Eu Department 1 subnetwork; IDs are local to this subnetwork.",
    ),
    "email-eu-dept2": DatasetSpec(
        name="email-eu-dept2",
        edge_url="https://snap.stanford.edu/data/email-Eu-core-temporal-Dept2.txt.gz",
        label_url=None,
        edge_file="email-Eu-core-temporal-Dept2.txt.gz",
        label_file=None,
        data_format="edges",
        source_col=0,
        target_col=1,
        time_col=2,
        directed=True,
        notes="SNAP temporal Email-Eu Department 2 subnetwork; IDs are local to this subnetwork.",
    ),
    "email-eu-dept3": DatasetSpec(
        name="email-eu-dept3",
        edge_url="https://snap.stanford.edu/data/email-Eu-core-temporal-Dept3.txt.gz",
        label_url=None,
        edge_file="email-Eu-core-temporal-Dept3.txt.gz",
        label_file=None,
        data_format="edges",
        source_col=0,
        target_col=1,
        time_col=2,
        directed=True,
        notes="SNAP temporal Email-Eu Department 3 subnetwork; IDs are local to this subnetwork.",
    ),
    "email-eu-dept4": DatasetSpec(
        name="email-eu-dept4",
        edge_url="https://snap.stanford.edu/data/email-Eu-core-temporal-Dept4.txt.gz",
        label_url=None,
        edge_file="email-Eu-core-temporal-Dept4.txt.gz",
        label_file=None,
        data_format="edges",
        source_col=0,
        target_col=1,
        time_col=2,
        directed=True,
        notes="SNAP temporal Email-Eu Department 4 subnetwork; IDs are local to this subnetwork.",
    ),
    "sociopatterns-workplace": DatasetSpec(
        name="sociopatterns-workplace",
        edge_url="https://sociopatterns.org/assets/data/workplace_InVS15_tij.dat.gz",
        label_url="https://sociopatterns.org/assets/data/workplace_InVS15_metadata.txt",
        edge_file="workplace_InVS15_tij.dat.gz",
        label_file="workplace_InVS15_metadata.txt",
        data_format="edges",
        source_col=1,
        target_col=2,
        time_col=0,
        directed=False,
        notes="SocioPatterns workplace contacts with matching department metadata.",
    ),
    "enron-email-simplices": DatasetSpec(
        name="enron-email-simplices",
        edge_url="https://drive.google.com/uc?export=download&id=1tTVZkdpgRW47WWmsrdUCukHz0x2M6N77",
        label_url=None,
        edge_file="email-Enron.tar.gz",
        label_file=None,
        data_format="simplices",
        source_col=0,
        target_col=1,
        time_col=2,
        directed=False,
        notes=(
            "Cornell temporal higher-order Enron email dataset. Each simplex is "
            "sender plus recipients among core Enron employees; projected to "
            "co-participation edges for this adapter."
        ),
    ),
}


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _clean_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed dataset URLs.
        tmp.write_bytes(response.read())
    tmp.replace(path)


def _safe_extract_tar(path: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (directory / member.name).resolve()
            if not str(target).startswith(str(directory.resolve())):
                raise ValueError(f"archive member escapes extraction directory: {member.name}")
        archive.extractall(directory)


def resolve_dataset_files(args: argparse.Namespace) -> tuple[Path, Path | None, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if args.dataset:
        spec = DATASETS[args.dataset]
        dataset_dir = args.data_dir / spec.name
        edge_path = dataset_dir / spec.edge_file
        label_path = dataset_dir / spec.label_file if spec.label_file else None
        if args.download:
            if not edge_path.exists() or args.force_download:
                _download(spec.edge_url, edge_path)
            if spec.label_url and label_path is not None and (
                not label_path.exists() or args.force_download
            ):
                _download(spec.label_url, label_path)
        metadata.update(
            {
                "dataset": spec.name,
                "edge_url": spec.edge_url,
                "label_url": spec.label_url,
                "data_format": spec.data_format,
                "notes": spec.notes,
            }
        )
        args.source_col = spec.source_col
        args.target_col = spec.target_col
        args.time_col = spec.time_col
        args.directed = spec.directed
        if spec.data_format == "simplices":
            extracted = dataset_dir / "email-Enron"
            if not extracted.exists():
                _safe_extract_tar(edge_path, dataset_dir)
            return extracted, label_path, metadata
        return edge_path, label_path, metadata

    if args.edges is None:
        raise ValueError("either --dataset or --edges is required")
    metadata["data_format"] = args.data_format
    return args.edges, args.labels, metadata


def read_edges(
    path: Path,
    *,
    source_col: int,
    target_col: int,
    time_col: int,
    max_edges: int,
) -> list[TemporalEdge]:
    edges: list[TemporalEdge] = []
    max_col = max(source_col, target_col, time_col)
    with _open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) <= max_col:
                continue
            try:
                timestamp = float(parts[time_col])
            except ValueError:
                continue
            edges.append(
                TemporalEdge(
                    source=parts[source_col],
                    target=parts[target_col],
                    timestamp=timestamp,
                )
            )
            if max_edges > 0 and len(edges) >= max_edges:
                break
    edges.sort(key=lambda edge: edge.timestamp)
    return edges


def read_simplicial_edges(
    directory: Path,
    *,
    prefix: str,
    max_edges: int,
) -> list[TemporalEdge]:
    """Project timestamped simplices to unordered co-participation edges."""

    nverts_path = directory / f"{prefix}-nverts.txt"
    simplices_path = directory / f"{prefix}-simplices.txt"
    times_path = directory / f"{prefix}-times.txt"
    if not nverts_path.exists() or not simplices_path.exists() or not times_path.exists():
        raise ValueError(f"missing simplicial dataset files under {directory}")

    nverts = [int(line.strip()) for line in nverts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    times = [float(line.strip()) for line in times_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    nodes = [
        line.strip()
        for line in simplices_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(nverts) != len(times):
        raise ValueError("simplicial nverts and times files have different lengths")
    if sum(nverts) != len(nodes):
        raise ValueError("simplicial node count does not match nverts sum")

    edges: list[TemporalEdge] = []
    cursor = 0
    for simplex_size, timestamp in zip(nverts, times, strict=True):
        simplex = nodes[cursor : cursor + simplex_size]
        cursor += simplex_size
        if len(simplex) < 2:
            continue
        for source, target in itertools.combinations(sorted(set(simplex)), 2):
            edges.append(TemporalEdge(source=source, target=target, timestamp=timestamp))
            if max_edges > 0 and len(edges) >= max_edges:
                edges.sort(key=lambda edge: edge.timestamp)
                return edges
    edges.sort(key=lambda edge: edge.timestamp)
    return edges


def read_labels(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with _open_text(path) as handle:
        tokens = handle.read().replace(",", " ").split()
    labels: dict[str, str] = {}
    for index in range(0, len(tokens) - 1, 2):
        labels[tokens[index]] = tokens[index + 1]
    return labels


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log(p)
    return float(value / math.log(len(counter)))


def _concentration(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return float(sum((count / total) ** 2 for count in counter.values()))


def _robust_standardize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values, dtype=np.float64)
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = 1.4826 * mad
    if scale < 1e-10:
        scale = float(np.std(finite))
    if scale < 1e-10:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray((values - center) / scale, dtype=np.float64)


def _transform(values: NDArray[np.float64], mode: str) -> NDArray[np.float64]:
    transformed = np.asarray(values, dtype=np.float64)
    if mode in ("log-robust", "log"):
        transformed = np.log1p(np.maximum(transformed, 0.0))
    if mode in ("log-robust", "robust"):
        transformed = _robust_standardize(transformed)
    return np.where(np.isfinite(transformed), transformed, 0.0)


def _add_series(raw: dict[str, NDArray[np.float64]], name: str, n_bins: int) -> None:
    raw.setdefault(name, np.zeros(n_bins, dtype=np.float64))


def _bin_index(timestamp: float, *, start: float, bin_seconds: float) -> int:
    return int((timestamp - start) // bin_seconds)


def build_network_series(
    edges: Sequence[TemporalEdge],
    labels: Mapping[str, str],
    *,
    bin_seconds: float,
    top_nodes: int,
    top_groups: int,
    include_global: bool,
    include_nodes: bool,
    include_groups: bool,
    include_relation: bool,
    directed: bool,
    transform: str,
    max_bins: int,
) -> NetworkSeries:
    if not edges:
        raise ValueError("no temporal edges loaded")
    if bin_seconds <= 0:
        raise ValueError("--bin-seconds must be positive")

    start = min(edge.timestamp for edge in edges)
    end = max(edge.timestamp for edge in edges)
    n_bins = int((end - start) // bin_seconds) + 1
    if max_bins > 0:
        n_bins = min(n_bins, max_bins)
    if n_bins < 4:
        raise ValueError("not enough bins for observation")

    included_edges = [
        edge
        for edge in edges
        if 0 <= _bin_index(edge.timestamp, start=start, bin_seconds=bin_seconds) < n_bins
    ]
    node_activity: Counter[str] = Counter()
    group_activity: Counter[str] = Counter()
    for edge in included_edges:
        node_activity[edge.source] += 1
        node_activity[edge.target] += 1
        if labels:
            group_activity[labels.get(edge.source, "unknown")] += 1
            group_activity[labels.get(edge.target, "unknown")] += 1

    selected_nodes = [node for node, _ in node_activity.most_common(max(0, top_nodes))]
    selected_groups = (
        [group for group, _ in group_activity.most_common(max(0, top_groups))]
        if labels
        else []
    )
    selected_node_set = set(selected_nodes)
    selected_group_set = set(selected_groups)

    raw: dict[str, NDArray[np.float64]] = {}
    if include_global:
        for name in (
            "global_total_edges",
            "global_unique_sources",
            "global_unique_targets",
            "global_active_nodes",
            "global_source_concentration",
            "global_target_concentration",
            "global_reciprocity",
        ):
            _add_series(raw, name, n_bins)
        if labels:
            for name in ("global_cross_group_edges", "global_cross_group_ratio", "global_group_entropy"):
                _add_series(raw, name, n_bins)
        if include_relation:
            for name in (
                "global_unique_pairs",
                "global_new_pairs",
                "global_returning_pairs",
                "global_persistent_pairs",
                "global_lost_pairs",
                "global_pair_churn",
                "global_pair_jaccard_to_previous",
                "global_new_pair_ratio",
                "global_returning_pair_ratio",
                "global_active_node_churn",
            ):
                _add_series(raw, name, n_bins)

    if include_nodes:
        for node in selected_nodes:
            prefix = f"node_{_clean_name(node)}"
            _add_series(raw, f"{prefix}_total", n_bins)
            if directed:
                _add_series(raw, f"{prefix}_out", n_bins)
                _add_series(raw, f"{prefix}_in", n_bins)
            if include_relation:
                for suffix in (
                    "neighbor_count",
                    "new_neighbors",
                    "lost_neighbors",
                    "neighbor_churn",
                    "neighbor_jaccard_to_previous",
                ):
                    _add_series(raw, f"{prefix}_{suffix}", n_bins)

    if include_groups:
        for group in selected_groups:
            prefix = f"group_{_clean_name(group)}"
            for suffix in ("activity", "internal_edges", "external_edges"):
                _add_series(raw, f"{prefix}_{suffix}", n_bins)
            if directed:
                _add_series(raw, f"{prefix}_sent", n_bins)
                _add_series(raw, f"{prefix}_received", n_bins)
            if include_relation:
                for suffix in (
                    "unique_internal_pairs",
                    "unique_external_pairs",
                    "new_internal_pairs",
                    "new_external_pairs",
                    "external_pair_churn",
                ):
                    _add_series(raw, f"{prefix}_{suffix}", n_bins)

    sources_by_bin: list[Counter[str]] = [Counter() for _ in range(n_bins)]
    targets_by_bin: list[Counter[str]] = [Counter() for _ in range(n_bins)]
    groups_by_bin: list[Counter[str]] = [Counter() for _ in range(n_bins)]
    active_nodes_by_bin: list[set[str]] = [set() for _ in range(n_bins)]
    pair_sets: list[set[tuple[str, str]]] = [set() for _ in range(n_bins)]
    node_neighbors_by_bin: list[dict[str, set[str]]] = [
        {node: set() for node in selected_nodes}
        for _ in range(n_bins)
    ]
    group_internal_pairs_by_bin: list[dict[str, set[tuple[str, str]]]] = [
        {group: set() for group in selected_groups}
        for _ in range(n_bins)
    ]
    group_external_pairs_by_bin: list[dict[str, set[tuple[str, str]]]] = [
        {group: set() for group in selected_groups}
        for _ in range(n_bins)
    ]

    for edge in included_edges:
        bin_index = _bin_index(edge.timestamp, start=start, bin_seconds=bin_seconds)
        if bin_index < 0 or bin_index >= n_bins:
            continue
        source_group = labels.get(edge.source, "unknown")
        target_group = labels.get(edge.target, "unknown")

        if include_global:
            raw["global_total_edges"][bin_index] += 1.0
            if labels and source_group != target_group:
                raw["global_cross_group_edges"][bin_index] += 1.0
        active_nodes_by_bin[bin_index].update((edge.source, edge.target))
        sources_by_bin[bin_index][edge.source] += 1
        targets_by_bin[bin_index][edge.target] += 1
        pair_key = (
            (edge.source, edge.target)
            if directed
            else tuple(sorted((edge.source, edge.target)))
        )
        pair_sets[bin_index].add(pair_key)
        if labels:
            groups_by_bin[bin_index][source_group] += 1
            groups_by_bin[bin_index][target_group] += 1
        if include_relation and include_nodes:
            if edge.source in selected_node_set:
                node_neighbors_by_bin[bin_index][edge.source].add(edge.target)
            if edge.target in selected_node_set:
                node_neighbors_by_bin[bin_index][edge.target].add(edge.source)
        if include_relation and include_groups and labels:
            if source_group == target_group:
                if source_group in selected_group_set:
                    group_internal_pairs_by_bin[bin_index][source_group].add(pair_key)
            else:
                if source_group in selected_group_set:
                    group_external_pairs_by_bin[bin_index][source_group].add(pair_key)
                if target_group in selected_group_set:
                    group_external_pairs_by_bin[bin_index][target_group].add(pair_key)

        if include_nodes:
            if edge.source in selected_node_set:
                prefix = f"node_{_clean_name(edge.source)}"
                raw[f"{prefix}_total"][bin_index] += 1.0
                if directed:
                    raw[f"{prefix}_out"][bin_index] += 1.0
            if edge.target in selected_node_set:
                prefix = f"node_{_clean_name(edge.target)}"
                raw[f"{prefix}_total"][bin_index] += 1.0
                if directed:
                    raw[f"{prefix}_in"][bin_index] += 1.0

        if include_groups and labels:
            for group, node in ((source_group, edge.source), (target_group, edge.target)):
                if group not in selected_group_set:
                    continue
                prefix = f"group_{_clean_name(group)}"
                raw[f"{prefix}_activity"][bin_index] += 1.0
                if source_group == target_group:
                    if node == edge.source:
                        raw[f"{prefix}_internal_edges"][bin_index] += 1.0
                else:
                    raw[f"{prefix}_external_edges"][bin_index] += 1.0
            if directed:
                if source_group in selected_group_set:
                    raw[f"group_{_clean_name(source_group)}_sent"][bin_index] += 1.0
                if target_group in selected_group_set:
                    raw[f"group_{_clean_name(target_group)}_received"][bin_index] += 1.0

    if include_global:
        for bin_index in range(n_bins):
            total = raw["global_total_edges"][bin_index]
            raw["global_unique_sources"][bin_index] = len(sources_by_bin[bin_index])
            raw["global_unique_targets"][bin_index] = len(targets_by_bin[bin_index])
            raw["global_active_nodes"][bin_index] = len(active_nodes_by_bin[bin_index])
            raw["global_source_concentration"][bin_index] = _concentration(sources_by_bin[bin_index])
            raw["global_target_concentration"][bin_index] = _concentration(targets_by_bin[bin_index])
            pairs = pair_sets[bin_index]
            reciprocal = sum(1 for left, right in pairs if (right, left) in pairs)
            raw["global_reciprocity"][bin_index] = reciprocal / len(pairs) if pairs else 0.0
            if labels:
                raw["global_cross_group_ratio"][bin_index] = (
                    raw["global_cross_group_edges"][bin_index] / total if total > 0 else 0.0
                )
                raw["global_group_entropy"][bin_index] = _entropy(groups_by_bin[bin_index])

    if include_relation:
        seen_pairs: set[tuple[str, str]] = set()
        previous_pairs: set[tuple[str, str]] = set()
        previous_nodes: set[str] = set()
        seen_neighbors: dict[str, set[str]] = {node: set() for node in selected_nodes}
        previous_neighbors: dict[str, set[str]] = {node: set() for node in selected_nodes}
        seen_group_internal: dict[str, set[tuple[str, str]]] = {
            group: set() for group in selected_groups
        }
        seen_group_external: dict[str, set[tuple[str, str]]] = {
            group: set() for group in selected_groups
        }
        previous_group_external: dict[str, set[tuple[str, str]]] = {
            group: set() for group in selected_groups
        }

        for bin_index in range(n_bins):
            pairs = pair_sets[bin_index]
            nodes = active_nodes_by_bin[bin_index]
            new_pairs = pairs - seen_pairs
            returning_pairs = pairs & seen_pairs
            persistent_pairs = pairs & previous_pairs
            lost_pairs = previous_pairs - pairs
            pair_union = pairs | previous_pairs
            node_union = nodes | previous_nodes
            if include_global:
                raw["global_unique_pairs"][bin_index] = len(pairs)
                raw["global_new_pairs"][bin_index] = len(new_pairs)
                raw["global_returning_pairs"][bin_index] = len(returning_pairs)
                raw["global_persistent_pairs"][bin_index] = len(persistent_pairs)
                raw["global_lost_pairs"][bin_index] = len(lost_pairs)
                raw["global_pair_churn"][bin_index] = (
                    (len(new_pairs) + len(lost_pairs)) / len(pair_union)
                    if pair_union
                    else 0.0
                )
                raw["global_pair_jaccard_to_previous"][bin_index] = (
                    len(persistent_pairs) / len(pair_union) if pair_union else 0.0
                )
                raw["global_new_pair_ratio"][bin_index] = (
                    len(new_pairs) / len(pairs) if pairs else 0.0
                )
                raw["global_returning_pair_ratio"][bin_index] = (
                    len(returning_pairs) / len(pairs) if pairs else 0.0
                )
                raw["global_active_node_churn"][bin_index] = (
                    len(nodes ^ previous_nodes) / len(node_union) if node_union else 0.0
                )

            if include_nodes:
                for node in selected_nodes:
                    neighbors = node_neighbors_by_bin[bin_index][node]
                    new_neighbors = neighbors - seen_neighbors[node]
                    lost_neighbors = previous_neighbors[node] - neighbors
                    neighbor_union = neighbors | previous_neighbors[node]
                    prefix = f"node_{_clean_name(node)}"
                    raw[f"{prefix}_neighbor_count"][bin_index] = len(neighbors)
                    raw[f"{prefix}_new_neighbors"][bin_index] = len(new_neighbors)
                    raw[f"{prefix}_lost_neighbors"][bin_index] = len(lost_neighbors)
                    raw[f"{prefix}_neighbor_churn"][bin_index] = (
                        (len(new_neighbors) + len(lost_neighbors)) / len(neighbor_union)
                        if neighbor_union
                        else 0.0
                    )
                    raw[f"{prefix}_neighbor_jaccard_to_previous"][bin_index] = (
                        len(neighbors & previous_neighbors[node]) / len(neighbor_union)
                        if neighbor_union
                        else 0.0
                    )
                    seen_neighbors[node].update(neighbors)
                    previous_neighbors[node] = set(neighbors)

            if include_groups and labels:
                for group in selected_groups:
                    internal = group_internal_pairs_by_bin[bin_index][group]
                    external = group_external_pairs_by_bin[bin_index][group]
                    new_internal = internal - seen_group_internal[group]
                    new_external = external - seen_group_external[group]
                    lost_external = previous_group_external[group] - external
                    external_union = external | previous_group_external[group]
                    prefix = f"group_{_clean_name(group)}"
                    raw[f"{prefix}_unique_internal_pairs"][bin_index] = len(internal)
                    raw[f"{prefix}_unique_external_pairs"][bin_index] = len(external)
                    raw[f"{prefix}_new_internal_pairs"][bin_index] = len(new_internal)
                    raw[f"{prefix}_new_external_pairs"][bin_index] = len(new_external)
                    raw[f"{prefix}_external_pair_churn"][bin_index] = (
                        (len(new_external) + len(lost_external)) / len(external_union)
                        if external_union
                        else 0.0
                    )
                    seen_group_internal[group].update(internal)
                    seen_group_external[group].update(external)
                    previous_group_external[group] = set(external)

            seen_pairs.update(pairs)
            previous_pairs = set(pairs)
            previous_nodes = set(nodes)

    transformed: dict[str, NDArray[np.float64]] = {}
    dropped: list[str] = []
    for name, values in raw.items():
        adjusted = _transform(values, transform)
        if float(np.std(adjusted)) < 1e-10:
            dropped.append(name)
            continue
        transformed[name] = adjusted

    metadata = {
        "edge_count": len(edges),
        "included_edge_count": len(included_edges),
        "bin_seconds": bin_seconds,
        "bin_count": n_bins,
        "start_timestamp": start,
        "end_timestamp": start + n_bins * bin_seconds,
        "label_count": len(labels),
        "selected_nodes": selected_nodes,
        "selected_groups": selected_groups,
        "raw_series_count": len(raw),
        "series_count": len(transformed),
        "dropped_series_count": len(dropped),
        "directed": directed,
        "transform": transform,
    }
    if len(transformed) < 2:
        raise ValueError("fewer than two non-constant series survived feature building")
    return NetworkSeries(
        series=transformed,
        raw_series=raw,
        dropped_series=dropped,
        metadata=metadata,
    )


def slice_series(
    series: Mapping[str, NDArray[np.float64]],
    *,
    start: int,
    end: int,
) -> dict[str, NDArray[np.float64]]:
    return {
        name: np.asarray(values[start:end], dtype=np.float64)
        for name, values in series.items()
    }


def relation_only_network(network: NetworkSeries) -> NetworkSeries:
    """Keep only relational-change series from a built feature surface."""

    relation_markers = (
        "new_",
        "new_pair",
        "returning",
        "lost",
        "churn",
        "jaccard",
        "neighbor",
        "unique_pairs",
        "unique_internal_pairs",
        "unique_external_pairs",
        "persistent",
    )
    series = {
        name: values
        for name, values in network.series.items()
        if any(marker in name for marker in relation_markers)
    }
    raw = {
        name: network.raw_series[name]
        for name in series
        if name in network.raw_series
    }
    if len(series) < 2:
        raise ValueError("relation-only filtering left fewer than two series")
    metadata = {
        **network.metadata,
        "relation_only": True,
        "series_count": len(series),
        "raw_series_count": len(raw),
        "dropped_series_count": int(network.metadata.get("raw_series_count", 0)) - len(raw),
    }
    return NetworkSeries(
        series=series,
        raw_series=raw,
        dropped_series=[
            name
            for name in network.raw_series
            if name not in raw
        ],
        metadata=metadata,
    )


def build_configs(args: argparse.Namespace) -> list[AutoTuneConfig]:
    configs: list[AutoTuneConfig] = []
    for baseline in _parse_int_list(args.baselines):
        for adaptive in _parse_int_list(args.adaptive_windows):
            for stride in _parse_int_list(args.strides):
                for threshold in _parse_float_list(args.thresholds):
                    for decay in _parse_float_list(args.decays):
                        for min_active in _parse_int_list(args.min_active_series):
                            config = AutoTuneConfig(
                                baseline_size=baseline,
                                adaptive_window=adaptive,
                                stride=stride,
                                emission_threshold=threshold,
                                decay=decay,
                                min_active_series=min_active,
                            )
                            try:
                                config.validate()
                            except ValueError:
                                continue
                            configs.append(config)
    if not configs:
        raise ValueError("no valid org-network autotune configs")
    return configs


def observation_for_config(
    series: Mapping[str, NDArray[np.float64]],
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
    sample_rate: float,
    segment_name: str,
) -> AutoTuneObservation:
    key = (
        segment_name,
        config.baseline_size,
        config.adaptive_window,
        config.stride,
        config.score,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    observation = build_autotune_observation(series, config, sample_rate=sample_rate)
    cache.put(key, observation)
    return observation


def _score_to_dict(score: AutoTuneScore) -> dict[str, Any]:
    return score.to_dict()


def combine_null_scores(
    config: AutoTuneConfig,
    scores: Sequence[AutoTuneScore],
    null_modes: Sequence[str],
    effective_block_sizes: Sequence[int],
) -> dict[str, Any]:
    if not scores:
        return {"candidate": config.to_dict(), "accepted": False, "quality": float("-inf")}

    worst = min(
        scores,
        key=lambda score: (
            score.accepted,
            score.quality,
            score.null_summary.observed_minus_null,
        ),
    )
    return {
        "candidate": config.to_dict(),
        "accepted": all(score.accepted for score in scores),
        "quality": min(float(score.quality) for score in scores),
        "readiness_score": min(float(score.readiness_score) for score in scores),
        "null_lift_score": min(float(score.null_lift_score) for score in scores),
        "stability_score": min(float(score.stability_score) for score in scores),
        "compression_score": min(float(score.compression_score) for score in scores),
        "residual_activity_score": min(float(score.residual_activity_score) for score in scores),
        "saturation_penalty": max(float(score.saturation_penalty) for score in scores),
        "fragility_penalty": max(float(score.fragility_penalty) for score in scores),
        "observed_total": worst.null_summary.observed_total,
        "observed_active_windows": worst.null_summary.observed_active_windows,
        "null_mean_total": worst.null_summary.null_mean,
        "null_std_total": worst.null_summary.null_std,
        "observed_minus_null_total": min(
            float(score.null_summary.observed_minus_null) for score in scores
        ),
        "z_effect": min(
            float(score.null_summary.z_effect)
            if score.null_summary.z_effect is not None
            else float("-inf")
            for score in scores
        ),
        "null_total_repeats": worst.null_summary.null_repeats,
        "null_total_exceedances": worst.null_summary.null_exceedances,
        "null_total_below_or_equal": worst.null_summary.null_below_or_equal,
        "null_total_empirical_p_ge_observed": worst.null_summary.empirical_p_ge_observed,
        "null_total_empirical_p_le_observed": worst.null_summary.empirical_p_le_observed,
        "null_total_empirical_p_two_sided": worst.null_summary.empirical_p_two_sided,
        "null_total_empirical_p_floor": worst.null_summary.empirical_p_floor,
        "null_total_unique_repeats": worst.null_summary.unique_null_totals,
        "coherence_direction": (
            "surplus"
            if worst.null_summary.observed_minus_null > 0
            else "deficit"
            if worst.null_summary.observed_minus_null < 0
            else "none"
        ),
        "worst_null_mode": null_modes[scores.index(worst)],
        "null_modes": list(null_modes),
        "null_mode_effective_block_sizes": {
            mode: size
            for mode, size in zip(null_modes, effective_block_sizes, strict=True)
        },
        "null_mode_scores": [
            {"null_mode": mode, "score": _score_to_dict(score)}
            for mode, score in zip(null_modes, scores, strict=True)
        ],
    }


def effective_null_block_size(requested: int, anchors: int) -> int:
    """Bound block size so block permutation has at least two blocks when possible."""

    return min(requested, max(1, anchors // 2))


def score_candidate(
    series: Mapping[str, NDArray[np.float64]],
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
    args: argparse.Namespace,
    segment_name: str,
    seed_offset: int,
) -> dict[str, Any]:
    observation = observation_for_config(
        series,
        config,
        cache=cache,
        sample_rate=args.sample_rate,
        segment_name=segment_name,
    )
    null_modes = _parse_str_list(args.null_modes)
    scores: list[AutoTuneScore] = []
    effective_block_sizes: list[int] = []
    for mode_index, null_mode in enumerate(null_modes):
        effective_block_size = args.null_block_size
        if null_mode == "block-permute":
            effective_block_size = effective_null_block_size(
                args.null_block_size,
                observation.matrix.shape[0],
            )
        effective_block_sizes.append(effective_block_size)
        null_totals = observation_null_totals_for_config(
            observation,
            config,
            null_repeats=args.null_repeats,
            seed=args.seed + seed_offset + 100000 * mode_index,
            null_mode=null_mode,  # type: ignore[arg-type]
            null_block_size=effective_block_size,
        )
        scores.append(
            score_autotune_observation(
                observation,
                config,
                null_totals=null_totals,
            )
        )
    return combine_null_scores(config, scores, null_modes, effective_block_sizes)


def score_configs(
    series: Mapping[str, NDArray[np.float64]],
    configs: Sequence[AutoTuneConfig],
    *,
    args: argparse.Namespace,
    segment_name: str,
    cache: ObservationCache | None = None,
) -> dict[str, Any]:
    started = time.time()
    cache = cache or ObservationCache()
    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    last_progress = 0.0
    for index, config in enumerate(configs):
        try:
            reports.append(
                score_candidate(
                    series,
                    config,
                    cache=cache,
                    args=args,
                    segment_name=segment_name,
                    seed_offset=index,
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve invalid candidates with reason.
            skipped.append({"candidate": config.to_dict(), "reason": str(exc)})
        if args.progress and args.progress_every > 0:
            now = time.time()
            if now - last_progress >= args.progress_every:
                done = index + 1
                elapsed = max(now - started, 1e-9)
                eta = (len(configs) - done) / (done / elapsed) if done > 0 else 0.0
                print(
                    (
                        "org_network_autotune progress segment=%s configs=%d/%d "
                        "skipped=%d elapsed=%.1fs eta=%.1fs cache_hits=%d cache_misses=%d"
                    )
                    % (
                        segment_name,
                        done,
                        len(configs),
                        len(skipped),
                        elapsed,
                        eta,
                        cache.hits,
                        cache.misses,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                last_progress = now
    reports.sort(key=lambda row: float(row.get("quality", float("-inf"))), reverse=True)
    return {
        "best": reports[0] if reports else None,
        "scores": reports,
        "skipped": skipped,
        "cache": cache.to_dict(),
        "elapsed_seconds": time.time() - started,
    }


def structure_readiness_report(
    series: Mapping[str, NDArray[np.float64]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        return structure_readiness(
            series,
            null_repeats=args.structure_null_repeats,
            seed=args.seed + 707,
            active_z_threshold=args.structure_active_z,
            min_series=args.structure_min_series,
            min_score=args.structure_min_score,
        ).to_dict()
    except Exception as exc:  # noqa: BLE001 - report diagnostic failure without killing autotune.
        return {"ready": False, "reason": "structure_readiness_error", "error": str(exc)}


def _emissions(matrix: NDArray[np.float64], config: AutoTuneConfig) -> NDArray[np.float64]:
    excess = np.maximum(matrix - config.emission_threshold, 0.0)
    active = np.sum(excess > 0.0, axis=1)
    emissions = np.sum(excess, axis=1)
    emissions[active < config.min_active_series] = 0.0
    return emissions.astype(np.float64)


def canonical_context_for_interval(
    edges: Sequence[TemporalEdge],
    labels: Mapping[str, str],
    *,
    absolute_start_timestamp: float,
    bin_seconds: float,
    start_bin: int,
    end_bin: int,
    directed: bool,
    max_events: int,
    max_entities: int,
    max_pairs: int,
    canonical_edges: Sequence[TemporalEdge] | None = None,
    canonical_edge_indices: Sequence[int] | None = None,
    coordinate_kind: str = "canonical_time",
) -> dict[str, Any]:
    """Map a transformed bin interval back to bounded canonical edge context."""

    entity_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    group_counts: Counter[str] = Counter()
    sampled_events: list[dict[str, Any]] = []
    event_count = 0
    cross_group_events = 0
    if canonical_edge_indices is not None and len(canonical_edge_indices) != len(edges):
        raise ValueError("canonical edge index mapping must align with analysis edges")
    for edge_index, analysis_edge in enumerate(edges):
        bin_index = _bin_index(
            analysis_edge.timestamp,
            start=absolute_start_timestamp,
            bin_seconds=bin_seconds,
        )
        if bin_index < start_bin or bin_index >= end_bin:
            continue
        canonical_edge_index = (
            int(canonical_edge_indices[edge_index])
            if canonical_edge_indices is not None
            else edge_index
        )
        edge = (
            canonical_edges[canonical_edge_index]
            if canonical_edges is not None
            else analysis_edge
        )
        event_count += 1
        entity_counts.update((edge.source, edge.target))
        pair = (
            (edge.source, edge.target)
            if directed
            else tuple(sorted((edge.source, edge.target)))
        )
        pair_counts[pair] += 1
        source_group = labels.get(edge.source)
        target_group = labels.get(edge.target)
        if source_group is not None:
            group_counts[source_group] += 1
        if target_group is not None:
            group_counts[target_group] += 1
        if source_group is not None and target_group is not None and source_group != target_group:
            cross_group_events += 1
        if len(sampled_events) < max_events:
            sampled_events.append(
                {
                    "analysis_edge_index": edge_index,
                    "canonical_edge_index": canonical_edge_index,
                    "source": edge.source,
                    "target": edge.target,
                    "timestamp": float(edge.timestamp),
                    "source_group": source_group,
                    "target_group": target_group,
                }
            )

    coordinate_start = absolute_start_timestamp + start_bin * bin_seconds
    coordinate_end = absolute_start_timestamp + end_bin * bin_seconds
    return {
        "start_bin": start_bin,
        "end_bin": end_bin,
        "coordinate_kind": coordinate_kind,
        "analysis_coordinate_start": coordinate_start,
        "analysis_coordinate_end": coordinate_end,
        "timestamp_start": coordinate_start if coordinate_kind == "canonical_time" else None,
        "timestamp_end": coordinate_end if coordinate_kind == "canonical_time" else None,
        "event_count": event_count,
        "sampled_events": sampled_events,
        "sampled_events_truncated": event_count > len(sampled_events),
        "unique_entity_count": len(entity_counts),
        "top_entities": [
            {
                "entity": entity,
                "event_endpoint_count": count,
                "group": labels.get(entity),
            }
            for entity, count in entity_counts.most_common(max_entities)
        ],
        "unique_pair_count": len(pair_counts),
        "top_pairs": [
            {
                "source": pair[0],
                "target": pair[1],
                "event_count": count,
            }
            for pair, count in pair_counts.most_common(max_pairs)
        ],
        "top_groups": [
            {"group": group, "event_endpoint_count": count}
            for group, count in group_counts.most_common(max_entities)
        ],
        "cross_group_event_count": cross_group_events,
        "cross_group_event_fraction": (
            cross_group_events / event_count if event_count > 0 else 0.0
        ),
    }


def top_windows(
    series: Mapping[str, NDArray[np.float64]],
    raw_series: Mapping[str, NDArray[np.float64]],
    config: AutoTuneConfig,
    *,
    args: argparse.Namespace,
    segment_name: str,
    segment_bin_offset: int,
    absolute_start_timestamp: float,
    canonical_edges: Sequence[TemporalEdge] | None = None,
    analysis_edges: Sequence[TemporalEdge] | None = None,
    canonical_edge_indices: Sequence[int] | None = None,
    labels: Mapping[str, str] | None = None,
    directed: bool = True,
    observation: AutoTuneObservation | None = None,
) -> list[dict[str, Any]]:
    if observation is None:
        observation = build_autotune_observation(series, config, sample_rate=args.sample_rate)
    emissions = _emissions(observation.matrix, config)
    ordered = np.argsort(emissions)[::-1]
    names = list(series.keys())
    rows: list[dict[str, Any]] = []
    for row_index in ordered:
        emission = float(emissions[int(row_index)])
        if emission <= 0.0:
            continue
        anchor = int(observation.anchors[int(row_index)])
        absolute_bin = segment_bin_offset + anchor
        scores = observation.matrix[int(row_index)]
        active = [
            {
                "series": name,
                "score": float(score),
                "raw_value": float(raw_series[name][absolute_bin])
                if name in raw_series and absolute_bin < len(raw_series[name])
                else None,
            }
            for name, score in zip(names, scores, strict=True)
            if score > config.emission_threshold
        ]
        rows.append(
            {
                "segment": segment_name,
                "anchor": anchor,
                "absolute_bin": absolute_bin,
                "coordinate_kind": (
                    "canonical_time"
                    if getattr(args, "event_order", "real-time") == "real-time"
                    else "event_rank"
                ),
                "analysis_coordinate_start": absolute_start_timestamp
                + absolute_bin * args.bin_seconds,
                "analysis_coordinate_end": absolute_start_timestamp
                + (absolute_bin + 1) * args.bin_seconds,
                "timestamp_start": (
                    absolute_start_timestamp + absolute_bin * args.bin_seconds
                    if getattr(args, "event_order", "real-time") == "real-time"
                    else None
                ),
                "timestamp_end": (
                    absolute_start_timestamp + (absolute_bin + 1) * args.bin_seconds
                    if getattr(args, "event_order", "real-time") == "real-time"
                    else None
                ),
                "emission": emission,
                "active_series_count": len(active),
                "max_score": float(np.max(scores)) if len(scores) else 0.0,
                "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
                "active_series": active,
                "analysis_bins": {
                    "frozen_baseline_start": segment_bin_offset,
                    "frozen_baseline_end": segment_bin_offset + config.baseline_size,
                    "adaptive_start": max(segment_bin_offset, absolute_bin - config.adaptive_window),
                    "adaptive_end": absolute_bin,
                    "anchor": absolute_bin,
                },
            }
        )
        if canonical_edges is not None:
            context_labels = labels or {}
            context_edges = analysis_edges or canonical_edges
            rows[-1]["canonical_context"] = {
                "anchor_bin": canonical_context_for_interval(
                    context_edges,
                    context_labels,
                    absolute_start_timestamp=absolute_start_timestamp,
                    bin_seconds=args.bin_seconds,
                    start_bin=absolute_bin,
                    end_bin=absolute_bin + 1,
                    directed=directed,
                    max_events=args.canonical_max_events,
                    max_entities=args.canonical_max_entities,
                    max_pairs=args.canonical_max_pairs,
                    canonical_edges=canonical_edges if analysis_edges is not None else None,
                    canonical_edge_indices=canonical_edge_indices,
                    coordinate_kind=rows[-1]["coordinate_kind"],
                ),
                "adaptive_history": canonical_context_for_interval(
                    context_edges,
                    context_labels,
                    absolute_start_timestamp=absolute_start_timestamp,
                    bin_seconds=args.bin_seconds,
                    start_bin=max(segment_bin_offset, absolute_bin - config.adaptive_window),
                    end_bin=absolute_bin,
                    directed=directed,
                    max_events=args.canonical_max_events,
                    max_entities=args.canonical_max_entities,
                    max_pairs=args.canonical_max_pairs,
                    canonical_edges=canonical_edges if analysis_edges is not None else None,
                    canonical_edge_indices=canonical_edge_indices,
                    coordinate_kind=rows[-1]["coordinate_kind"],
                ),
            }
        if len(rows) >= args.top_windows:
            break
    return rows


def _config_from_report(report: Mapping[str, Any]) -> AutoTuneConfig:
    payload = report["candidate"]
    return AutoTuneConfig(
        baseline_size=int(payload["baseline_size"]),
        adaptive_window=int(payload["adaptive_window"]),
        stride=int(payload["stride"]),
        score=payload.get("score", "max"),
        emission_threshold=float(payload["emission_threshold"]),
        decay=float(payload["decay"]),
        min_active_series=int(payload["min_active_series"]),
    )


def build_selected_network(
    edges: Sequence[TemporalEdge],
    labels: Mapping[str, str],
    *,
    args: argparse.Namespace,
) -> NetworkSeries:
    """Build the requested structural feature family for downstream analysis."""

    mode = "relation" if args.relation_only else args.surface_mode
    if mode == "group" and not labels:
        raise ValueError("group surface requires canonical group labels")
    if mode == "full":
        include_global = not args.no_global_series
        include_nodes = not args.no_node_series
        include_groups = not args.no_group_series
        include_relation = not args.no_relation_series
    elif mode == "relation":
        include_global = not args.no_global_series
        include_nodes = not args.no_node_series
        include_groups = not args.no_group_series
        include_relation = True
    elif mode == "graph":
        include_global = not args.no_global_series
        include_nodes = not args.no_node_series
        include_groups = False
        include_relation = False
    elif mode == "group":
        include_global = not args.no_global_series
        include_nodes = False
        include_groups = True
        include_relation = True
    else:
        raise ValueError(f"unknown surface mode: {mode}")

    network = build_network_series(
        edges,
        labels,
        bin_seconds=args.bin_seconds,
        top_nodes=args.top_nodes,
        top_groups=args.top_groups,
        include_global=include_global,
        include_nodes=include_nodes,
        include_groups=include_groups,
        include_relation=include_relation,
        directed=args.directed,
        transform=args.transform,
        max_bins=args.max_bins,
    )
    if mode == "relation":
        network = relation_only_network(network)
    return NetworkSeries(
        series=network.series,
        raw_series=network.raw_series,
        dropped_series=network.dropped_series,
        metadata={
            **network.metadata,
            "surface_mode": mode,
            "event_order": args.event_order,
        },
    )


def run_org_network_autotune(args: argparse.Namespace) -> dict[str, Any]:
    args = argparse.Namespace(**vars(args))
    runtime_defaults = {
        "surface_mode": "full",
        "event_order": "real-time",
        "event_bin_size": 1024,
        "canonical_max_events": 12,
        "canonical_max_entities": 12,
        "canonical_max_pairs": 12,
    }
    for name, value in runtime_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    edge_path, label_path, dataset_metadata = resolve_dataset_files(args)
    data_format = str(dataset_metadata.get("data_format", args.data_format))
    if data_format == "edges":
        edges = read_edges(
            edge_path,
            source_col=args.source_col,
            target_col=args.target_col,
            time_col=args.time_col,
            max_edges=args.max_edges,
        )
    elif data_format == "simplices":
        edges = read_simplicial_edges(
            edge_path,
            prefix=args.simplex_prefix,
            max_edges=args.max_edges,
        )
    else:
        raise ValueError(f"unknown data format: {data_format}")
    labels = read_labels(label_path)
    canonical_edges = edges
    analysis_edges = edges
    canonical_edge_indices: list[int] | None = None
    requested_bin_seconds = float(args.bin_seconds)
    if args.event_order != "real-time":
        analysis_edges, canonical_edge_indices = remap_edges_by_order_with_mapping(
            canonical_edges,
            order=args.event_order,
            labels=labels,
            directed=args.directed,
            seed=args.seed,
        )
        args.bin_seconds = float(args.event_bin_size)
    network = build_selected_network(analysis_edges, labels, args=args)
    projection = {
        "surface_mode": network.metadata["surface_mode"],
        "event_order": args.event_order,
        "requested_bin_seconds": requested_bin_seconds,
        "analysis_bin_seconds": float(args.bin_seconds),
        "coordinate_kind": "canonical_time" if args.event_order == "real-time" else "event_rank",
        "canonical_mapping_available": canonical_edge_indices is not None,
    }
    n_bins = int(network.metadata["bin_count"])
    split = int(n_bins * args.validation_start_fraction)
    if args.no_validation:
        split = n_bins
    calibration_series = slice_series(network.series, start=0, end=split)
    readiness: dict[str, Any] = {
        "calibration": structure_readiness_report(calibration_series, args=args),
    }
    validation_series: dict[str, NDArray[np.float64]] | None = None
    if not args.no_validation:
        validation_series = slice_series(network.series, start=split, end=n_bins)
        readiness["validation"] = structure_readiness_report(validation_series, args=args)
    if args.readiness_only:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in {"output"}
            },
            "dataset": {
                **dataset_metadata,
                "edge_path": str(edge_path),
                "label_path": str(label_path) if label_path else None,
                "data_format": data_format,
            },
            "analysis_projection": projection,
            "series_metadata": network.metadata,
            "dropped_series": network.dropped_series[:50],
            "calibration_bins": split,
            "validation_bins": 0 if args.no_validation else n_bins - split,
            "structure_readiness": readiness,
            "configs_searched": 0,
            "calibration": None,
            "validation": None,
            "top_windows": {
                "calibration": [],
                "validation": [],
            },
        }
    min_needed = max(_parse_int_list(args.baselines)) + max(_parse_int_list(args.strides)) + 1
    if split < min_needed:
        raise ValueError(
            f"calibration bins {split} are too short for requested configs; need at least {min_needed}"
        )
    if not args.no_validation and n_bins - split < min_needed:
        raise ValueError(
            f"validation bins {n_bins - split} are too short for requested configs; need at least {min_needed}"
        )
    configs = build_configs(args)
    calibration_cache = ObservationCache()
    calibration = score_configs(
        calibration_series,
        configs,
        args=args,
        segment_name="calibration",
        cache=calibration_cache,
    )
    selected = calibration.get("best")
    validation: dict[str, Any] | None = None
    calibration_top: list[dict[str, Any]] = []
    validation_top: list[dict[str, Any]] = []
    if selected:
        selected_config = _config_from_report(selected)
        calibration_observation = observation_for_config(
            calibration_series,
            selected_config,
            cache=calibration_cache,
            sample_rate=args.sample_rate,
            segment_name="calibration",
        )
        calibration_top = top_windows(
            calibration_series,
            network.raw_series,
            selected_config,
            args=args,
            segment_name="calibration",
            segment_bin_offset=0,
            absolute_start_timestamp=float(network.metadata["start_timestamp"]),
            canonical_edges=edges,
            analysis_edges=analysis_edges if canonical_edge_indices is not None else None,
            canonical_edge_indices=canonical_edge_indices,
            labels=labels,
            directed=args.directed,
            observation=calibration_observation,
        )
        if not args.no_validation:
            assert validation_series is not None
            validation_cache = ObservationCache()
            validation_scored = score_candidate(
                validation_series,
                selected_config,
                cache=validation_cache,
                args=args,
                segment_name="validation",
                seed_offset=999_000,
            )
            validation = validation_scored
            validation_observation = observation_for_config(
                validation_series,
                selected_config,
                cache=validation_cache,
                sample_rate=args.sample_rate,
                segment_name="validation",
            )
            validation_top = top_windows(
                validation_series,
                network.raw_series,
                selected_config,
                args=args,
                segment_name="validation",
                segment_bin_offset=split,
                absolute_start_timestamp=float(network.metadata["start_timestamp"]),
                canonical_edges=edges,
                analysis_edges=analysis_edges if canonical_edge_indices is not None else None,
                canonical_edge_indices=canonical_edge_indices,
                labels=labels,
                directed=args.directed,
                observation=validation_observation,
            )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"output"}
        },
        "dataset": {
            **dataset_metadata,
            "edge_path": str(edge_path),
            "label_path": str(label_path) if label_path else None,
            "data_format": data_format,
        },
        "analysis_projection": projection,
        "series_metadata": network.metadata,
        "dropped_series": network.dropped_series[:50],
        "calibration_bins": split,
        "validation_bins": 0 if args.no_validation else n_bins - split,
        "structure_readiness": readiness,
        "configs_searched": len(configs),
        "calibration": calibration,
        "validation": validation,
        "top_windows": {
            "calibration": calibration_top,
            "validation": validation_top,
        },
    }


def print_report(report: Mapping[str, Any], *, top: int) -> None:
    dataset = report.get("dataset", {})
    series = report.get("series_metadata", {})
    print("Organizational network autotune")
    print("  dataset=%s" % dataset.get("dataset", dataset.get("edge_path", "custom")))
    projection = dict(report.get("analysis_projection") or {})
    if projection:
        print(
            "  projection surface=%s order=%s analysis_bin_seconds=%s canonical_mapping=%s"
            % (
                projection.get("surface_mode"),
                projection.get("event_order"),
                projection.get("analysis_bin_seconds"),
                projection.get("canonical_mapping_available"),
            )
        )
    print(
        "  bins=%s calibration=%s validation=%s series=%s dropped=%s"
        % (
            series.get("bin_count"),
            report.get("calibration_bins"),
            report.get("validation_bins"),
            series.get("series_count"),
            series.get("dropped_series_count"),
        )
    )
    readiness = report.get("structure_readiness", {})
    if readiness:
        calibration_ready = dict(readiness.get("calibration") or {})
        if calibration_ready:
            print(
                "  structure calibration ready=%s score=%.3f reason=%s cov_z=%s pattern_z=%s temporal_z=%s"
                % (
                    calibration_ready.get("ready"),
                    float(calibration_ready.get("structure_score", 0.0)),
                    calibration_ready.get("reason"),
                    "None"
                    if calibration_ready.get("covariance_z_effect") is None
                    else "%.2f" % float(calibration_ready.get("covariance_z_effect", 0.0)),
                    "None"
                    if calibration_ready.get("pattern_z_effect") is None
                    else "%.2f" % float(calibration_ready.get("pattern_z_effect", 0.0)),
                    "None"
                    if calibration_ready.get("temporal_z_effect") is None
                    else "%.2f" % float(calibration_ready.get("temporal_z_effect", 0.0)),
                )
            )
        validation_ready = dict(readiness.get("validation") or {})
        if validation_ready:
            print(
                "  structure validation ready=%s score=%.3f reason=%s cov_z=%s pattern_z=%s temporal_z=%s"
                % (
                    validation_ready.get("ready"),
                    float(validation_ready.get("structure_score", 0.0)),
                    validation_ready.get("reason"),
                    "None"
                    if validation_ready.get("covariance_z_effect") is None
                    else "%.2f" % float(validation_ready.get("covariance_z_effect", 0.0)),
                    "None"
                    if validation_ready.get("pattern_z_effect") is None
                    else "%.2f" % float(validation_ready.get("pattern_z_effect", 0.0)),
                    "None"
                    if validation_ready.get("temporal_z_effect") is None
                    else "%.2f" % float(validation_ready.get("temporal_z_effect", 0.0)),
                )
            )
    calibration = dict(report.get("calibration") or {})
    best = dict(calibration.get("best") or {})
    if not best:
        print("  no calibration candidate")
        return
    cfg = best["candidate"]
    print(
        "  calibration accepted=%s quality=%.4f direction=%s z=%.2f delta=%.3f p_ge=%.4f p_le=%.4f p_two=%.4f unique=%s worst_null=%s"
        % (
            best["accepted"],
            best["quality"],
            best["coherence_direction"],
            best["z_effect"],
            best["observed_minus_null_total"],
            best["null_total_empirical_p_ge_observed"],
            best["null_total_empirical_p_le_observed"],
            best["null_total_empirical_p_two_sided"],
            best["null_total_unique_repeats"],
            best["worst_null_mode"],
        )
    )
    print(
        "  config baseline=%d adaptive=%d stride=%d threshold=%.2f decay=%.2f min_active=%d"
        % (
            cfg["baseline_size"],
            cfg["adaptive_window"],
            cfg["stride"],
            cfg["emission_threshold"],
            cfg["decay"],
            cfg["min_active_series"],
        )
    )
    validation = report.get("validation")
    if validation:
        validation = dict(validation)
        print(
            "  validation accepted=%s quality=%.4f direction=%s z=%.2f delta=%.3f p_ge=%.4f p_le=%.4f p_two=%.4f unique=%s worst_null=%s"
            % (
                validation["accepted"],
                validation["quality"],
                validation["coherence_direction"],
                validation["z_effect"],
                validation["observed_minus_null_total"],
                validation["null_total_empirical_p_ge_observed"],
                validation["null_total_empirical_p_le_observed"],
                validation["null_total_empirical_p_two_sided"],
                validation["null_total_unique_repeats"],
                validation["worst_null_mode"],
            )
        )
    scores = list(calibration.get("scores", []))[:top]
    if len(scores) > 1:
        print("  top calibration configs")
        for row in scores:
            candidate = row["candidate"]
            print(
                "    accepted=%s quality=%.4f z=%.2f threshold=%.2f min_active=%d"
                % (
                    row["accepted"],
                    row["quality"],
                    row["z_effect"],
                    candidate["emission_threshold"],
                    candidate["min_active_series"],
                )
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data/org_networks"))
    parser.add_argument("--edges", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--data-format", choices=["edges", "simplices"], default="edges")
    parser.add_argument("--simplex-prefix", default="email-Enron")
    parser.add_argument("--source-col", type=int, default=0)
    parser.add_argument("--target-col", type=int, default=1)
    parser.add_argument("--time-col", type=int, default=2)
    parser.add_argument("--directed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-edges", type=int, default=0)
    parser.add_argument("--max-bins", type=int, default=0)
    parser.add_argument("--bin-seconds", type=float, default=86400.0)
    parser.add_argument("--top-nodes", type=int, default=12)
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--no-global-series", action="store_true")
    parser.add_argument("--no-node-series", action="store_true")
    parser.add_argument("--no-group-series", action="store_true")
    parser.add_argument("--no-relation-series", action="store_true")
    parser.add_argument("--relation-only", action="store_true")
    parser.add_argument(
        "--surface-mode",
        choices=["full", "relation", "graph", "group"],
        default="full",
    )
    parser.add_argument(
        "--event-order",
        choices=[
            "real-time",
            "timestamp",
            "timestamp_bucket_partial_order",
            "first_seen_pair",
            "degree_descending",
            "group_then_time",
            "stable_id_or_alphabetic",
            "hash_order",
            "random_order",
            "silly_proxy_order",
        ],
        default="real-time",
    )
    parser.add_argument("--event-bin-size", type=int, default=1024)
    parser.add_argument("--transform", choices=["none", "log", "robust", "log-robust"], default="log-robust")
    parser.add_argument("--validation-start-fraction", type=float, default=0.5)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--null-modes", default="block-permute")
    parser.add_argument("--null-block-size", type=int, default=8)
    parser.add_argument("--structure-null-repeats", type=int, default=50)
    parser.add_argument("--structure-active-z", type=float, default=1.5)
    parser.add_argument("--structure-min-series", type=int, default=3)
    parser.add_argument("--structure-min-score", type=float, default=0.35)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--baselines", default="96,128")
    parser.add_argument("--adaptive-windows", default="32,48")
    parser.add_argument("--strides", default="8,16")
    parser.add_argument("--thresholds", default="2.5,3.0")
    parser.add_argument("--decays", default="0.75,0.9")
    parser.add_argument("--min-active-series", default="2,3")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--top-windows", type=int, default=10)
    parser.add_argument("--canonical-max-events", type=int, default=12)
    parser.add_argument("--canonical-max-entities", type=int, default=12)
    parser.add_argument("--canonical-max-pairs", type=int, default=12)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if not 0.0 < args.validation_start_fraction < 1.0:
        raise ValueError("--validation-start-fraction must be in (0, 1)")
    if min(args.canonical_max_events, args.canonical_max_entities, args.canonical_max_pairs) < 0:
        raise ValueError("canonical context limits must be non-negative")
    if args.event_bin_size <= 0:
        raise ValueError("--event-bin-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_org_network_autotune(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report, top=args.top)


if __name__ == "__main__":
    main()
