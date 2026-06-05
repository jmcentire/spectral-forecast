"""Post-hoc geographic audit of label-free CDIP relationship discoveries.

Coordinates are used only after relationship discovery and FDR correction.
They do not participate in candidate generation, relationship scoring,
residualization, calibration, null construction, or pair selection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.io import netcdf_file
from scipy.stats import hypergeom


DEFAULT_RELATIONSHIP_REPORT = Path(
    "experiments/results/"
    "2026-06-04-cdip-relationship-discovery-graph64-null4999.json"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    """Return great-circle distance in kilometers."""

    left_lat, left_lon = (math.radians(value) for value in left)
    right_lat, right_lon = (math.radians(value) for value in right)
    delta_lat = right_lat - left_lat
    delta_lon = right_lon - left_lon
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(left_lat)
        * math.cos(right_lat)
        * math.sin(delta_lon / 2.0) ** 2
    )
    return float(6371.0 * 2.0 * math.asin(math.sqrt(value)))


def _coordinate(path: Path) -> tuple[float, float]:
    with netcdf_file(path, mmap=True) as nc:
        latitude = np.asarray(
            nc.variables["metaDeployLatitude"].data,
            dtype=np.float64,
        ).ravel()
        longitude = np.asarray(
            nc.variables["metaDeployLongitude"].data,
            dtype=np.float64,
        ).ravel()
    valid = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (np.abs(latitude) <= 90.0)
        & (np.abs(longitude) <= 180.0)
    )
    if not np.any(valid):
        raise ValueError(f"{path} has no valid deployment coordinates")
    return (
        float(np.median(latitude[valid])),
        float(np.median(longitude[valid])),
    )


def audit_geography(
    pair_summaries: Sequence[Mapping[str, Any]],
    *,
    coordinates: Mapping[str, tuple[float, float]],
    names: Mapping[str, str],
    profile_key: str = "raw_profile",
    thresholds_km: Sequence[float] = (100.0, 250.0, 1000.0),
    matched_permutation_repeats: int = 100_000,
    seed: int = 20260604,
) -> dict[str, Any]:
    """Compare discovered pair distances with the complete tested pair graph."""

    all_pairs = {tuple(str(value) for value in row["entities"]) for row in pair_summaries}
    detected_layers: dict[tuple[str, str], set[str]] = defaultdict(set)
    replicated: dict[tuple[str, str], bool] = defaultdict(bool)
    pair_strata = {}
    for row in pair_summaries:
        pair = tuple(str(value) for value in row["entities"])
        pair_strata[pair] = (
            int(row.get("occurrences", 0)),
            int(row.get("independent_time_clusters", 0)),
        )
        if bool(
            row.get(profile_key, {})
            .get("domain_regroup", {})
            .get("detected_fdr", False)
        ):
            detected_layers[pair].add(str(row["layer"]))
            replicated[pair] = bool(
                replicated[pair]
                or row.get(
                    "replicated_across_distinct_times",
                    row.get("replicated_across_segments", False),
                )
            )

    distances = {
        pair: haversine_km(coordinates[pair[0]], coordinates[pair[1]])
        for pair in all_pairs
        if pair[0] in coordinates and pair[1] in coordinates
    }
    detected_pairs = sorted(pair for pair in detected_layers if pair in distances)
    strata: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    for pair in distances:
        strata[pair_strata[pair]].append(pair)
    rng = np.random.default_rng(seed)
    threshold_rows = []
    for threshold in thresholds_km:
        population_close = sum(distance <= threshold for distance in distances.values())
        detected_close = sum(distances[pair] <= threshold for pair in detected_pairs)
        matched_exceedances = 0
        for _ in range(matched_permutation_repeats):
            matched_close = 0
            for pair in detected_pairs:
                candidates = strata[pair_strata[pair]]
                candidate = candidates[int(rng.integers(len(candidates)))]
                matched_close += int(distances[candidate] <= threshold)
            matched_exceedances += int(matched_close >= detected_close)
        threshold_rows.append(
            {
                "threshold_km": float(threshold),
                "population_pairs": len(distances),
                "population_close": population_close,
                "population_close_fraction": population_close / max(len(distances), 1),
                "detected_pairs": len(detected_pairs),
                "detected_close": detected_close,
                "detected_close_fraction": detected_close / max(len(detected_pairs), 1),
                "hypergeometric_p_ge_detected_close": float(
                    hypergeom.sf(
                        detected_close - 1,
                        len(distances),
                        population_close,
                        len(detected_pairs),
                    )
                ),
                "matched_permutation_repeats": matched_permutation_repeats,
                "matched_permutation_exceedances": matched_exceedances,
                "matched_permutation_p_ge_detected_close": float(
                    (matched_exceedances + 1) / (matched_permutation_repeats + 1)
                ),
            }
        )

    pair_rows = []
    for pair in detected_pairs:
        pair_rows.append(
            {
                "entities": list(pair),
                "platform_names": [names.get(pair[0], pair[0]), names.get(pair[1], pair[1])],
                "distance_km": distances[pair],
                "layers": sorted(detected_layers[pair]),
                "replicated_across_segments": replicated[pair],
            }
        )
    pair_rows.sort(key=lambda row: row["distance_km"])
    all_distance_values = list(distances.values())
    detected_distance_values = [distances[pair] for pair in detected_pairs]
    return {
        "method": {
            "label_use": "post-hoc audit only",
            "selection_use": "none",
            "matched_geography_null": (
                "samples one tested pair per discovery while matching each discovery's "
                "occurrence count and independent-time-cluster count"
            ),
            "detected_pair_rule": (
                f"at least one calibrated layer has FDR-significant {profile_key} "
                "domain-regroup evidence"
            ),
        },
        "summary": {
            "profile_key": profile_key,
            "population_pair_identities": len(distances),
            "detected_pair_identities": len(detected_pairs),
            "population_median_distance_km": (
                float(np.median(all_distance_values)) if all_distance_values else None
            ),
            "detected_median_distance_km": (
                float(np.median(detected_distance_values))
                if detected_distance_values
                else None
            ),
        },
        "thresholds": threshold_rows,
        "detected_pairs": pair_rows,
    }


def run_audit(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pair_summaries = report["aggregate"]["spectral_specificity"]["pair_summary"]
    paths = {}
    names = {}
    for window in report["windows"]:
        for source in window["source"]["sources"]:
            entity = str(source["entity"])
            paths[entity] = Path(source["path"])
            names[entity] = str(source["platform_name"])
    coordinates = {entity: _coordinate(path) for entity, path in paths.items()}
    profile_keys = [
        profile_key
        for profile_key in (
            "raw_profile",
            "envelope_attenuation_0.25",
            "envelope_attenuation_0.50",
            "envelope_attenuation_0.75",
            "envelope_attenuation_1.00",
            "signed_envelope_residual",
        )
        if pair_summaries and profile_key in pair_summaries[0]
    ]
    profile_audits = {
        profile_key: audit_geography(
            pair_summaries,
            coordinates=coordinates,
            names=names,
            profile_key=profile_key,
        )
        for profile_key in profile_keys
    }
    return {
        "relationship_report": str(report_path),
        "audit": profile_audits["raw_profile"],
        "profile_audits": profile_audits,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relationship-report",
        type=Path,
        default=DEFAULT_RELATIONSHIP_REPORT,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_audit(args.relationship_report)
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
