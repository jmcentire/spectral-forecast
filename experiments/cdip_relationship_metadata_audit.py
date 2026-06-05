"""Post-hoc metadata audit of label-free CDIP relationship discoveries.

Metadata is used only after relationship discovery and FDR correction. The
matched audit asks whether discovered pair identities share acquisition or
processing characteristics more often than tested pairs with comparable
observation opportunity.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.io import netcdf_file

from experiments.cdip_observe import _decode_attr, _scalar


DEFAULT_RELATIONSHIP_REPORT = Path(
    "experiments/results/"
    "2026-06-04-cdip-relationship-envelope-ladder-graph128-independent-time.json"
)
PROFILE_KEYS = (
    "raw_profile",
    "envelope_attenuation_0.25",
    "envelope_attenuation_0.50",
    "envelope_attenuation_0.75",
    "envelope_attenuation_1.00",
    "signed_envelope_residual",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _processing_family(history: str) -> str:
    match = re.search(r",\s*([A-Za-z]+)\d+_", history)
    return match.group(1).lower() if match else "unknown"


def _station_metadata(path: Path) -> dict[str, Any]:
    with netcdf_file(path, mmap=True) as nc:
        history = _decode_attr(getattr(nc, "history", ""), "")
        return {
            "sample_rate": float(_scalar(nc.variables["xyzSampleRate"])),
            "processing_family": _processing_family(history),
            "water_depth": float(_scalar(nc.variables["metaWaterDepth"])),
            "deployment_id": _decode_attr(
                getattr(nc, "cdip_deployment_id", ""),
                "",
            ),
        }


def audit_metadata(
    pair_summaries: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Mapping[str, Any]],
    profile_key: str,
    matched_permutation_repeats: int = 100_000,
    seed: int = 20260604,
) -> dict[str, Any]:
    """Audit acquisition and processing metadata against matched tested pairs."""

    all_pairs = {tuple(str(value) for value in row["entities"]) for row in pair_summaries}
    pair_strata = {}
    detected_layers: dict[tuple[str, str], set[str]] = defaultdict(set)
    replicated: dict[tuple[str, str], bool] = defaultdict(bool)
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

    def pair_features(pair: tuple[str, str]) -> dict[str, Any]:
        left = metadata[pair[0]]
        right = metadata[pair[1]]
        left_depth = float(left["water_depth"])
        right_depth = float(right["water_depth"])
        return {
            "sample_rate_match": bool(
                math.isclose(
                    float(left["sample_rate"]),
                    float(right["sample_rate"]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ),
            "processing_family_match": bool(
                str(left["processing_family"]) == str(right["processing_family"])
            ),
            "sample_rate_and_processing_family_match": bool(
                math.isclose(
                    float(left["sample_rate"]),
                    float(right["sample_rate"]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                and str(left["processing_family"])
                == str(right["processing_family"])
            ),
            "water_depth_absolute_log_ratio": (
                float(abs(math.log(left_depth / right_depth)))
                if left_depth > 0.0 and right_depth > 0.0
                else None
            ),
        }

    usable_pairs = sorted(
        pair
        for pair in all_pairs
        if pair[0] in metadata and pair[1] in metadata
    )
    features = {pair: pair_features(pair) for pair in usable_pairs}
    detected_pairs = sorted(pair for pair in detected_layers if pair in features)
    strata: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    for pair in usable_pairs:
        strata[pair_strata[pair]].append(pair)

    observed = {
        key: int(sum(features[pair][key] for pair in detected_pairs))
        for key in (
            "sample_rate_match",
            "processing_family_match",
            "sample_rate_and_processing_family_match",
        )
    }
    observed_depth_values = [
        float(features[pair]["water_depth_absolute_log_ratio"])
        for pair in detected_pairs
        if features[pair]["water_depth_absolute_log_ratio"] is not None
    ]
    observed_depth_median = (
        float(np.median(observed_depth_values)) if observed_depth_values else None
    )
    exceedances = {key: 0 for key in observed}
    depth_similarity_exceedances = 0
    rng = np.random.default_rng(seed)
    for _ in range(matched_permutation_repeats):
        sampled = [
            candidates[int(rng.integers(len(candidates)))]
            for pair in detected_pairs
            for candidates in [strata[pair_strata[pair]]]
        ]
        for key in exceedances:
            exceedances[key] += int(
                sum(features[pair][key] for pair in sampled) >= observed[key]
            )
        sampled_depth = [
            float(features[pair]["water_depth_absolute_log_ratio"])
            for pair in sampled
            if features[pair]["water_depth_absolute_log_ratio"] is not None
        ]
        if observed_depth_median is not None and sampled_depth:
            depth_similarity_exceedances += int(
                float(np.median(sampled_depth)) <= observed_depth_median
            )

    metric_rows = []
    for key in observed:
        metric_rows.append(
            {
                "metric": key,
                "detected_matches": observed[key],
                "detected_pairs": len(detected_pairs),
                "matched_permutation_repeats": matched_permutation_repeats,
                "matched_permutation_exceedances": exceedances[key],
                "matched_permutation_p_ge_detected_matches": float(
                    (exceedances[key] + 1) / (matched_permutation_repeats + 1)
                ),
            }
        )
    metric_rows.append(
        {
            "metric": "water_depth_absolute_log_ratio_median",
            "detected_median": observed_depth_median,
            "detected_pairs": len(observed_depth_values),
            "matched_permutation_repeats": matched_permutation_repeats,
            "matched_permutation_exceedances": depth_similarity_exceedances,
            "matched_permutation_p_le_detected_median": float(
                (depth_similarity_exceedances + 1)
                / (matched_permutation_repeats + 1)
            ),
        }
    )

    pair_rows = []
    for pair in detected_pairs:
        left = metadata[pair[0]]
        right = metadata[pair[1]]
        pair_rows.append(
            {
                "entities": list(pair),
                "layers": sorted(detected_layers[pair]),
                "replicated_across_distinct_times": replicated[pair],
                "sample_rates": [left["sample_rate"], right["sample_rate"]],
                "processing_families": [
                    left["processing_family"],
                    right["processing_family"],
                ],
                "water_depths": [left["water_depth"], right["water_depth"]],
                **features[pair],
            }
        )
    pair_rows.sort(
        key=lambda row: (
            not row["replicated_across_distinct_times"],
            row["entities"],
        )
    )
    return {
        "method": {
            "label_use": "post-hoc audit only",
            "selection_use": "none",
            "matched_metadata_null": (
                "samples one tested pair per discovery while matching each "
                "discovery's occurrence count and independent-time-cluster count"
            ),
            "profile_key": profile_key,
        },
        "summary": {
            "profile_key": profile_key,
            "population_pair_identities": len(usable_pairs),
            "detected_pair_identities": len(detected_pairs),
            "detected_sample_rate_counts": dict(
                Counter(
                    str(metadata[entity]["sample_rate"])
                    for pair in detected_pairs
                    for entity in pair
                )
            ),
            "detected_processing_family_counts": dict(
                Counter(
                    str(metadata[entity]["processing_family"])
                    for pair in detected_pairs
                    for entity in pair
                )
            ),
        },
        "metrics": metric_rows,
        "detected_pairs": pair_rows,
    }


def run_audit(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pair_summaries = report["aggregate"]["spectral_specificity"]["pair_summary"]
    paths = {
        str(source["entity"]): Path(source["path"])
        for window in report["windows"]
        for source in window["source"]["sources"]
    }
    metadata = {entity: _station_metadata(path) for entity, path in paths.items()}
    profile_keys = [
        profile_key
        for profile_key in PROFILE_KEYS
        if pair_summaries and profile_key in pair_summaries[0]
    ]
    return {
        "relationship_report": str(report_path),
        "station_metadata": metadata,
        "profile_audits": {
            profile_key: audit_metadata(
                pair_summaries,
                metadata=metadata,
                profile_key=profile_key,
            )
            for profile_key in profile_keys
        },
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
