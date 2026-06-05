"""Remove known CDIP acquisition classes and retest residual pair profiles.

This iterative control is intentionally no longer label-blind with respect to
the acquisition layer. Sample rate and processing family were discovered
post-hoc in the prior relationship graph; this runner treats them as known,
normalizes within those classes, restricts candidates to within-class pairs,
and uses exact-occurrence matched pair controls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cdip_relationship_discovery import (
    SIGNED_ENVELOPE_PROFILE,
    _spectral_specificity,
)
from experiments.cdip_relationship_metadata_audit import _station_metadata


DEFAULT_SOURCE_REPORT = Path(
    "experiments/results/"
    "2026-06-04-cdip-relationship-envelope-ladder-graph128-independent-time.json"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def acquisition_stratum(metadata: Mapping[str, Any]) -> str:
    """Return the known acquisition/processing class used by this control."""

    return "%0.2f:%s" % (
        float(metadata["sample_rate"]),
        str(metadata["processing_family"]),
    )


def run_control(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    paths = {
        str(source_row["entity"]): Path(source_row["path"])
        for window in source["windows"]
        for source_row in window["source"]["sources"]
    }
    metadata = {entity: _station_metadata(path) for entity, path in paths.items()}
    entity_strata = {
        entity: acquisition_stratum(payload)
        for entity, payload in metadata.items()
    }
    supported_by_layer = {
        str(layer): [str(family) for family in families]
        for layer, families in source["calibration"]["supported_hypotheses"].items()
    }
    spectral_specificity = _spectral_specificity(
        source["windows"],
        supported_by_layer=supported_by_layer,
        null_repeats=args.null_repeats,
        seed=args.seed,
        replication_separation_seconds=args.replication_separation_hours * 3600.0,
        entity_strata=entity_strata,
        within_stratum_pairs_only=True,
        pair_domain_control="matched_occurrence_regroup",
        require_complete_matched_occurrences=True,
        pair_fdr_method="by",
        compute_identity_persistence=True,
        dyadic_persistence_hypotheses=(
            ("raw", "envelope_attenuation_1.00"),
            ("raw", SIGNED_ENVELOPE_PROFILE),
        ),
        dyadic_entity_effect_scopes=("global", "segment", "exact_context"),
        persistence_minimum_time_clusters=(1, 2),
        progress=args.progress,
    )
    pair_summaries = spectral_specificity["pair_summary"]
    testable = [
        row
        for row in pair_summaries
        if row["raw_profile"]["domain_regroup"]["empirical_p_ge_observed"] is not None
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "source_report": str(args.source_report),
        "method": {
            "purpose": (
                "iteratively remove the post-hoc identified acquisition/processing "
                "layer and test whether residual pair-profile structure remains"
            ),
            "known_layer_use": (
                "sample rate and source processing family define corpus strata; "
                "geography, oceanographic quantities, and event labels remain unused"
            ),
            "candidate_surface": "within-acquisition-stratum pair profiles only",
            "envelope_normalization": "independent within each acquisition stratum",
            "pair_domain_control": (
                "exact-occurrence matched alternatives from the same segment, "
                "group, window index, and acquisition stratum"
            ),
            "control_coverage_gate": (
                "a pair is testable only when every occurrence has at least one "
                "same-window same-stratum alternative and at least two matched "
                "occurrences exist"
            ),
            "multiplicity": (
                "joint BH and arbitrary-dependence-safe BY FDR are computed across "
                "every supported pair, waveform view, and attenuation representation "
                "for each control family; BY is the selected decision rule"
            ),
            "aggregate_identity_persistence": (
                "within each exact context, all local pair-alignment ranks are "
                "preserved and reassigned among pair identities; full, early, and "
                "late chronological scopes are corrected jointly"
            ),
            "nested_dyadic_control": (
                "for the canonical raw waveform's two deepest residual profiles, "
                "use an identifiability ladder of global, segment-specific, and "
                "exact-context additive buoy effects; then use within-context "
                "Freedman-Lane residual permutations to test whether pair identity "
                "retains structure beyond the declared buoy-effect model"
            ),
            "independent_time_gate": (
                "aggregate identity and nested dyadic controls are run both for "
                "all repeated pairs and for pairs recurring in at least two time "
                "clusters separated by the declared replication interval"
            ),
        },
        "signature": {
            "null_repeats": args.null_repeats,
            "replication_separation_hours": args.replication_separation_hours,
            "seed": args.seed,
            "source_report": str(args.source_report),
        },
        "station_metadata": metadata,
        "entity_strata": entity_strata,
        "stratum_entity_counts": dict(Counter(entity_strata.values())),
        "source_windows": int(source["completed_windows"]),
        "candidate_pair_view_rows": len(pair_summaries),
        "testable_pair_view_rows": len(testable),
        "spectral_specificity": spectral_specificity,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--null-repeats", type=int, default=4999)
    parser.add_argument("--replication-separation-hours", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.null_repeats < 1:
        parser.error("--null-repeats must be positive")
    if args.replication_separation_hours <= 0.0:
        parser.error("--replication-separation-hours must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_control(args)
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
