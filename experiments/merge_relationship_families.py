"""Correct PMU relationship-dynamics evidence across tested families."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


def apply_by_fdr(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        (float(row["empirical_p_ge_observed"]), index)
        for index, row in enumerate(rows)
    )
    total = len(ordered)
    harmonic = sum(1.0 / rank for rank in range(1, total + 1))
    adjusted: dict[int, float] = {}
    running = 1.0
    for rank in range(total, 0, -1):
        p_value, index = ordered[rank - 1]
        running = min(running, p_value * total / rank)
        adjusted[index] = min(1.0, running * harmonic)
    for index, row in enumerate(rows):
        q_value = adjusted[index]
        row["cross_family_fdr_by_q_value"] = q_value
        row["detected_cross_family_fdr_by"] = bool(
            q_value <= 0.05 and float(row["observed_minus_null"]) > 0.0
        )


def merge_artifacts(paths: Sequence[Path], minimum_fraction: float) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two relationship-family artifacts are required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(len(payload["corpora"]) != 1 for payload in payloads):
        raise ValueError("each artifact must contain exactly one corpus")
    corpus_paths = {payload["corpora"][0]["path"] for payload in payloads}
    if len(corpus_paths) != 1:
        raise ValueError("artifacts must describe the same PMU corpus")
    event_counts = {payload["corpora"][0]["events_analyzed"] for payload in payloads}
    if len(event_counts) != 1:
        raise ValueError("artifacts must contain the same event count")

    events = []
    counts: Counter[tuple[str, str, str]] = Counter()
    event_count = event_counts.pop()
    for event_index in range(event_count):
        rows = []
        for payload in payloads:
            family = str(payload["parameters"]["relationship_family"])
            event = payload["corpora"][0]["events"][event_index]
            if int(event["event_index"]) != event_index:
                raise ValueError("event ordering differs across artifacts")
            for source in event["result"]["evidence"]:
                row = dict(source)
                row["family"] = family
                rows.append(row)
        apply_by_fdr(rows)
        detected = sorted(
            (row["family"], row["mode"], row["metric"])
            for row in rows
            if row["detected_cross_family_fdr_by"]
        )
        counts.update(detected)
        events.append(
            {
                "event_index": event_index,
                "detected": [
                    {"family": family, "mode": mode, "metric": metric}
                    for family, mode, metric in detected
                ],
                "evidence": rows,
            }
        )
    required = math.ceil(event_count * minimum_fraction)
    recurring = sorted(key for key, count in counts.items() if count >= required)
    return {
        "method": {
            "multiplicity": (
                "Benjamini-Yekutieli across all relationship families, modes, "
                "and metrics within each event"
            ),
            "minimum_event_fraction": minimum_fraction,
            "required_events": required,
        },
        "corpus": corpus_paths.pop(),
        "families": sorted(
            str(payload["parameters"]["relationship_family"]) for payload in payloads
        ),
        "events_analyzed": event_count,
        "detected_event_counts": [
            {
                "family": family,
                "mode": mode,
                "metric": metric,
                "events": count,
                "fraction": count / event_count,
            }
            for (family, mode, metric), count in sorted(counts.items())
        ],
        "recurring": [
            {"family": family, "mode": mode, "metric": metric}
            for family, mode, metric in recurring
        ],
        "events": events,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, required=True)
    parser.add_argument("--minimum-event-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = merge_artifacts(args.artifact, args.minimum_event_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recurring = ",".join(
        f"{row['family']}:{row['mode']}:{row['metric']}" for row in report["recurring"]
    )
    print(
        f"{report['corpus']}: events={report['events_analyzed']} "
        f"recurring={recurring or 'none'}"
    )


if __name__ == "__main__":
    main()
