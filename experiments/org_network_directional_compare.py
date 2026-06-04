"""Compare directional-quality candidate reports with matched control reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.org_network_directional_quality import compare_candidate_to_controls


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison(candidate_path: Path, control_paths: Sequence[Path]) -> dict[str, Any]:
    candidate = _read(candidate_path)
    controls = [_read(path) for path in control_paths]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_path": str(candidate_path),
        "control_paths": [str(path) for path in control_paths],
        "dataset": candidate["dataset"],
        "candidate_projection": candidate["analysis_projection"],
        "control_projections": [control["analysis_projection"] for control in controls],
        "comparison": compare_candidate_to_controls(candidate, controls),
    }


def print_report(report: Mapping[str, Any]) -> None:
    comparison = report["comparison"]
    print(
        "dataset=%s grade=%s selective=%s shared_stronger=%s"
        % (
            report["dataset"].get("dataset", "custom"),
            comparison["grade"],
            ",".join(comparison["selective_candidate_mechanisms"]) or "none",
            ",".join(comparison["shared_but_descriptively_stronger_mechanisms"]) or "none",
        )
    )
    for name, row in comparison["metrics"].items():
        print(
            "  %-18s candidate_replicated=%s control_replicated=%s delta_advantage=% .6f"
            % (
                name,
                row["candidate_replicated"],
                row["control_replicated"],
                row["descriptive_delta_advantage"],
            )
        )
    print("control_gap_confirmed_by_second_level_null=False")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_comparison(args.candidate, args.control)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
