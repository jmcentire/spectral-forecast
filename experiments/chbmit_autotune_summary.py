"""Summarize CHB-MIT autotune result JSON files as a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _phase_rate(summary: dict[str, Any], phase: str) -> float:
    anchors = summary.get("anchor_phase_counts", {})
    positives = summary.get("positive_phase_counts", {})
    return _safe_rate(int(positives.get(phase, 0)), int(anchors.get(phase, 0)))


def _phase_count(summary: dict[str, Any], key: str, phase: str) -> int:
    return int(summary.get(key, {}).get(phase, 0))


def _subject(report: dict[str, Any], path: Path) -> str:
    files = report.get("parameters", {}).get("files", [])
    if files:
        parent = Path(str(files[0])).parent.name
        if parent:
            return parent
    stem = path.stem
    for part in stem.split("-"):
        if part.startswith("chb"):
            return part
    return stem


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def summarize_file(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    validation = report.get("validation", {}).get("result") or {}
    calibration = report.get("calibration", {}).get("best") or {}
    phase = report.get("validation", {}).get("phase_summary") or {}
    candidate = validation.get("candidate") or report.get("validation", {}).get("selected_candidate") or {}
    return {
        "path": path,
        "subject": _subject(report, path),
        "files": len(report.get("parameters", {}).get("files", [])),
        "calibration_accepted": bool(calibration.get("accepted", False)),
        "validation_accepted": bool(validation.get("accepted", False)),
        "validation_delta": validation.get("observed_minus_null_total"),
        "validation_z": validation.get("z_effect"),
        "validation_p_ge": validation.get("null_total_empirical_p_ge_observed"),
        "validation_p_floor": validation.get("null_total_empirical_p_floor"),
        "validation_unique_nulls": validation.get("null_total_unique_repeats"),
        "positive_file_fraction": validation.get("positive_file_fraction"),
        "baseline": candidate.get("baseline_size"),
        "adaptive": candidate.get("adaptive_window"),
        "threshold": candidate.get("emission_threshold"),
        "min_active": candidate.get("min_active_series"),
        "ictal_rate": _phase_rate(phase, "ictal"),
        "preictal_rate": _phase_rate(phase, "preictal"),
        "postictal_rate": _phase_rate(phase, "postictal"),
        "interictal_rate": _phase_rate(phase, "interictal"),
        "ictal_positive": _phase_count(phase, "positive_phase_counts", "ictal"),
        "ictal_anchors": _phase_count(phase, "anchor_phase_counts", "ictal"),
        "top_ictal": _phase_count(phase, "top_phase_counts", "ictal"),
        "top_preictal": _phase_count(phase, "top_phase_counts", "preictal"),
        "top_postictal": _phase_count(phase, "top_phase_counts", "postictal"),
        "top_interictal": _phase_count(phase, "top_phase_counts", "interictal"),
    }


def render_markdown(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# CHB-MIT autotune cross-subject summary",
        "",
        "Labels are not used for candidate selection. Phase rates are post-hoc overlays on the selected held-out validation run.",
        "",
        "| Subject | Files | Val accepted | Delta | z effect | p_ge | p floor | Pos files | Config | Ictal pos rate | Pre | Post | Inter | Top phases I/P/Post/Inter |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: item["subject"]):
        config = (
            f"{row['baseline']}/{row['adaptive']}"
            f" t={_fmt(row['threshold'], 1)} min={row['min_active']}"
        )
        top = (
            f"{row['top_ictal']}/"
            f"{row['top_preictal']}/"
            f"{row['top_postictal']}/"
            f"{row['top_interictal']}"
        )
        lines.append(
            "| {subject} | {files} | {accepted} | {delta} | {z} | {p_ge} | {p_floor} | {pos} | {config} | {ictal} | {pre} | {post} | {inter} | {top} |".format(
                subject=row["subject"],
                files=row["files"],
                accepted="yes" if row["validation_accepted"] else "no",
                delta=_fmt(row["validation_delta"], 1),
                z=_fmt(row["validation_z"], 2),
                p_ge=_fmt(row["validation_p_ge"], 4),
                p_floor=_fmt(row["validation_p_floor"], 4),
                pos=_fmt(row["positive_file_fraction"], 2),
                config=config,
                ictal=f"{row['ictal_positive']}/{row['ictal_anchors']}={_fmt(row['ictal_rate'], 3)}",
                pre=_fmt(row["preictal_rate"], 3),
                post=_fmt(row["postictal_rate"], 3),
                inter=_fmt(row["interictal_rate"], 3),
                top=top,
            )
        )
    lines.extend(
        [
            "",
            "Use empirical p as a floor-limited bound when null exceedances are zero. The z column is a standardized effect size, not a normal-theory p-value.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="CHB-MIT autotune JSON files")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = [summarize_file(path) for path in args.files]
    markdown = render_markdown(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
