"""Run a bounded spectral+stigmergy observation on CHB-MIT EEG EDF files.

This experiment keeps seizure labels out of the detector. Labels from the
CHB-MIT summary files are used only after observation to attribute surfaced
cross-channel structure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from spectral_forecast.observation import (
    ObservationResult,
    ScoreName,
    StigmergyPoint,
    build_stigmergy,
    observe_series,
)


DEFAULT_CHANNELS = ("FP1-F7", "T7-P7", "FZ-CZ", "CZ-PZ", "FP2-F8", "P8-O2")


@dataclass(frozen=True)
class EdfSignal:
    """One EDF signal converted to physical units."""

    label: str
    sample_rate: float
    values: NDArray[np.float64]


@dataclass(frozen=True)
class EdfHeader:
    """EDF header fields needed by the experiment."""

    path: Path
    header_bytes: int
    n_records: int
    record_duration_seconds: float
    labels: tuple[str, ...]
    physical_min: NDArray[np.float64]
    physical_max: NDArray[np.float64]
    digital_min: NDArray[np.float64]
    digital_max: NDArray[np.float64]
    samples_per_record: NDArray[np.int64]


@dataclass(frozen=True)
class FileSeizures:
    """Seizure labels parsed from a CHB-MIT patient summary."""

    file_name: str
    seizures: tuple[tuple[float, float], ...]


def _decode_ascii(raw: bytes) -> str:
    return raw.decode("ascii", errors="ignore").strip()


def _read_field_block(handle: Any, count: int, width: int) -> list[str]:
    return [_decode_ascii(handle.read(width)) for _ in range(count)]


def read_edf_header(path: Path) -> EdfHeader:
    """Read enough EDF header metadata to load selected signals."""

    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"{path} is too small to be an EDF file")
        header_bytes = int(_decode_ascii(fixed[184:192]))
        n_records = int(_decode_ascii(fixed[236:244]))
        record_duration = float(_decode_ascii(fixed[244:252]))
        n_signals = int(_decode_ascii(fixed[252:256]))

        labels = tuple(_read_field_block(handle, n_signals, 16))
        _read_field_block(handle, n_signals, 80)  # transducer
        _read_field_block(handle, n_signals, 8)  # physical dimension
        physical_min = np.asarray(_read_field_block(handle, n_signals, 8), dtype=np.float64)
        physical_max = np.asarray(_read_field_block(handle, n_signals, 8), dtype=np.float64)
        digital_min = np.asarray(_read_field_block(handle, n_signals, 8), dtype=np.float64)
        digital_max = np.asarray(_read_field_block(handle, n_signals, 8), dtype=np.float64)
        _read_field_block(handle, n_signals, 80)  # prefilter
        samples_per_record = np.asarray(_read_field_block(handle, n_signals, 8), dtype=np.int64)

    if n_records <= 0:
        raise ValueError(f"{path} has unsupported EDF record count {n_records}")
    if record_duration <= 0:
        raise ValueError(f"{path} has invalid EDF record duration {record_duration}")

    return EdfHeader(
        path=path,
        header_bytes=header_bytes,
        n_records=n_records,
        record_duration_seconds=record_duration,
        labels=labels,
        physical_min=physical_min,
        physical_max=physical_max,
        digital_min=digital_min,
        digital_max=digital_max,
        samples_per_record=samples_per_record,
    )


def load_edf_signals(path: Path, channels: Sequence[str]) -> list[EdfSignal]:
    """Load selected EDF signals by label.

    Duplicate labels are resolved to the first matching signal, which is enough
    for CHB-MIT pilot runs where channel names are used as coarse sensors.
    """

    header = read_edf_header(path)
    requested = list(channels)
    first_index: dict[str, int] = {}
    for index, label in enumerate(header.labels):
        first_index.setdefault(label, index)

    missing = [label for label in requested if label not in first_index]
    if missing:
        raise ValueError(f"{path} is missing requested EDF channels: {', '.join(missing)}")

    selected = {first_index[label]: label for label in requested}
    buffers = {
        index: np.empty(header.n_records * int(header.samples_per_record[index]), dtype=np.float64)
        for index in selected
    }
    offsets = {index: 0 for index in selected}

    with path.open("rb") as handle:
        handle.seek(header.header_bytes)
        for _record in range(header.n_records):
            for index, count in enumerate(header.samples_per_record):
                raw = handle.read(int(count) * 2)
                if len(raw) != int(count) * 2:
                    raise ValueError(f"{path} ended unexpectedly while reading EDF samples")
                if index not in selected:
                    continue
                digital = np.frombuffer(raw, dtype="<i2").astype(np.float64)
                scale = (
                    (header.physical_max[index] - header.physical_min[index])
                    / (header.digital_max[index] - header.digital_min[index])
                )
                physical = header.physical_min[index] + (digital - header.digital_min[index]) * scale
                start = offsets[index]
                buffers[index][start : start + len(physical)] = physical
                offsets[index] += len(physical)

    out = []
    for index in selected:
        sample_rate = float(header.samples_per_record[index]) / header.record_duration_seconds
        out.append(EdfSignal(label=selected[index], sample_rate=sample_rate, values=buffers[index]))
    return out


def parse_chbmit_summary(path: Path) -> dict[str, FileSeizures]:
    """Parse seizure intervals from a CHB-MIT patient summary file."""

    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n(?=File Name: )", text)
    out: dict[str, FileSeizures] = {}
    for block in blocks:
        file_match = re.search(r"File Name:\s+(\S+)", block)
        if not file_match:
            continue
        file_name = file_match.group(1)
        starts = [
            float(value)
            for value in re.findall(
                r"Seizure(?: \d+)? Start Time:\s+([0-9.]+)\s+seconds",
                block,
            )
        ]
        ends = [
            float(value)
            for value in re.findall(
                r"Seizure(?: \d+)? End Time:\s+([0-9.]+)\s+seconds",
                block,
            )
        ]
        out[file_name] = FileSeizures(
            file_name=file_name,
            seizures=tuple(zip(starts, ends, strict=False)),
        )
    return out


def downsample_and_standardize(
    values: NDArray[np.float64],
    *,
    source_rate: float,
    target_rate: float,
) -> NDArray[np.float64]:
    """Resample a signal and put it on a robust comparable scale."""

    y = np.asarray(values, dtype=np.float64)
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if source_rate <= 0:
        raise ValueError("source_rate must be positive")
    if abs(source_rate - target_rate) > 1e-9:
        rounded_source = int(round(source_rate))
        rounded_target = int(round(target_rate))
        if abs(source_rate - rounded_source) > 1e-6 or abs(target_rate - rounded_target) > 1e-6:
            raise ValueError("Only integer sample rates are supported for pilot resampling")
        y = resample_poly(y, rounded_target, rounded_source).astype(np.float64)

    center = float(np.median(y))
    mad = float(np.median(np.abs(y - center)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.std(y))
    if scale < 1e-9:
        scale = 1.0
    return ((y - center) / scale).astype(np.float64)


def _point_phase(
    seconds: float,
    seizures: Sequence[tuple[float, float]],
    *,
    preictal_seconds: float,
    postictal_seconds: float,
) -> str:
    for start, end in seizures:
        if start <= seconds <= end:
            return "ictal"
        if start - preictal_seconds <= seconds < start:
            return "preictal"
        if end < seconds <= end + postictal_seconds:
            return "postictal"
    return "interictal"


def _nearest_seizure_distance(
    seconds: float,
    seizures: Sequence[tuple[float, float]],
) -> float | None:
    if not seizures:
        return None
    distances = []
    for start, end in seizures:
        if start <= seconds <= end:
            distances.append(0.0)
        else:
            distances.append(min(abs(seconds - start), abs(seconds - end)))
    return float(min(distances))


def _score_matrix(
    results: Sequence[ObservationResult],
    score: ScoreName,
) -> tuple[list[int], NDArray[np.float64]]:
    by_series = [{point.index: point.score(score) for point in result.points} for result in results]
    common = sorted(set.intersection(*(set(row) for row in by_series)))
    matrix = np.asarray([[row[index] for row in by_series] for index in common], dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    return common, matrix


def _multi_channel_emissions(
    matrix: NDArray[np.float64],
    *,
    threshold: float,
    min_active_channels: int,
) -> NDArray[np.float64]:
    excess = np.maximum(matrix - threshold, 0.0)
    active = np.sum(excess > 0.0, axis=1)
    emissions = np.sum(excess, axis=1)
    emissions[active < min_active_channels] = 0.0
    return emissions.astype(np.float64)


def permutation_null_summary(
    results: Sequence[ObservationResult],
    *,
    score: ScoreName,
    threshold: float,
    min_active_channels: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Break cross-channel timing while preserving per-channel score distributions."""

    anchors, matrix = _score_matrix(results, score)
    observed_by_index = _multi_channel_emissions(
        matrix,
        threshold=threshold,
        min_active_channels=min_active_channels,
    )
    observed_total = float(np.sum(observed_by_index))

    rng = np.random.default_rng(seed)
    null_totals = []
    for _ in range(repeats):
        permuted = matrix.copy()
        for column in range(permuted.shape[1]):
            permuted[:, column] = rng.permutation(permuted[:, column])
        null_totals.append(
            float(
                np.sum(
                    _multi_channel_emissions(
                        permuted,
                        threshold=threshold,
                        min_active_channels=min_active_channels,
                    )
                )
            )
        )

    null = np.asarray(null_totals, dtype=np.float64)
    null_mean = float(np.mean(null)) if len(null) else 0.0
    null_std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
    exceedances = int(np.sum(null >= observed_total))
    return {
        "anchors": len(anchors),
        "observed_total": observed_total,
        "observed_active_windows": int(np.sum(observed_by_index > 0.0)),
        "null_repeats": repeats,
        "null_mean": null_mean,
        "null_std": null_std,
        "observed_minus_null": observed_total - null_mean,
        "z_effect": (observed_total - null_mean) / null_std if null_std > 0 else None,
        "null_exceedances": exceedances,
        "empirical_p_ge_observed": (exceedances + 1) / (repeats + 1),
        "empirical_p_floor": 1 / (repeats + 1),
        "unique_null_totals": int(len(set(float(value) for value in null))),
        "min_active_channels": min_active_channels,
        "threshold": threshold,
    }


def _top_rows(
    points: Sequence[StigmergyPoint],
    *,
    sample_rate: float,
    seizures: Sequence[tuple[float, float]],
    preictal_seconds: float,
    postictal_seconds: float,
    top: int,
) -> list[dict[str, Any]]:
    rows = []
    for point in sorted(points, key=lambda item: item.pheromone, reverse=True)[:top]:
        seconds = float(point.index) / sample_rate
        rows.append(
            {
                "index": point.index,
                "seconds": seconds,
                "phase": _point_phase(
                    seconds,
                    seizures,
                    preictal_seconds=preictal_seconds,
                    postictal_seconds=postictal_seconds,
                ),
                "nearest_seizure_distance_seconds": _nearest_seizure_distance(seconds, seizures),
                "pheromone": point.pheromone,
                "emission": point.emission,
                "active_series_count": point.active_series_count,
                "active_series": list(point.active_series),
                "max_score": point.max_score,
                "mean_score": point.mean_score,
            }
        )
    return rows


def _phase_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        phase = str(row["phase"])
        counts[phase] = counts.get(phase, 0) + 1
    return dict(sorted(counts.items()))


def _coherent_rows(
    results: Sequence[ObservationResult],
    *,
    score: ScoreName,
    threshold: float,
    min_active_channels: int,
    sample_rate: float,
    seizures: Sequence[tuple[float, float]],
    preictal_seconds: float,
    postictal_seconds: float,
    top: int,
) -> list[dict[str, Any]]:
    anchors, matrix = _score_matrix(results, score)
    series = [result.series for result in results]
    emissions = _multi_channel_emissions(
        matrix,
        threshold=threshold,
        min_active_channels=min_active_channels,
    )
    rows = []
    for row_index, emission in sorted(
        enumerate(emissions),
        key=lambda item: item[1],
        reverse=True,
    )[:top]:
        anchor = anchors[row_index]
        seconds = float(anchor) / sample_rate
        scores = matrix[row_index]
        active = [
            {"series": name, "score": float(value)}
            for name, value in zip(series, scores, strict=True)
            if value > threshold
        ]
        rows.append(
            {
                "index": anchor,
                "seconds": seconds,
                "phase": _point_phase(
                    seconds,
                    seizures,
                    preictal_seconds=preictal_seconds,
                    postictal_seconds=postictal_seconds,
                ),
                "nearest_seizure_distance_seconds": _nearest_seizure_distance(seconds, seizures),
                "emission": float(emission),
                "active_series_count": len(active),
                "active_series": active,
                "max_score": float(np.max(scores)) if len(scores) else 0.0,
                "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
            }
        )
    return rows


def observe_file(
    path: Path,
    *,
    channels: Sequence[str],
    summary_labels: dict[str, FileSeizures],
    target_sample_rate: float,
    baseline: int,
    adaptive_window: int,
    stride: int,
    score: ScoreName,
    emission_threshold: float,
    decay: float,
    min_active_channels: int,
    null_repeats: int,
    seed: int,
    top: int,
    preictal_seconds: float,
    postictal_seconds: float,
) -> dict[str, Any]:
    raw_signals = load_edf_signals(path, channels)
    observed_signals = [
        EdfSignal(
            label=signal.label,
            sample_rate=target_sample_rate,
            values=downsample_and_standardize(
                signal.values,
                source_rate=signal.sample_rate,
                target_rate=target_sample_rate,
            ),
        )
        for signal in raw_signals
    ]

    results = [
        observe_series(
            signal.values,
            series=signal.label,
            baseline_size=baseline,
            adaptive_window=adaptive_window,
            stride=stride,
            sample_rate=signal.sample_rate,
        )
        for signal in observed_signals
    ]
    stig = build_stigmergy(
        results,
        score=score,
        emission_threshold=emission_threshold,
        decay=decay,
    )

    labels = summary_labels.get(path.name, FileSeizures(path.name, ()))
    top_rows = _top_rows(
        stig.points,
        sample_rate=target_sample_rate,
        seizures=labels.seizures,
        preictal_seconds=preictal_seconds,
        postictal_seconds=postictal_seconds,
        top=top,
    )
    null = permutation_null_summary(
        results,
        score=score,
        threshold=emission_threshold,
        min_active_channels=min_active_channels,
        repeats=null_repeats,
        seed=seed,
    )
    coherent_rows = _coherent_rows(
        results,
        score=score,
        threshold=emission_threshold,
        min_active_channels=min_active_channels,
        sample_rate=target_sample_rate,
        seizures=labels.seizures,
        preictal_seconds=preictal_seconds,
        postictal_seconds=postictal_seconds,
        top=top,
    )

    return {
        "file": path.name,
        "path": str(path),
        "source_sample_rate": raw_signals[0].sample_rate if raw_signals else None,
        "target_sample_rate": target_sample_rate,
        "samples": int(len(observed_signals[0].values)) if observed_signals else 0,
        "duration_seconds": (
            float(len(observed_signals[0].values)) / target_sample_rate if observed_signals else 0.0
        ),
        "channels": [signal.label for signal in observed_signals],
        "seizures": [{"start": start, "end": end} for start, end in labels.seizures],
        "observations": {
            "series": len(results),
            "points_per_series": len(results[0].points) if results else 0,
            "score": score,
        },
        "stigmergy": {
            "points": len(stig.points),
            "top": top_rows,
            "top_phase_counts": _phase_counts(top_rows),
            "coherent_top": coherent_rows,
            "coherent_top_phase_counts": _phase_counts(coherent_rows),
        },
        "permutation_null": null,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    summary_labels = parse_chbmit_summary(args.summary)
    file_reports = [
        observe_file(
            path,
            channels=args.channels,
            summary_labels=summary_labels,
            target_sample_rate=args.target_sample_rate,
            baseline=args.baseline,
            adaptive_window=args.adaptive_window,
            stride=args.stride,
            score=args.score,
            emission_threshold=args.emission_threshold,
            decay=args.decay,
            min_active_channels=args.min_active_channels,
            null_repeats=args.null_repeats,
            seed=args.seed + offset,
            top=args.top,
            preictal_seconds=args.preictal_minutes * 60.0,
            postictal_seconds=args.postictal_minutes * 60.0,
        )
        for offset, path in enumerate(args.files)
    ]

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "CHB-MIT Scalp EEG Database",
            "source": "https://physionet.org/content/chbmit/1.0.0/",
            "summary_path": str(args.summary),
            "labels_used": "posthoc attribution only",
        },
        "parameters": {
            "channels": list(args.channels),
            "target_sample_rate": args.target_sample_rate,
            "baseline": args.baseline,
            "adaptive_window": args.adaptive_window,
            "stride": args.stride,
            "score": args.score,
            "emission_threshold": args.emission_threshold,
            "decay": args.decay,
            "min_active_channels": args.min_active_channels,
            "null_repeats": args.null_repeats,
            "seed": args.seed,
            "top": args.top,
            "preictal_minutes": args.preictal_minutes,
            "postictal_minutes": args.postictal_minutes,
        },
        "files": file_reports,
    }


def print_report(report: dict[str, Any]) -> None:
    print("CHB-MIT spectral+stigmergy pilot")
    print("  labels=posthoc only")
    params = report["parameters"]
    print(
        "  channels=%d target_rate=%.1fHz baseline=%d adaptive=%d stride=%d nulls=%d"
        % (
            len(params["channels"]),
            params["target_sample_rate"],
            params["baseline"],
            params["adaptive_window"],
            params["stride"],
            params["null_repeats"],
        )
    )
    for row in report["files"]:
        null = row["permutation_null"]
        print("\n%s" % row["file"])
        print(
            "  seizures=%d points/channel=%d active_windows=%d observed=%.3f null_mean=%.3f z=%s p_ge=%.4f exceed=%d/%d"
            % (
                len(row["seizures"]),
                row["observations"]["points_per_series"],
                null["observed_active_windows"],
                null["observed_total"],
                null["null_mean"],
                "None" if null["z_effect"] is None else f"{null['z_effect']:.2f}",
                null["empirical_p_ge_observed"],
                null["null_exceedances"],
                null["null_repeats"],
            )
        )
        print("  top phases=%s" % row["stigmergy"]["top_phase_counts"])
        print("  coherent top phases=%s" % row["stigmergy"]["coherent_top_phase_counts"])
        for point in row["stigmergy"]["coherent_top"][:5]:
            active = ",".join(item["series"] for item in point["active_series"][:5])
            print(
                "    coherent t=%7.1fs phase=%-10s emission=%7.3f active=%d %s"
                % (
                    point["seconds"],
                    point["phase"],
                    point["emission"],
                    point["active_series_count"],
                    active,
                )
            )
        for point in row["stigmergy"]["top"][:5]:
            print(
                "    t=%7.1fs phase=%-10s pheromone=%8.3f emission=%7.3f active=%d %s"
                % (
                    point["seconds"],
                    point["phase"],
                    point["pheromone"],
                    point["emission"],
                    point["active_series_count"],
                    ",".join(point["active_series"][:5]),
                )
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="CHB-MIT EDF files")
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="CHB-MIT patient summary file, such as chb01-summary.txt",
    )
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS), help="EDF channel labels")
    parser.add_argument("--target-sample-rate", type=float, default=16.0, help="Pilot resampling rate")
    parser.add_argument("--baseline", type=int, default=1024, help="Frozen baseline samples after resampling")
    parser.add_argument("--adaptive-window", type=int, default=512, help="Sliding window samples after resampling")
    parser.add_argument("--stride", type=int, default=256, help="Observation stride after resampling")
    parser.add_argument(
        "--score",
        choices=["frozen", "sliding", "drift", "state", "max"],
        default="max",
        help="Observation score used for stigmergy",
    )
    parser.add_argument("--emission-threshold", type=float, default=3.0)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--min-active-channels", type=int, default=2)
    parser.add_argument("--null-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--preictal-minutes", type=float, default=10.0)
    parser.add_argument("--postictal-minutes", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if args.baseline <= args.adaptive_window:
        parser.error("--baseline must be greater than --adaptive-window")
    if args.min_active_channels < 1:
        parser.error("--min-active-channels must be >= 1")
    if args.null_repeats < 1:
        parser.error("--null-repeats must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_experiment(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
