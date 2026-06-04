"""Supervised CHB-MIT ablation for agnostic latent-structure features.

This script is intentionally separate from the label-free CHB-MIT autotune
experiments. Here labels are used to train/evaluate a simple predictor so we
can measure whether the agnostic spectral+stigmergic feature layer contributes
additional signal beyond pedestrian window statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numpy.typing import NDArray

from experiments.chbmit_observe import (
    DEFAULT_CHANNELS,
    FileSeizures,
    downsample_and_standardize,
    load_edf_signals,
    parse_chbmit_summary,
)
from spectral_forecast.autotune import AutoTuneConfig
from spectral_forecast.observation import ObservationResult, ScoreName, observe_series

try:
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - exercised only when optional sklearn is absent.
    SelectKBest = None  # type: ignore[assignment]
    f_classif = None  # type: ignore[assignment]
    SimpleImputer = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    average_precision_score = None  # type: ignore[assignment]
    brier_score_loss = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


SCORE_NAMES: tuple[ScoreName, ...] = ("frozen", "sliding", "drift", "state", "max")


@dataclass(frozen=True)
class LabelInfo:
    """Window label for supervised ablation."""

    label: int
    phase: str
    event_id: str | None
    event_start_seconds: float | None


@dataclass(frozen=True)
class FeatureRow:
    """One supervised example row."""

    subject: str
    file: str
    anchor: int
    seconds: float
    label: int
    phase: str
    event_id: str | None
    event_start_seconds: float | None
    features: Mapping[str, float]


def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _stats(prefix: str, values: NDArray[np.float64]) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        finite = np.asarray([0.0], dtype=np.float64)
    return {
        f"{prefix}_mean": _finite(float(np.mean(finite))),
        f"{prefix}_std": _finite(float(np.std(finite))),
        f"{prefix}_min": _finite(float(np.min(finite))),
        f"{prefix}_max": _finite(float(np.max(finite))),
        f"{prefix}_q25": _finite(float(np.quantile(finite, 0.25))),
        f"{prefix}_median": _finite(float(np.median(finite))),
        f"{prefix}_q75": _finite(float(np.quantile(finite, 0.75))),
        f"{prefix}_abs_mean": _finite(float(np.mean(np.abs(finite)))),
    }


def _entropy(weights: NDArray[np.float64]) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    total = float(np.sum(weights))
    if total <= 0.0 or len(weights) <= 1:
        return 0.0
    probs = weights / total
    entropy = -float(np.sum(probs * np.log(probs)))
    return _finite(entropy / math.log(len(probs)))


def _score_matrices(
    results: Sequence[ObservationResult],
    scores: Sequence[ScoreName],
) -> tuple[list[int], dict[ScoreName, NDArray[np.float64]], list[str]]:
    if not results:
        raise ValueError("at least one observation result is required")
    by_series = [
        {point.index: point for point in result.points}
        for result in results
    ]
    common = sorted(set.intersection(*(set(row) for row in by_series)))
    if not common:
        raise ValueError("observation results have no common anchors")
    matrices = {
        score: np.asarray(
            [[row[index].score(score) for row in by_series] for index in common],
            dtype=np.float64,
        )
        for score in scores
    }
    matrices = {
        score: np.where(np.isfinite(matrix), matrix, 0.0)
        for score, matrix in matrices.items()
    }
    return common, matrices, [result.series for result in results]


def _row_emissions(
    matrix: NDArray[np.float64],
    config: AutoTuneConfig,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    excess = np.maximum(matrix - config.emission_threshold, 0.0)
    active = excess > 0.0
    emissions = np.sum(excess, axis=1)
    active_counts = np.sum(active, axis=1)
    emissions[active_counts < config.min_active_series] = 0.0
    return emissions.astype(np.float64), active.astype(np.bool_), excess.astype(np.float64)


def _pedestrian_features(window: NDArray[np.float64]) -> dict[str, float]:
    if window.ndim != 2 or window.shape[0] < 1 or window.shape[1] < 2:
        raise ValueError("pedestrian feature window must be channels x samples")
    features: dict[str, float] = {}
    means = np.mean(window, axis=1)
    stds = np.std(window, axis=1)
    ranges = np.max(window, axis=1) - np.min(window, axis=1)
    diffs = np.diff(window, axis=1)
    diff_stds = np.std(diffs, axis=1)
    line_lengths = np.mean(np.abs(diffs), axis=1)
    slopes = (window[:, -1] - window[:, 0]) / max(window.shape[1] - 1, 1)
    centered = window - np.median(window, axis=1, keepdims=True)
    zero_crossing = np.mean(np.diff(np.signbit(centered), axis=1), axis=1)
    energy = np.mean(window**2, axis=1)

    features.update(_stats("ped_channel_mean", means))
    features.update(_stats("ped_channel_std", stds))
    features.update(_stats("ped_channel_range", ranges))
    features.update(_stats("ped_channel_diff_std", diff_stds))
    features.update(_stats("ped_channel_line_length", line_lengths))
    features.update(_stats("ped_channel_slope", slopes))
    features.update(_stats("ped_channel_zero_crossing", zero_crossing.astype(np.float64)))
    features.update(_stats("ped_channel_energy", energy))

    if window.shape[0] >= 2:
        channel_std = np.std(window, axis=1)
        valid = channel_std > 1e-9
        if int(np.sum(valid)) >= 2:
            corr = np.corrcoef(window[valid])
            upper = corr[np.triu_indices_from(corr, k=1)]
            upper = np.where(np.isfinite(upper), upper, 0.0)
        else:
            upper = np.asarray([0.0], dtype=np.float64)
    else:
        upper = np.asarray([0.0], dtype=np.float64)
    features.update(_stats("ped_channel_corr", upper.astype(np.float64)))
    features.update(_stats("ped_channel_corr_abs", np.abs(upper).astype(np.float64)))
    return features


def _spectral_features(
    matrices: Mapping[ScoreName, NDArray[np.float64]],
    row_index: int,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for score in SCORE_NAMES:
        row = np.asarray(matrices[score][row_index], dtype=np.float64)
        features.update(_stats(f"spectral_{score}", row))
        features[f"spectral_{score}_count_ge_1"] = float(np.sum(row >= 1.0))
        features[f"spectral_{score}_count_ge_2"] = float(np.sum(row >= 2.0))
        features[f"spectral_{score}_count_ge_3"] = float(np.sum(row >= 3.0))
    return features


def _active_pattern_entropy(active: NDArray[np.bool_]) -> float:
    if active.size == 0:
        return 0.0
    rows = [tuple(bool(item) for item in row) for row in active]
    counts: dict[tuple[bool, ...], int] = {}
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
    return _entropy(np.asarray(list(counts.values()), dtype=np.float64))


def _latent_features(
    matrix: NDArray[np.float64],
    row_index: int,
    config: AutoTuneConfig,
    *,
    history_rows: int,
) -> dict[str, float]:
    features: dict[str, float] = {}
    emissions, active, excess = _row_emissions(matrix, config)
    row = matrix[row_index]
    row_active = active[row_index]
    row_excess = excess[row_index]
    active_count = int(np.sum(row_active))
    gated_emission = float(emissions[row_index])
    raw_emission = float(np.sum(row_excess))
    features["latent_active_count"] = float(active_count)
    features["latent_active_fraction"] = float(active_count / max(matrix.shape[1], 1))
    features["latent_raw_emission"] = _finite(raw_emission)
    features["latent_gated_emission"] = _finite(gated_emission)
    features["latent_excess_entropy"] = _entropy(row_excess)
    features.update(_stats("latent_excess", row_excess))
    features.update(_stats("latent_row_score", row))

    start = max(0, row_index - max(history_rows, 1) + 1)
    hist = matrix[start : row_index + 1]
    hist_emissions = emissions[start : row_index + 1]
    hist_active = active[start : row_index + 1]
    hist_excess = excess[start : row_index + 1]

    pheromone = 0.0
    for emission in hist_emissions:
        pheromone = config.decay * pheromone + float(emission)
    features["latent_recent_pheromone"] = _finite(pheromone)
    features.update(_stats("latent_recent_emission", hist_emissions))
    features.update(_stats("latent_recent_active_count", np.sum(hist_active, axis=1).astype(np.float64)))
    features["latent_recent_pattern_entropy"] = _active_pattern_entropy(hist_active)
    features.update(_stats("latent_recent_channel_persistence", np.mean(hist_active, axis=0).astype(np.float64)))
    features["latent_recent_excess_entropy_mean"] = _finite(
        float(np.mean([_entropy(row_values) for row_values in hist_excess]))
    )

    if len(hist_emissions) >= 2:
        x = np.arange(len(hist_emissions), dtype=np.float64)
        slope = float(np.polyfit(x, hist_emissions, 1)[0])
        diffs = np.diff(hist, axis=0)
        active_changes = np.mean(np.not_equal(hist_active[1:], hist_active[:-1]), axis=1)
        prev_active = hist_active[-2]
        union = int(np.sum(row_active | prev_active))
        jaccard = int(np.sum(row_active & prev_active)) / union if union else 1.0
        features["latent_recent_emission_slope"] = _finite(slope)
        features.update(_stats("latent_recent_score_drift_l2", np.linalg.norm(diffs, axis=1)))
        features.update(_stats("latent_recent_active_hamming", active_changes.astype(np.float64)))
        features["latent_active_jaccard_prev"] = _finite(float(jaccard))
    else:
        features["latent_recent_emission_slope"] = 0.0
        features.update(_stats("latent_recent_score_drift_l2", np.asarray([0.0], dtype=np.float64)))
        features.update(_stats("latent_recent_active_hamming", np.asarray([0.0], dtype=np.float64)))
        features["latent_active_jaccard_prev"] = 1.0
    return features


def _label_anchor(
    subject: str,
    file_name: str,
    seconds: float,
    seizures: Sequence[tuple[float, float]],
    *,
    preictal_seconds: float,
    negative_gap_seconds: float,
    postictal_seconds: float,
) -> LabelInfo | None:
    for event_index, (start, end) in enumerate(seizures):
        if start <= seconds <= end:
            return None
        if end < seconds <= end + postictal_seconds:
            return None
        if start - preictal_seconds <= seconds < start:
            return LabelInfo(
                label=1,
                phase="preictal",
                event_id=f"{subject}:{file_name}:{event_index}:{start:.3f}",
                event_start_seconds=float(start),
            )
        if start - negative_gap_seconds <= seconds < start:
            return None
        if end < seconds <= end + negative_gap_seconds:
            return None
    return LabelInfo(label=0, phase="interictal", event_id=None, event_start_seconds=None)


def build_file_rows(
    subject: str,
    path: Path,
    *,
    labels: Mapping[str, FileSeizures],
    channels: Sequence[str],
    config: AutoTuneConfig,
    target_sample_rate: float,
    feature_window_seconds: float,
    history_rows: int,
    preictal_seconds: float,
    negative_gap_seconds: float,
    postictal_seconds: float,
) -> list[FeatureRow]:
    raw_signals = load_edf_signals(path, channels)
    series = {
        signal.label: downsample_and_standardize(
            signal.values,
            source_rate=signal.sample_rate,
            target_rate=target_sample_rate,
        )
        for signal in raw_signals
    }
    results = [
        observe_series(
            values,
            series=label,
            baseline_size=config.baseline_size,
            adaptive_window=config.adaptive_window,
            stride=config.stride,
            sample_rate=target_sample_rate,
        )
        for label, values in series.items()
    ]
    anchors, matrices, ordered_channels = _score_matrices(results, SCORE_NAMES)
    seizures = labels.get(path.name, FileSeizures(path.name, ())).seizures
    window_samples = max(2, int(round(feature_window_seconds * target_sample_rate)))
    rows: list[FeatureRow] = []
    for row_index, anchor in enumerate(anchors):
        seconds = float(anchor) / target_sample_rate
        label = _label_anchor(
            subject,
            path.name,
            seconds,
            seizures,
            preictal_seconds=preictal_seconds,
            negative_gap_seconds=negative_gap_seconds,
            postictal_seconds=postictal_seconds,
        )
        if label is None:
            continue
        start = anchor - window_samples
        if start < 0:
            continue
        window = np.vstack([series[channel][start:anchor] for channel in ordered_channels])
        if window.shape[1] < 2:
            continue
        features: dict[str, float] = {}
        features.update(_pedestrian_features(window))
        features.update(_spectral_features(matrices, row_index))
        features.update(
            _latent_features(
                matrices["max"],
                row_index,
                config,
                history_rows=history_rows,
            )
        )
        rows.append(
            FeatureRow(
                subject=subject,
                file=path.name,
                anchor=int(anchor),
                seconds=seconds,
                label=label.label,
                phase=label.phase,
                event_id=label.event_id,
                event_start_seconds=label.event_start_seconds,
                features=features,
            )
        )
    return rows


def _subject_summary_path(data_root: Path, subject: str) -> Path:
    return data_root / subject / f"{subject}-summary.txt"


def _subject_edf_paths(data_root: Path, subject: str, *, max_files: int) -> list[Path]:
    paths = sorted((data_root / subject).glob(f"{subject}_*.edf"))
    if max_files > 0:
        return paths[:max_files]
    return paths


def build_dataset(args: argparse.Namespace, config: AutoTuneConfig) -> tuple[list[FeatureRow], list[dict[str, Any]]]:
    rows: list[FeatureRow] = []
    skipped: list[dict[str, Any]] = []
    started = time.time()
    for subject in args.subjects:
        summary_path = _subject_summary_path(args.data_root, subject)
        labels = parse_chbmit_summary(summary_path)
        paths = _subject_edf_paths(args.data_root, subject, max_files=args.max_files_per_subject)
        for file_index, path in enumerate(paths):
            try:
                file_rows = build_file_rows(
                    subject,
                    path,
                    labels=labels,
                    channels=args.channels,
                    config=config,
                    target_sample_rate=args.target_sample_rate,
                    feature_window_seconds=args.feature_window_seconds,
                    history_rows=args.history_rows,
                    preictal_seconds=args.preictal_minutes * 60.0,
                    negative_gap_seconds=args.negative_gap_minutes * 60.0,
                    postictal_seconds=args.postictal_minutes * 60.0,
                )
            except Exception as exc:  # noqa: BLE001 - bad EDFs should be reported and skipped.
                skipped.append({"subject": subject, "file": str(path), "reason": str(exc)})
                continue
            rows.extend(file_rows)
            if args.progress:
                elapsed = time.time() - started
                positives = sum(row.label for row in rows)
                print(
                    (
                        "chbmit_predictor progress subject=%s file=%d/%d rows=%d positives=%d "
                        "skipped=%d elapsed=%.1fs"
                    )
                    % (subject, file_index + 1, len(paths), len(rows), positives, len(skipped), elapsed),
                    file=sys.stderr,
                    flush=True,
                )
    return rows, skipped


def _feature_names(rows: Sequence[FeatureRow]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(row.features)
    return sorted(names)


def _matrix(rows: Sequence[FeatureRow], names: Sequence[str]) -> NDArray[np.float64]:
    return np.asarray(
        [[float(row.features.get(name, 0.0)) for name in names] for row in rows],
        dtype=np.float64,
    )


def _labels(rows: Sequence[FeatureRow]) -> NDArray[np.int64]:
    return np.asarray([row.label for row in rows], dtype=np.int64)


def _parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def _parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _candidate_feature_counts(feature_count: int, text: str) -> list[int]:
    if feature_count < 1:
        return []
    values = _parse_int_list(text)
    values.append(feature_count)
    return sorted({min(max(1, value), feature_count) for value in values})


def _make_model(
    args: argparse.Namespace,
    *,
    regularization_c: float,
    selected_k: int | None = None,
) -> Any:
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if selected_k is not None:
        steps.append(("select", SelectKBest(score_func=f_classif, k=selected_k)))
    steps.append(
        (
            "logistic",
            LogisticRegression(
                C=regularization_c,
                class_weight="balanced",
                max_iter=args.max_iter,
                random_state=args.seed,
            ),
        )
    )
    return Pipeline(
        steps=steps,
    )


def _select_model_hyperparameters(
    train_rows: Sequence[FeatureRow],
    names: Sequence[str],
    args: argparse.Namespace,
    *,
    feature_selection: bool,
) -> dict[str, Any]:
    """Select model settings with labels from training subjects only."""

    c_values = _parse_float_list(args.regularization_cs)
    k_values: list[int | None]
    if feature_selection:
        k_values = _candidate_feature_counts(len(names), args.selection_ks)
    else:
        k_values = [None]
    train_subjects = sorted({row.subject for row in train_rows})
    if len(train_subjects) < 2:
        return {
            "selected_c": c_values[0],
            "selected_k": k_values[0],
            "inner_pr_auc": None,
            "candidates": [],
            "reason": "fewer than two training subjects",
        }

    candidate_rows = []
    for regularization_c in c_values:
        for selected_k in k_values:
            scores = []
            for validation_subject in train_subjects:
                inner_train = [row for row in train_rows if row.subject != validation_subject]
                inner_validation = [row for row in train_rows if row.subject == validation_subject]
                inner_train_y = _labels(inner_train)
                inner_validation_y = _labels(inner_validation)
                if len(np.unique(inner_train_y)) < 2 or len(np.unique(inner_validation_y)) < 2:
                    continue
                model = _make_model(
                    args,
                    regularization_c=regularization_c,
                    selected_k=selected_k,
                )
                model.fit(_matrix(inner_train, names), inner_train_y)
                probabilities = model.predict_proba(_matrix(inner_validation, names))[:, 1]
                scores.append(float(average_precision_score(inner_validation_y, probabilities)))
            candidate_rows.append(
                {
                    "c": regularization_c,
                    "k": selected_k,
                    "inner_pr_auc": _finite(float(np.mean(scores))) if scores else None,
                    "folds": len(scores),
                }
            )

    scored = [row for row in candidate_rows if row["inner_pr_auc"] is not None]
    if not scored:
        return {
            "selected_c": c_values[0],
            "selected_k": k_values[0],
            "inner_pr_auc": None,
            "candidates": candidate_rows,
            "reason": "no valid inner folds",
        }
    best = max(
        scored,
        key=lambda row: (
            float(row["inner_pr_auc"]),
            -float(row["k"] if row["k"] is not None else len(names)),
            -float(row["c"]),
        ),
    )
    return {
        "selected_c": float(best["c"]),
        "selected_k": best["k"],
        "inner_pr_auc": float(best["inner_pr_auc"]),
        "candidates": candidate_rows,
        "reason": "training-subject inner PR-AUC",
    }


def _select_regularization_c(
    train_rows: Sequence[FeatureRow],
    names: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Backward-compatible wrapper for non-selected models."""

    return _select_model_hyperparameters(
        train_rows,
        names,
        args,
        feature_selection=False,
    )


def _threshold_for_false_alarm_rate(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    row_hours: float,
    false_alarms_per_hour: float,
) -> float:
    negative_scores = np.asarray(probabilities[labels == 0], dtype=np.float64)
    if len(negative_scores) == 0:
        return 1.0
    allowed = int(math.floor(max(false_alarms_per_hour, 0.0) * len(negative_scores) * row_hours))
    ordered = np.sort(negative_scores)[::-1]
    if allowed < 1:
        return float(np.nextafter(np.max(ordered), np.inf))
    if allowed >= len(ordered):
        return float(np.nextafter(np.min(ordered), -np.inf))
    return float(ordered[allowed - 1])


def _event_metrics(
    rows: Sequence[FeatureRow],
    probabilities: NDArray[np.float64],
    threshold: float,
) -> dict[str, float | None]:
    events = sorted({row.event_id for row in rows if row.label == 1 and row.event_id is not None})
    if not events:
        return {
            "event_count": 0.0,
            "event_recall": None,
            "lead_time_minutes_mean": None,
            "lead_time_minutes_median": None,
        }
    hits: dict[str, list[float]] = {}
    for row, probability in zip(rows, probabilities, strict=True):
        if row.label != 1 or row.event_id is None or probability < threshold:
            continue
        if row.event_start_seconds is None:
            continue
        lead = max(0.0, (row.event_start_seconds - row.seconds) / 60.0)
        hits.setdefault(row.event_id, []).append(lead)
    earliest_leads = [max(values) for values in hits.values() if values]
    return {
        "event_count": float(len(events)),
        "event_recall": float(len(hits) / len(events)),
        "lead_time_minutes_mean": (
            _finite(float(np.mean(earliest_leads))) if earliest_leads else None
        ),
        "lead_time_minutes_median": (
            _finite(float(np.median(earliest_leads))) if earliest_leads else None
        ),
    }


def _classification_metrics(
    rows: Sequence[FeatureRow],
    probabilities: NDArray[np.float64],
    threshold: float,
    *,
    row_hours: float,
) -> dict[str, float | None]:
    y = _labels(rows)
    predicted = probabilities >= threshold
    positives = y == 1
    negatives = y == 0
    tp = int(np.sum(predicted & positives))
    fp = int(np.sum(predicted & negatives))
    fn = int(np.sum(~predicted & positives))
    tn = int(np.sum(~predicted & negatives))
    negative_hours = float(np.sum(negatives) * row_hours)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    metrics: dict[str, float | None] = {
        "rows": float(len(rows)),
        "positives": float(np.sum(positives)),
        "negatives": float(np.sum(negatives)),
        "positive_rate": float(np.mean(positives)) if len(rows) else None,
        "threshold": _finite(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": _finite(precision) if precision is not None else None,
        "recall": _finite(recall) if recall is not None else None,
        "false_alarms_per_hour": _finite(fp / negative_hours) if negative_hours > 0 else None,
        "brier": _finite(float(brier_score_loss(y, probabilities))) if brier_score_loss else None,
    }
    if average_precision_score and len(np.unique(y)) == 2:
        metrics["pr_auc"] = _finite(float(average_precision_score(y, probabilities)))
    else:
        metrics["pr_auc"] = None
    if roc_auc_score and len(np.unique(y)) == 2:
        metrics["roc_auc"] = _finite(float(roc_auc_score(y, probabilities)))
    else:
        metrics["roc_auc"] = None
    metrics.update(_event_metrics(rows, probabilities, threshold))
    return metrics


def _mean_metric(rows: Sequence[Mapping[str, float | None]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return _finite(float(np.mean(values)))


def _evaluate_probability_model(
    *,
    train_rows: Sequence[FeatureRow],
    test_rows: Sequence[FeatureRow],
    train_prob: NDArray[np.float64],
    test_prob: NDArray[np.float64],
    row_hours: float,
    args: argparse.Namespace,
) -> dict[str, float | None]:
    threshold = _threshold_for_false_alarm_rate(
        train_prob,
        _labels(train_rows),
        row_hours=row_hours,
        false_alarms_per_hour=args.false_alarms_per_hour,
    )
    return _classification_metrics(
        test_rows,
        test_prob,
        threshold,
        row_hours=row_hours,
    )


def _fit_one_model(
    train_rows: Sequence[FeatureRow],
    test_rows: Sequence[FeatureRow],
    names: Sequence[str],
    args: argparse.Namespace,
    *,
    row_hours: float,
    feature_selection: bool,
) -> dict[str, Any]:
    selection = _select_model_hyperparameters(
        train_rows,
        names,
        args,
        feature_selection=feature_selection,
    )
    model = _make_model(
        args,
        regularization_c=float(selection["selected_c"]),
        selected_k=selection["selected_k"],
    )
    train_x = _matrix(train_rows, names)
    test_x = _matrix(test_rows, names)
    model.fit(train_x, _labels(train_rows))
    train_prob = model.predict_proba(train_x)[:, 1]
    test_prob = model.predict_proba(test_x)[:, 1]
    metrics = _evaluate_probability_model(
        train_rows=train_rows,
        test_rows=test_rows,
        train_prob=train_prob,
        test_prob=test_prob,
        row_hours=row_hours,
        args=args,
    )
    metrics["feature_count"] = float(len(names))
    metrics["selected_c"] = float(selection["selected_c"])
    metrics["selected_k"] = (
        float(selection["selected_k"]) if selection["selected_k"] is not None else None
    )
    metrics["feature_selection"] = bool(feature_selection)  # type: ignore[assignment]
    metrics["inner_pr_auc"] = selection["inner_pr_auc"]
    metrics["model_selection"] = selection  # type: ignore[assignment]
    return metrics


def _family_probability_matrix(
    source_rows: Sequence[FeatureRow],
    target_rows: Sequence[FeatureRow],
    family_groups: Mapping[str, Sequence[str]],
    args: argparse.Namespace,
    *,
    feature_selection: bool,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    columns = []
    selections = []
    for family_name, names in family_groups.items():
        selection = _select_model_hyperparameters(
            source_rows,
            names,
            args,
            feature_selection=feature_selection,
        )
        model = _make_model(
            args,
            regularization_c=float(selection["selected_c"]),
            selected_k=selection["selected_k"],
        )
        model.fit(_matrix(source_rows, names), _labels(source_rows))
        columns.append(model.predict_proba(_matrix(target_rows, names))[:, 1])
        selections.append(
            {
                "family": family_name,
                "feature_count": len(names),
                "selected_c": selection["selected_c"],
                "selected_k": selection["selected_k"],
                "inner_pr_auc": selection["inner_pr_auc"],
                "reason": selection["reason"],
            }
        )
    if not columns:
        raise ValueError("late fusion requires at least one feature family")
    return np.column_stack(columns).astype(np.float64), selections


def _late_fusion_models(
    train_rows: Sequence[FeatureRow],
    test_rows: Sequence[FeatureRow],
    family_groups: Mapping[str, Sequence[str]],
    args: argparse.Namespace,
    *,
    row_hours: float,
) -> dict[str, dict[str, Any]]:
    """Train late-fusion controls using only outer-training labels."""

    train_subjects = sorted({row.subject for row in train_rows})
    if len(train_subjects) < 2:
        return {}
    train_family_prob = np.zeros((len(train_rows), len(family_groups)), dtype=np.float64)
    oof_family_selections: list[dict[str, Any]] = []
    for validation_subject in train_subjects:
        validation_indices = [
            index for index, row in enumerate(train_rows)
            if row.subject == validation_subject
        ]
        inner_train = [
            row for row in train_rows
            if row.subject != validation_subject
        ]
        inner_validation = [train_rows[index] for index in validation_indices]
        if len(np.unique(_labels(inner_train))) < 2 or len(np.unique(_labels(inner_validation))) < 2:
            continue
        probabilities, selections = _family_probability_matrix(
            inner_train,
            inner_validation,
            family_groups,
            args,
            feature_selection=False,
        )
        for local_index, row_index in enumerate(validation_indices):
            train_family_prob[row_index] = probabilities[local_index]
        oof_family_selections.append(
            {
                "validation_subject": validation_subject,
                "families": selections,
            }
        )

    test_family_prob, test_family_selections = _family_probability_matrix(
        train_rows,
        test_rows,
        family_groups,
        args,
        feature_selection=False,
    )
    reports: dict[str, dict[str, Any]] = {}

    train_mean = np.mean(train_family_prob, axis=1)
    test_mean = np.mean(test_family_prob, axis=1)
    mean_metrics = _evaluate_probability_model(
        train_rows=train_rows,
        test_rows=test_rows,
        train_prob=train_mean,
        test_prob=test_mean,
        row_hours=row_hours,
        args=args,
    )
    mean_metrics["feature_count"] = float(len(family_groups))
    mean_metrics["fusion"] = "mean_family_probability"  # type: ignore[assignment]
    mean_metrics["families"] = list(family_groups)  # type: ignore[assignment]
    mean_metrics["family_model_selection"] = test_family_selections  # type: ignore[assignment]
    mean_metrics["oof_family_model_selection"] = oof_family_selections  # type: ignore[assignment]
    reports["late_fusion_mean"] = mean_metrics

    meta = LogisticRegression(
        C=args.fusion_meta_c,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    meta.fit(train_family_prob, _labels(train_rows))
    train_meta = meta.predict_proba(train_family_prob)[:, 1]
    test_meta = meta.predict_proba(test_family_prob)[:, 1]
    meta_metrics = _evaluate_probability_model(
        train_rows=train_rows,
        test_rows=test_rows,
        train_prob=train_meta,
        test_prob=test_meta,
        row_hours=row_hours,
        args=args,
    )
    meta_metrics["feature_count"] = float(len(family_groups))
    meta_metrics["fusion"] = "logistic_family_probability"  # type: ignore[assignment]
    meta_metrics["families"] = list(family_groups)  # type: ignore[assignment]
    meta_metrics["selected_c"] = float(args.fusion_meta_c)
    meta_metrics["family_model_selection"] = test_family_selections  # type: ignore[assignment]
    meta_metrics["oof_family_model_selection"] = oof_family_selections  # type: ignore[assignment]
    reports["late_fusion_logistic"] = meta_metrics
    return reports


def evaluate(rows: Sequence[FeatureRow], args: argparse.Namespace) -> dict[str, Any]:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "scikit-learn is required for predictor ablation; install sklearn or run in the dev environment"
        ) from _SKLEARN_IMPORT_ERROR
    if not rows:
        raise ValueError("no labeled rows were produced")
    feature_names = _feature_names(rows)
    groups = {
        "pedestrian": [name for name in feature_names if name.startswith("ped_")],
        "spectral": [name for name in feature_names if name.startswith("spectral_")],
        "latent": [name for name in feature_names if name.startswith("latent_")],
    }
    ablations = {
        "pedestrian": groups["pedestrian"],
        "pedestrian_spectral": groups["pedestrian"] + groups["spectral"],
        "latent_only": groups["latent"],
        "spectral_latent": groups["spectral"] + groups["latent"],
        "pedestrian_spectral_latent": groups["pedestrian"] + groups["spectral"] + groups["latent"],
    }
    selected_ablations = {
        f"{name}_selected": names
        for name, names in ablations.items()
        if name != "pedestrian"
    }
    model_order = list(ablations) + list(selected_ablations) + [
        "late_fusion_mean",
        "late_fusion_logistic",
    ]
    subjects = sorted({row.subject for row in rows})
    if len(subjects) < 2:
        raise ValueError("leave-one-subject-out evaluation requires at least two subjects")
    row_hours = (args.stride / args.target_sample_rate) / 3600.0
    split_reports: list[dict[str, Any]] = []
    for subject in subjects:
        train_rows = [row for row in rows if row.subject != subject]
        test_rows = [row for row in rows if row.subject == subject]
        train_y = _labels(train_rows)
        test_y = _labels(test_rows)
        split: dict[str, Any] = {
            "test_subject": subject,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_positives": int(np.sum(train_y)),
            "test_positives": int(np.sum(test_y)),
            "models": {},
        }
        for model_name, names in ablations.items():
            if not names:
                continue
            split["models"][model_name] = _fit_one_model(
                train_rows,
                test_rows,
                names,
                args,
                row_hours=row_hours,
                feature_selection=False,
            )
        for model_name, names in selected_ablations.items():
            if not names:
                continue
            split["models"][model_name] = _fit_one_model(
                train_rows,
                test_rows,
                names,
                args,
                row_hours=row_hours,
                feature_selection=True,
            )
        split["models"].update(
            _late_fusion_models(
                train_rows,
                test_rows,
                groups,
                args,
                row_hours=row_hours,
            )
        )
        split_reports.append(split)

    aggregate: dict[str, dict[str, float | None]] = {}
    metric_keys = (
        "pr_auc",
        "roc_auc",
        "brier",
        "precision",
        "recall",
        "false_alarms_per_hour",
        "event_recall",
        "lead_time_minutes_mean",
    )
    for model_name in model_order:
        model_rows = [
            split["models"][model_name]
            for split in split_reports
            if model_name in split["models"]
        ]
        if not model_rows:
            continue
        aggregate[model_name] = {
            key: _mean_metric(model_rows, key)
            for key in metric_keys
        }
        aggregate[model_name]["feature_count"] = _mean_metric(model_rows, "feature_count")

    def _delta(model_name: str, baseline_name: str, key: str) -> float | None:
        model_row = aggregate.get(model_name, {})
        baseline_row = aggregate.get(baseline_name, {})
        if model_row.get(key) is None or baseline_row.get(key) is None:
            return None
        return float(model_row[key]) - float(baseline_row[key])

    scored_models = [
        (name, row)
        for name, row in aggregate.items()
        if row.get("pr_auc") is not None
    ]
    best_model = (
        max(scored_models, key=lambda item: float(item[1]["pr_auc"]))
        if scored_models
        else None
    )
    latent_candidates = [
        (name, row)
        for name, row in scored_models
        if "latent" in name or name.startswith("late_fusion")
    ]
    best_latent_model = (
        max(latent_candidates, key=lambda item: float(item[1]["pr_auc"]))
        if latent_candidates
        else None
    )
    comparison = {
        "baseline": "pedestrian_spectral",
        "full": "pedestrian_spectral_latent",
        "delta_pr_auc": _delta("pedestrian_spectral_latent", "pedestrian_spectral", "pr_auc"),
        "delta_event_recall": _delta(
            "pedestrian_spectral_latent",
            "pedestrian_spectral",
            "event_recall",
        ),
        "delta_recall": _delta("pedestrian_spectral_latent", "pedestrian_spectral", "recall"),
        "delta_false_alarms_per_hour": _delta(
            "pedestrian_spectral_latent",
            "pedestrian_spectral",
            "false_alarms_per_hour",
        ),
        "selected_baseline": "pedestrian_spectral_selected",
        "selected_full": "pedestrian_spectral_latent_selected",
        "selected_delta_pr_auc": _delta(
            "pedestrian_spectral_latent_selected",
            "pedestrian_spectral_selected",
            "pr_auc",
        ),
        "selected_delta_event_recall": _delta(
            "pedestrian_spectral_latent_selected",
            "pedestrian_spectral_selected",
            "event_recall",
        ),
        "late_fusion_mean_delta_pr_auc": _delta("late_fusion_mean", "pedestrian_spectral", "pr_auc"),
        "late_fusion_logistic_delta_pr_auc": _delta(
            "late_fusion_logistic",
            "pedestrian_spectral",
            "pr_auc",
        ),
        "best_model": (
            {"name": best_model[0], "pr_auc": best_model[1]["pr_auc"]}
            if best_model is not None
            else None
        ),
        "best_latent_model": (
            {"name": best_latent_model[0], "pr_auc": best_latent_model[1]["pr_auc"]}
            if best_latent_model is not None
            else None
        ),
    }
    return {
        "feature_names": feature_names,
        "feature_groups": {name: len(values) for name, values in groups.items()},
        "ablations": {
            **{name: len(values) for name, values in ablations.items()},
            **{name: len(values) for name, values in selected_ablations.items()},
            "late_fusion_mean": len(groups),
            "late_fusion_logistic": len(groups),
        },
        "model_order": model_order,
        "splits": split_reports,
        "aggregate": aggregate,
        "comparison": comparison,
    }


def _phase_counts(rows: Sequence[FeatureRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.phase] = counts.get(row.phase, 0) + 1
    return dict(sorted(counts.items()))


def _subject_counts(rows: Sequence[FeatureRow]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for subject in sorted({row.subject for row in rows}):
        subject_rows = [row for row in rows if row.subject == subject]
        out[subject] = {
            "rows": len(subject_rows),
            "positives": sum(row.label for row in subject_rows),
            "files": len({row.file for row in subject_rows}),
            "events": len({row.event_id for row in subject_rows if row.event_id}),
        }
    return out


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    config = AutoTuneConfig(
        baseline_size=args.baseline,
        adaptive_window=args.adaptive_window,
        stride=args.stride,
        score=args.score,
        emission_threshold=args.emission_threshold,
        decay=args.decay,
        min_active_series=args.min_active_channels,
    )
    config.validate()
    started = time.time()
    rows, skipped = build_dataset(args, config)
    evaluation = evaluate(rows, args)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "dataset": {
            "name": "CHB-MIT Scalp EEG Database",
            "subjects": list(args.subjects),
            "data_root": str(args.data_root),
            "labels_used": "supervised predictor ablation only",
            "file_validation": "each EDF is parsed with load_edf_signals before use; parser failures are skipped",
        },
        "method": {
            "purpose": (
                "measure whether agnostic latent/stigmergic structure features improve a "
                "simple held-out-subject predictor over pedestrian and spectral features"
            ),
            "split": "leave-one-subject-out",
            "predictor": "median imputer + standard scaler + class-balanced logistic regression",
            "model_selection": (
                "regularization C and optional feature count selected by inner "
                "leave-one-subject-out on training subjects only"
            ),
            "integration_controls": (
                "selected-feature logistic models and late fusion over pedestrian/spectral/latent "
                "family probabilities are trained without outer-held-out labels"
            ),
            "positive_label": "anchor falls inside the preictal horizon before a labeled seizure",
            "negative_label": "anchor is outside seizure, postictal, and near-seizure exclusion gaps",
            "event_note": "events are file-local CHB-MIT summary seizures with at least one eligible preictal row",
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
            "feature_window_seconds": args.feature_window_seconds,
            "history_rows": args.history_rows,
            "preictal_minutes": args.preictal_minutes,
            "negative_gap_minutes": args.negative_gap_minutes,
            "postictal_minutes": args.postictal_minutes,
            "false_alarms_per_hour": args.false_alarms_per_hour,
            "regularization_cs": _parse_float_list(args.regularization_cs),
            "selection_ks": _parse_int_list(args.selection_ks),
            "fusion_meta_c": args.fusion_meta_c,
            "max_files_per_subject": args.max_files_per_subject,
            "seed": args.seed,
        },
        "rows": {
            "total": len(rows),
            "positive": sum(row.label for row in rows),
            "negative": len(rows) - sum(row.label for row in rows),
            "phase_counts": _phase_counts(rows),
            "subjects": _subject_counts(rows),
            "skipped": skipped,
        },
        "evaluation": evaluation,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_summary(report: Mapping[str, Any]) -> str:
    evaluation = report["evaluation"]
    aggregate = evaluation["aggregate"]
    comparison = evaluation["comparison"]
    lines = [
        "# CHB-MIT predictor ablation",
        "",
        "This is a supervised measurement harness, not a label-free discovery run. Labels train and evaluate the predictor; the feature layers remain generic.",
        "Logistic regularization and optional feature counts are selected by inner leave-one-subject-out on training subjects only.",
        "Late-fusion controls combine pedestrian/spectral/latent family probabilities without using outer-held-out labels.",
        "",
        "## Dataset",
        "",
        f"- Subjects: {', '.join(report['dataset']['subjects'])}",
        f"- Rows: {report['rows']['total']} ({report['rows']['positive']} positive, {report['rows']['negative']} negative)",
        f"- Phase counts: {report['rows']['phase_counts']}",
        f"- Skipped EDFs: {len(report['rows']['skipped'])}",
        "",
        "## Aggregate Leave-One-Subject-Out Metrics",
        "",
        "| Model | Features | PR-AUC | ROC-AUC | Brier | Recall | Event recall | False alarms/hr | Lead min |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in evaluation["model_order"]:
        if model_name not in aggregate:
            continue
        metrics = aggregate[model_name]
        lines.append(
            "| %s | %d | %s | %s | %s | %s | %s | %s | %s |"
            % (
                model_name,
                int(metrics["feature_count"]),
                _fmt(metrics["pr_auc"]),
                _fmt(metrics["roc_auc"]),
                _fmt(metrics["brier"]),
                _fmt(metrics["recall"]),
                _fmt(metrics["event_recall"]),
                _fmt(metrics["false_alarms_per_hour"]),
                _fmt(metrics["lead_time_minutes_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Latent Contribution",
            "",
            f"- Baseline: `{comparison['baseline']}`",
            f"- Full: `{comparison['full']}`",
            f"- Delta PR-AUC: {_fmt(comparison['delta_pr_auc'])}",
            f"- Delta event recall: {_fmt(comparison['delta_event_recall'])}",
            f"- Delta recall: {_fmt(comparison['delta_recall'])}",
            f"- Delta false alarms/hr: {_fmt(comparison['delta_false_alarms_per_hour'])}",
            f"- Selected baseline: `{comparison['selected_baseline']}`",
            f"- Selected full: `{comparison['selected_full']}`",
            f"- Selected delta PR-AUC: {_fmt(comparison['selected_delta_pr_auc'])}",
            f"- Selected delta event recall: {_fmt(comparison['selected_delta_event_recall'])}",
            f"- Late-fusion mean delta PR-AUC: {_fmt(comparison['late_fusion_mean_delta_pr_auc'])}",
            f"- Late-fusion logistic delta PR-AUC: {_fmt(comparison['late_fusion_logistic_delta_pr_auc'])}",
            f"- Best model: `{comparison['best_model']['name'] if comparison['best_model'] else 'n/a'}` PR-AUC {_fmt(comparison['best_model']['pr_auc'] if comparison['best_model'] else None)}",
            f"- Best latent-bearing model: `{comparison['best_latent_model']['name'] if comparison['best_latent_model'] else 'n/a'}` PR-AUC {_fmt(comparison['best_latent_model']['pr_auc'] if comparison['best_latent_model'] else None)}",
            "",
            "## Integration Readout",
            "",
            "- Raw full-stack integration remains worse than raw pedestrian+spectral on PR-AUC.",
            "- Train-only feature selection makes the full stack beat the selected pedestrian+spectral baseline, but the best model is still latent-only selected.",
            "- Late fusion improves PR-AUC over raw pedestrian+spectral, but the fixed false-alarm threshold can still suppress recall.",
            "",
            "## Split Details",
            "",
        "| Held-out subject | Model | PR-AUC | ROC-AUC | Recall | Event recall | False alarms/hr | Threshold |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in evaluation["splits"]:
        for model_name, metrics in split["models"].items():
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    split["test_subject"],
                    model_name,
                    _fmt(metrics["pr_auc"]),
                    _fmt(metrics["roc_auc"]),
                    _fmt(metrics["recall"]),
                    _fmt(metrics["event_recall"]),
                    _fmt(metrics["false_alarms_per_hour"]),
                    _fmt(metrics["threshold"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- This does not claim domain-general prediction. It asks whether latent structure contributes to a simple downstream predictor when labels exist.",
            "- Leave-one-subject-out prevents random-window leakage, but the cohort is still small.",
            "- CHB-MIT labels are file-local here; cross-file preictal continuity is not modeled.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/chbmit"))
    parser.add_argument("--subjects", nargs="+", default=["chb01", "chb02", "chb03"])
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--target-sample-rate", type=float, default=16.0)
    parser.add_argument("--baseline", type=int, default=1024)
    parser.add_argument("--adaptive-window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--score", choices=list(SCORE_NAMES), default="max")
    parser.add_argument("--emission-threshold", type=float, default=3.0)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--min-active-channels", type=int, default=3)
    parser.add_argument("--feature-window-seconds", type=float, default=120.0)
    parser.add_argument("--history-rows", type=int, default=6)
    parser.add_argument("--preictal-minutes", type=float, default=10.0)
    parser.add_argument("--negative-gap-minutes", type=float, default=30.0)
    parser.add_argument("--postictal-minutes", type=float, default=10.0)
    parser.add_argument("--false-alarms-per-hour", type=float, default=1.0)
    parser.add_argument(
        "--regularization-cs",
        default="0.01,0.03,0.1,0.3,1.0",
        help="Comma-separated logistic C values selected by training-subject inner CV",
    )
    parser.add_argument(
        "--selection-ks",
        default="8,16,32,64",
        help="Comma-separated feature counts for selected-feature logistic controls",
    )
    parser.add_argument(
        "--fusion-meta-c",
        type=float,
        default=1.0,
        help="Fixed logistic C for the late-fusion meta model",
    )
    parser.add_argument("--max-files-per-subject", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if args.target_sample_rate <= 0:
        parser.error("--target-sample-rate must be positive")
    if args.feature_window_seconds <= 0:
        parser.error("--feature-window-seconds must be positive")
    if args.history_rows < 1:
        parser.error("--history-rows must be >= 1")
    if args.preictal_minutes <= 0:
        parser.error("--preictal-minutes must be positive")
    if args.negative_gap_minutes < args.preictal_minutes:
        parser.error("--negative-gap-minutes must be >= --preictal-minutes")
    if args.postictal_minutes < 0:
        parser.error("--postictal-minutes must be >= 0")
    if args.false_alarms_per_hour < 0:
        parser.error("--false-alarms-per-hour must be >= 0")
    try:
        regularization_cs = _parse_float_list(args.regularization_cs)
    except ValueError as exc:
        parser.error(str(exc))
    if any(value <= 0 for value in regularization_cs):
        parser.error("--regularization-cs values must be positive")
    try:
        selection_ks = _parse_int_list(args.selection_ks)
    except ValueError as exc:
        parser.error(str(exc))
    if any(value <= 0 for value in selection_ks):
        parser.error("--selection-ks values must be positive")
    if args.fusion_meta_c <= 0:
        parser.error("--fusion-meta-c must be positive")
    if args.max_files_per_subject < 0:
        parser.error("--max-files-per-subject must be >= 0")
    return args


def print_report(report: Mapping[str, Any]) -> None:
    comparison = report["evaluation"]["comparison"]
    aggregate = report["evaluation"]["aggregate"]
    print("CHB-MIT predictor ablation")
    print(
        "  subjects=%s rows=%d positives=%d skipped=%d elapsed=%.1fs"
        % (
            ",".join(report["dataset"]["subjects"]),
            report["rows"]["total"],
            report["rows"]["positive"],
            len(report["rows"]["skipped"]),
            report["elapsed_seconds"],
        )
    )
    print("  feature groups=%s" % report["evaluation"]["feature_groups"])
    for model_name, metrics in aggregate.items():
        print(
            "  %-29s pr_auc=%s roc_auc=%s recall=%s event_recall=%s false_alarms/hr=%s"
            % (
                model_name,
                _fmt(metrics["pr_auc"]),
                _fmt(metrics["roc_auc"]),
                _fmt(metrics["recall"]),
                _fmt(metrics["event_recall"]),
                _fmt(metrics["false_alarms_per_hour"]),
            )
        )
    print(
        "  latent delta vs pedestrian_spectral: pr_auc=%s event_recall=%s recall=%s false_alarms/hr=%s"
        % (
            _fmt(comparison["delta_pr_auc"]),
            _fmt(comparison["delta_event_recall"]),
            _fmt(comparison["delta_recall"]),
            _fmt(comparison["delta_false_alarms_per_hour"]),
        )
    )
    print(
        "  selected full delta: pr_auc=%s event_recall=%s"
        % (
            _fmt(comparison["selected_delta_pr_auc"]),
            _fmt(comparison["selected_delta_event_recall"]),
        )
    )
    best_latent = comparison["best_latent_model"]
    if best_latent:
        print("  best latent-bearing model=%s pr_auc=%s" % (
            best_latent["name"],
            _fmt(best_latent["pr_auc"]),
        ))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_ablation(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(markdown_summary(report), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
