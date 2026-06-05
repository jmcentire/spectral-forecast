import numpy as np

from experiments.chbmit_predictor_ablation import (
    FeatureRow,
    _candidate_feature_counts,
    _classification_metrics,
    _late_fusion_models,
    _label_anchor,
    _latent_features,
    _temporal_shift_audit,
    _threshold_for_false_alarm_rate,
    parse_args,
)
from spectral_forecast.autotune import AutoTuneConfig


def test_label_anchor_keeps_only_preictal_and_far_interictal() -> None:
    seizures = ((100.0, 120.0),)

    positive = _label_anchor(
        "chbxx",
        "chbxx_01.edf",
        95.0,
        seizures,
        preictal_seconds=10.0,
        negative_gap_seconds=30.0,
        postictal_seconds=10.0,
    )
    ambiguous = _label_anchor(
        "chbxx",
        "chbxx_01.edf",
        80.0,
        seizures,
        preictal_seconds=10.0,
        negative_gap_seconds=30.0,
        postictal_seconds=10.0,
    )
    interictal = _label_anchor(
        "chbxx",
        "chbxx_01.edf",
        60.0,
        seizures,
        preictal_seconds=10.0,
        negative_gap_seconds=30.0,
        postictal_seconds=10.0,
    )
    ictal = _label_anchor(
        "chbxx",
        "chbxx_01.edf",
        110.0,
        seizures,
        preictal_seconds=10.0,
        negative_gap_seconds=30.0,
        postictal_seconds=10.0,
    )

    assert positive is not None
    assert positive.label == 1
    assert positive.event_id == "chbxx:chbxx_01.edf:0:100.000"
    assert ambiguous is None
    assert interictal is not None
    assert interictal.label == 0
    assert ictal is None


def test_threshold_for_false_alarm_rate_uses_training_negative_budget() -> None:
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.1, 0.95], dtype=np.float64)
    labels = np.asarray([0, 0, 0, 0, 1], dtype=np.int64)

    threshold = _threshold_for_false_alarm_rate(
        probabilities,
        labels,
        row_hours=0.5,
        false_alarms_per_hour=1.0,
    )

    assert threshold == 0.8
    assert int(np.sum((probabilities >= threshold) & (labels == 0))) == 2


def test_latent_features_capture_gated_emission_and_history() -> None:
    matrix = np.asarray(
        [
            [0.5, 0.2, 0.1],
            [3.5, 3.2, 0.1],
            [4.0, 3.8, 3.1],
        ],
        dtype=np.float64,
    )
    config = AutoTuneConfig(
        baseline_size=64,
        adaptive_window=32,
        stride=8,
        emission_threshold=3.0,
        decay=0.8,
        min_active_series=2,
    )

    features = _latent_features(matrix, 2, config, history_rows=3)

    assert features["latent_active_count"] == 3.0
    assert features["latent_gated_emission"] > 0.0
    assert features["latent_recent_pheromone"] > features["latent_gated_emission"]
    assert 0.0 <= features["latent_active_jaccard_prev"] <= 1.0


def test_classification_metrics_report_event_recall_and_false_alarm_rate() -> None:
    rows = [
        FeatureRow("s1", "f1", 1, 90.0, 1, "preictal", "evt", 100.0, {}),
        FeatureRow("s1", "f1", 2, 95.0, 1, "preictal", "evt", 100.0, {}),
        FeatureRow("s1", "f1", 3, 30.0, 0, "interictal", None, None, {}),
        FeatureRow("s1", "f1", 4, 40.0, 0, "interictal", None, None, {}),
    ]
    probabilities = np.asarray([0.2, 0.9, 0.8, 0.1], dtype=np.float64)

    metrics = _classification_metrics(rows, probabilities, 0.5, row_hours=0.25)

    assert metrics["event_recall"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["false_alarms_per_hour"] == 2.0


def test_candidate_feature_counts_caps_and_deduplicates() -> None:
    assert _candidate_feature_counts(20, "4,8,50") == [4, 8, 20]
    assert _candidate_feature_counts(3, "8,16") == [3]


def test_parse_args_exposes_temporal_null_repeats_once() -> None:
    assert parse_args([]).temporal_null_repeats == 999


def test_late_fusion_models_return_mean_and_logistic_controls() -> None:
    class Args:
        regularization_cs = "0.1"
        selection_ks = "2"
        false_alarms_per_hour = 100.0
        fusion_meta_c = 1.0
        max_iter = 500
        seed = 7

    rows = []
    for subject_index, subject in enumerate(("s1", "s2", "s3")):
        for i in range(20):
            label = int(i >= 10)
            value = float(i + subject_index * 0.1)
            rows.append(
                FeatureRow(
                    subject,
                    "f",
                    i,
                    float(i),
                    label,
                    "preictal" if label else "interictal",
                    f"{subject}:evt" if label else None,
                    20.0 if label else None,
                    {
                        "ped_a": value,
                        "spectral_a": value * 0.5,
                        "latent_a": value * 2.0,
                    },
                )
            )
    train_rows = [row for row in rows if row.subject != "s3"]
    test_rows = [row for row in rows if row.subject == "s3"]
    groups = {
        "pedestrian": ["ped_a"],
        "spectral": ["spectral_a"],
        "latent": ["latent_a"],
    }

    reports = _late_fusion_models(
        train_rows,
        test_rows,
        groups,
        Args(),
        row_hours=0.1,
    )

    assert set(reports) == {"late_fusion_mean", "late_fusion_logistic"}
    assert reports["late_fusion_mean"]["feature_count"] == 3.0
    assert reports["late_fusion_logistic"]["feature_count"] == 3.0
    assert reports["late_fusion_logistic"]["pr_auc"] is not None


def test_temporal_shift_audit_rejects_file_only_correlation() -> None:
    rows = [
        FeatureRow(
            "s1",
            "seizure_file" if index < 10 else "negative_file",
            index,
            float(index % 10),
            int(index in {8, 9}),
            "preictal" if index in {8, 9} else "interictal",
            "evt" if index in {8, 9} else None,
            10.0 if index in {8, 9} else None,
            {},
        )
        for index in range(20)
    ]
    file_only = np.asarray([0.9] * 10 + [0.1] * 10, dtype=np.float64)
    timing = np.asarray([0.1] * 8 + [0.9, 0.9] + [0.1] * 10, dtype=np.float64)

    audit = _temporal_shift_audit(
        [rows],
        [{"file_only": file_only, "timing": timing}],
        comparisons={"timing_delta": ("timing", "file_only")},
        null_repeats=999,
        seed=41,
    )

    models = {row["model"]: row for row in audit["models"]}
    comparison = audit["comparisons"][0]
    assert models["file_only"]["empirical_p_ge_observed"] == 1.0
    assert models["timing"]["empirical_p_ge_observed"] <= 0.05
    assert comparison["observed_minus_null"] > 0.0
    assert comparison["empirical_p_ge_observed"] <= 0.05
