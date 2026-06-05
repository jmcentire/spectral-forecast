# CHB-MIT predictor ablation

This is a supervised measurement harness, not a label-free discovery run. Labels train and evaluate the predictor; the feature layers remain generic.
Logistic regularization and optional feature counts are selected by inner leave-one-subject-out on training subjects only.
Late-fusion controls combine pedestrian/spectral/latent family probabilities without using outer-held-out labels.

## Dataset

- Subjects: chb01, chb02, chb03
- Rows: 6626 (535 positive, 6091 negative)
- Phase counts: {'interictal': 6091, 'preictal': 535}
- Skipped EDFs: 0

## Aggregate Leave-One-Subject-Out Metrics

| Model | Features | PR-AUC | ROC-AUC | Brier | Recall | Event recall | False alarms/hr | Lead min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pedestrian | 79 | 0.115 | 0.396 | 0.242 | 0.061 | 0.143 | 3.038 | 7.528 |
| pedestrian_spectral | 134 | 0.080 | 0.346 | 0.291 | 0.080 | 0.302 | 11.460 | 3.419 |
| latent_only | 66 | 0.148 | 0.539 | 0.298 | 0.147 | 0.206 | 7.641 | 7.987 |
| spectral_latent | 121 | 0.164 | 0.533 | 0.293 | 0.182 | 0.302 | 12.448 | 8.219 |
| pedestrian_spectral_latent | 200 | 0.070 | 0.330 | 0.327 | 0.085 | 0.254 | 23.603 | 5.661 |
| pedestrian_spectral_selected | 134 | 0.072 | 0.322 | 0.363 | 0.059 | 0.190 | 8.403 | 5.504 |
| latent_only_selected | 66 | 0.178 | 0.595 | 0.295 | 0.065 | 0.270 | 1.409 | 7.071 |
| spectral_latent_selected | 121 | 0.158 | 0.618 | 0.296 | 0.061 | 0.159 | 1.977 | 7.950 |
| pedestrian_spectral_latent_selected | 200 | 0.105 | 0.306 | 0.361 | 0.116 | 0.254 | 21.333 | 6.619 |
| late_fusion_mean | 3 | 0.116 | 0.557 | 0.222 | 0.024 | 0.143 | 7.873 | 5.378 |
| late_fusion_logistic | 3 | 0.106 | 0.483 | 0.298 | 0.000 | 0.000 | 0.000 | n/a |

## Latent Contribution

- Baseline: `pedestrian_spectral`
- Full: `pedestrian_spectral_latent`
- Delta PR-AUC: -0.010
- Delta event recall: -0.048
- Delta recall: 0.005
- Delta false alarms/hr: 12.143
- Selected baseline: `pedestrian_spectral_selected`
- Selected full: `pedestrian_spectral_latent_selected`
- Selected delta PR-AUC: 0.032
- Selected delta event recall: 0.063
- Late-fusion mean delta PR-AUC: 0.036
- Late-fusion logistic delta PR-AUC: 0.026
- Best model: `latent_only_selected` PR-AUC 0.178
- Best latent-bearing model: `latent_only_selected` PR-AUC 0.178

## Fixed-Prediction Temporal Shift Audit

Within-file label shifts preserve file membership and prevalence while breaking predictor timing.

| Test | Observed | Null mean | Delta | Empirical p | BY q |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_only | 0.148 | 0.132 | 0.016 | 0.0960 | 0.7150 |
| latent_only_selected | 0.178 | 0.143 | 0.035 | 0.0910 | 0.7150 |
| pedestrian | 0.115 | 0.120 | -0.005 | 0.6460 | 1.0000 |
| pedestrian_spectral | 0.080 | 0.079 | 0.001 | 0.4390 | 1.0000 |
| pedestrian_spectral_latent | 0.070 | 0.070 | -0.000 | 0.5210 | 1.0000 |
| pedestrian_spectral_latent_selected | 0.105 | 0.088 | 0.016 | 0.2510 | 1.0000 |
| pedestrian_spectral_selected | 0.072 | 0.083 | -0.011 | 0.9980 | 1.0000 |
| spectral_latent | 0.164 | 0.134 | 0.030 | 0.0950 | 0.7150 |
| spectral_latent_selected | 0.158 | 0.133 | 0.026 | 0.0680 | 0.7150 |
| raw_latent_addition | -0.010 | -0.009 | -0.001 | 0.5860 | 1.0000 |
| selected_latent_addition | 0.032 | 0.005 | 0.027 | 0.0490 | 0.7150 |
| latent_only_vs_baseline | 0.069 | 0.053 | 0.015 | 0.1330 | 0.8255 |

## Integration Readout

- Raw full-stack integration remains worse than raw pedestrian+spectral on PR-AUC.
- Train-only feature selection makes the full stack beat the selected pedestrian+spectral baseline, but the best model is still latent-only selected.
- Late fusion improves PR-AUC over raw pedestrian+spectral, but the fixed false-alarm threshold can still suppress recall.

## Split Details

| Held-out subject | Model | PR-AUC | ROC-AUC | Recall | Event recall | False alarms/hr | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chb01 | pedestrian | 0.073 | 0.219 | 0.000 | 0.000 | 0.648 | 0.991 |
| chb01 | pedestrian_spectral | 0.074 | 0.227 | 0.000 | 0.000 | 0.130 | 0.985 |
| chb01 | latent_only | 0.139 | 0.563 | 0.000 | 0.000 | 0.130 | 0.974 |
| chb01 | spectral_latent | 0.146 | 0.566 | 0.000 | 0.000 | 0.130 | 0.976 |
| chb01 | pedestrian_spectral_latent | 0.073 | 0.219 | 0.000 | 0.000 | 0.389 | 0.979 |
| chb01 | pedestrian_spectral_selected | 0.071 | 0.186 | 0.000 | 0.000 | 0.259 | 0.973 |
| chb01 | latent_only_selected | 0.134 | 0.542 | 0.000 | 0.000 | 0.000 | 0.982 |
| chb01 | spectral_latent_selected | 0.145 | 0.557 | 0.000 | 0.000 | 0.000 | 0.969 |
| chb01 | pedestrian_spectral_latent_selected | 0.070 | 0.170 | 0.000 | 0.000 | 0.389 | 0.971 |
| chb01 | late_fusion_mean | 0.099 | 0.431 | 0.000 | 0.000 | 0.000 | 0.952 |
| chb01 | late_fusion_logistic | 0.153 | 0.552 | 0.000 | 0.000 | 0.000 | 0.736 |
| chb02 | pedestrian | 0.032 | 0.424 | 0.000 | 0.000 | 0.884 | 0.978 |
| chb02 | pedestrian_spectral | 0.033 | 0.320 | 0.040 | 0.333 | 1.680 | 0.979 |
| chb02 | latent_only | 0.219 | 0.671 | 0.413 | 0.333 | 6.633 | 0.847 |
| chb02 | spectral_latent | 0.254 | 0.637 | 0.400 | 0.333 | 6.633 | 0.861 |
| chb02 | pedestrian_spectral_latent | 0.026 | 0.404 | 0.013 | 0.333 | 6.899 | 0.969 |
| chb02 | pedestrian_spectral_selected | 0.021 | 0.294 | 0.000 | 0.000 | 0.088 | 0.991 |
| chb02 | latent_only_selected | 0.199 | 0.715 | 0.027 | 0.667 | 0.000 | 0.731 |
| chb02 | spectral_latent_selected | 0.156 | 0.769 | 0.013 | 0.333 | 0.088 | 0.760 |
| chb02 | pedestrian_spectral_latent_selected | 0.135 | 0.381 | 0.107 | 0.333 | 0.088 | 0.992 |
| chb02 | late_fusion_mean | 0.143 | 0.743 | 0.000 | 0.000 | 0.000 | 0.974 |
| chb02 | late_fusion_logistic | 0.068 | 0.471 | 0.000 | 0.000 | 0.000 | 0.947 |
| chb03 | pedestrian | 0.238 | 0.544 | 0.182 | 0.429 | 7.583 | 0.980 |
| chb03 | pedestrian_spectral | 0.132 | 0.491 | 0.200 | 0.571 | 32.569 | 0.980 |
| chb03 | latent_only | 0.086 | 0.382 | 0.027 | 0.286 | 16.160 | 0.955 |
| chb03 | spectral_latent | 0.092 | 0.396 | 0.145 | 0.571 | 30.580 | 0.927 |
| chb03 | pedestrian_spectral_latent | 0.109 | 0.366 | 0.241 | 0.429 | 63.522 | 0.940 |
| chb03 | pedestrian_spectral_selected | 0.125 | 0.486 | 0.177 | 0.571 | 24.862 | 0.997 |
| chb03 | latent_only_selected | 0.202 | 0.528 | 0.168 | 0.143 | 4.227 | 0.975 |
| chb03 | spectral_latent_selected | 0.173 | 0.527 | 0.168 | 0.143 | 5.843 | 0.977 |
| chb03 | pedestrian_spectral_latent_selected | 0.109 | 0.366 | 0.241 | 0.429 | 63.522 | 0.940 |
| chb03 | late_fusion_mean | 0.106 | 0.496 | 0.073 | 0.429 | 23.619 | 0.802 |
| chb03 | late_fusion_logistic | 0.097 | 0.427 | 0.000 | 0.000 | 0.000 | 0.747 |

## Interpretation Guardrails

- This does not claim domain-general prediction. It asks whether latent structure contributes to a simple downstream predictor when labels exist.
- Leave-one-subject-out prevents random-window leakage, but the cohort is still small.
- CHB-MIT labels are file-local here; cross-file preictal continuity is not modeled.
