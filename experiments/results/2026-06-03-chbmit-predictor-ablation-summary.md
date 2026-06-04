# CHB-MIT predictor ablation

This is a supervised measurement harness, not a label-free discovery run. Labels train and evaluate the predictor; the feature layers remain generic.
Logistic regularization is selected by inner leave-one-subject-out on training subjects only.

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

## Latent Contribution

- Baseline: `pedestrian_spectral`
- Full: `pedestrian_spectral_latent`
- Delta PR-AUC: -0.010
- Delta event recall: -0.048
- Delta recall: 0.005
- Delta false alarms/hr: 12.143

## Split Details

| Held-out subject | Model | PR-AUC | ROC-AUC | Recall | Event recall | False alarms/hr | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chb01 | pedestrian | 0.073 | 0.219 | 0.000 | 0.000 | 0.648 | 0.991 |
| chb01 | pedestrian_spectral | 0.074 | 0.227 | 0.000 | 0.000 | 0.130 | 0.985 |
| chb01 | latent_only | 0.139 | 0.563 | 0.000 | 0.000 | 0.130 | 0.974 |
| chb01 | spectral_latent | 0.146 | 0.566 | 0.000 | 0.000 | 0.130 | 0.976 |
| chb01 | pedestrian_spectral_latent | 0.073 | 0.219 | 0.000 | 0.000 | 0.389 | 0.979 |
| chb02 | pedestrian | 0.032 | 0.424 | 0.000 | 0.000 | 0.884 | 0.978 |
| chb02 | pedestrian_spectral | 0.033 | 0.320 | 0.040 | 0.333 | 1.680 | 0.979 |
| chb02 | latent_only | 0.219 | 0.671 | 0.413 | 0.333 | 6.633 | 0.847 |
| chb02 | spectral_latent | 0.254 | 0.637 | 0.400 | 0.333 | 6.633 | 0.861 |
| chb02 | pedestrian_spectral_latent | 0.026 | 0.404 | 0.013 | 0.333 | 6.899 | 0.969 |
| chb03 | pedestrian | 0.238 | 0.544 | 0.182 | 0.429 | 7.583 | 0.980 |
| chb03 | pedestrian_spectral | 0.132 | 0.491 | 0.200 | 0.571 | 32.569 | 0.980 |
| chb03 | latent_only | 0.086 | 0.382 | 0.027 | 0.286 | 16.160 | 0.955 |
| chb03 | spectral_latent | 0.092 | 0.396 | 0.145 | 0.571 | 30.580 | 0.927 |
| chb03 | pedestrian_spectral_latent | 0.109 | 0.366 | 0.241 | 0.429 | 63.522 | 0.940 |

## Interpretation Guardrails

- This does not claim domain-general prediction. It asks whether latent structure contributes to a simple downstream predictor when labels exist.
- Leave-one-subject-out prevents random-window leakage, but the cohort is still small.
- CHB-MIT labels are file-local here; cross-file preictal continuity is not modeled.
