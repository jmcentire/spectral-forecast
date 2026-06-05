# CHB-MIT predictor ablation

This is a supervised measurement harness, not a label-free discovery run. Labels train and evaluate the predictor; the feature layers remain generic.
Logistic regularization and optional feature counts are selected by inner leave-one-subject-out on training subjects only.
Late-fusion controls combine pedestrian, spectral, latent, and relationship family probabilities without using outer-held-out labels.

## Dataset

- Subjects: chb01, chb02, chb03
- Rows: 6384 (507 positive, 5877 negative)
- Phase counts: {'interictal': 5877, 'preictal': 507}
- Skipped EDFs: 0

## Aggregate Leave-One-Subject-Out Metrics

| Model | Features | PR-AUC | ROC-AUC | Brier | Recall | Event recall | False alarms/hr | Lead min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pedestrian | 79 | 0.177 | 0.468 | 0.270 | 0.167 | 0.476 | 11.362 | 7.088 |
| pedestrian_spectral | 134 | 0.146 | 0.414 | 0.337 | 0.145 | 0.405 | 15.332 | 5.463 |
| latent_only | 66 | 0.128 | 0.481 | 0.331 | 0.090 | 0.310 | 16.074 | 5.278 |
| relationship_only | 40 | 0.117 | 0.537 | 0.251 | 0.045 | 0.429 | 3.623 | 5.000 |
| spectral_latent | 121 | 0.126 | 0.467 | 0.343 | 0.087 | 0.190 | 26.695 | 3.821 |
| pedestrian_spectral_latent | 200 | 0.144 | 0.444 | 0.367 | 0.228 | 0.452 | 28.540 | 6.056 |
| pedestrian_spectral_relationship | 174 | 0.193 | 0.463 | 0.310 | 0.272 | 0.452 | 27.129 | 7.611 |
| pedestrian_spectral_latent_relationship | 240 | 0.188 | 0.470 | 0.319 | 0.325 | 0.452 | 36.055 | 8.100 |
| pedestrian_spectral_selected | 134 | 0.094 | 0.332 | 0.357 | 0.095 | 0.238 | 13.565 | 4.860 |
| latent_only_selected | 66 | 0.139 | 0.505 | 0.312 | 0.099 | 0.619 | 11.760 | 5.021 |
| relationship_only_selected | 40 | 0.074 | 0.441 | 0.249 | 0.018 | 0.262 | 3.231 | 3.012 |
| spectral_latent_selected | 121 | 0.124 | 0.512 | 0.328 | 0.100 | 0.524 | 14.217 | 4.865 |
| pedestrian_spectral_latent_selected | 200 | 0.091 | 0.413 | 0.354 | 0.103 | 0.238 | 16.521 | 5.653 |
| pedestrian_spectral_relationship_selected | 174 | 0.074 | 0.402 | 0.374 | 0.083 | 0.286 | 14.781 | 4.711 |
| pedestrian_spectral_latent_relationship_selected | 240 | 0.087 | 0.416 | 0.373 | 0.103 | 0.238 | 17.547 | 5.653 |
| late_fusion_mean | 4 | 0.088 | 0.493 | 0.241 | 0.022 | 0.143 | 7.246 | 2.983 |
| late_fusion_logistic | 4 | 0.074 | 0.353 | 0.358 | 0.041 | 0.429 | 19.441 | 3.650 |

## Latent Contribution

- Baseline: `pedestrian_spectral`
- Full: `pedestrian_spectral_latent`
- Delta PR-AUC: -0.002
- Delta event recall: 0.048
- Delta recall: 0.083
- Delta false alarms/hr: 13.208
- Selected baseline: `pedestrian_spectral_selected`
- Selected full: `pedestrian_spectral_latent_selected`
- Selected delta PR-AUC: -0.004
- Selected delta event recall: 0.000
- Late-fusion mean delta PR-AUC: -0.058
- Late-fusion logistic delta PR-AUC: -0.072
- Best model: `pedestrian_spectral_relationship` PR-AUC 0.193
- Best latent-bearing model: `pedestrian_spectral_latent_relationship` PR-AUC 0.188

## Relationship Contribution

- Full: `pedestrian_spectral_relationship`
- Raw delta PR-AUC: 0.048
- Selected full: `pedestrian_spectral_relationship_selected`
- Selected delta PR-AUC: -0.020
- Best relationship-bearing model: `pedestrian_spectral_relationship` PR-AUC 0.193

## Fixed-Prediction Temporal Shift Audit

Within-file label shifts preserve file membership and prevalence while breaking predictor timing.

| Test | Observed | Null mean | Delta | Empirical p | BY q |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_only | 0.128 | 0.120 | 0.007 | 0.0810 | 1.0000 |
| latent_only_selected | 0.139 | 0.129 | 0.010 | 0.1730 | 1.0000 |
| pedestrian | 0.177 | 0.139 | 0.038 | 0.1070 | 1.0000 |
| pedestrian_spectral | 0.146 | 0.125 | 0.021 | 0.2330 | 1.0000 |
| pedestrian_spectral_latent | 0.144 | 0.116 | 0.028 | 0.1970 | 1.0000 |
| pedestrian_spectral_latent_relationship | 0.188 | 0.164 | 0.024 | 0.2520 | 1.0000 |
| pedestrian_spectral_latent_relationship_selected | 0.087 | 0.092 | -0.005 | 0.5330 | 1.0000 |
| pedestrian_spectral_latent_selected | 0.091 | 0.093 | -0.003 | 0.4550 | 1.0000 |
| pedestrian_spectral_relationship | 0.193 | 0.169 | 0.025 | 0.2440 | 1.0000 |
| pedestrian_spectral_relationship_selected | 0.074 | 0.081 | -0.006 | 0.8640 | 1.0000 |
| pedestrian_spectral_selected | 0.094 | 0.108 | -0.013 | 0.8510 | 1.0000 |
| relationship_only | 0.117 | 0.101 | 0.016 | 0.0650 | 1.0000 |
| relationship_only_selected | 0.074 | 0.079 | -0.004 | 0.8450 | 1.0000 |
| spectral_latent | 0.126 | 0.122 | 0.004 | 0.1210 | 1.0000 |
| spectral_latent_selected | 0.124 | 0.120 | 0.004 | 0.2850 | 1.0000 |
| raw_latent_addition | -0.002 | -0.009 | 0.007 | 0.2530 | 1.0000 |
| selected_latent_addition | -0.004 | -0.014 | 0.010 | 0.0980 | 1.0000 |
| latent_only_vs_baseline | -0.018 | -0.005 | -0.013 | 0.7040 | 1.0000 |
| raw_relationship_addition | 0.048 | 0.044 | 0.004 | 0.4200 | 1.0000 |
| selected_relationship_addition | -0.020 | -0.027 | 0.007 | 0.2560 | 1.0000 |

## Integration Readout

- Raw latent addition delta PR-AUC: -0.002.
- Raw relationship addition delta PR-AUC: 0.048.
- Selected relationship addition delta PR-AUC: -0.020.
- Treat raw rankings as descriptive until the fixed-prediction temporal shift audit survives multiplicity correction.

## Split Details

| Held-out subject | Model | PR-AUC | ROC-AUC | Recall | Event recall | False alarms/hr | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chb01 | late_fusion_logistic | 0.078 | 0.272 | 0.000 | 0.000 | 2.275 | 0.758 |
| chb01 | late_fusion_mean | 0.084 | 0.328 | 0.000 | 0.000 | 1.472 | 0.749 |
| chb01 | latent_only | 0.103 | 0.420 | 0.000 | 0.000 | 0.268 | 0.984 |
| chb01 | latent_only_selected | 0.103 | 0.420 | 0.000 | 0.000 | 0.268 | 0.984 |
| chb01 | pedestrian | 0.072 | 0.199 | 0.000 | 0.000 | 9.637 | 0.972 |
| chb01 | pedestrian_spectral | 0.071 | 0.185 | 0.000 | 0.000 | 10.440 | 0.967 |
| chb01 | pedestrian_spectral_latent | 0.072 | 0.200 | 0.000 | 0.000 | 7.094 | 0.977 |
| chb01 | pedestrian_spectral_latent_relationship | 0.072 | 0.195 | 0.000 | 0.000 | 10.173 | 0.950 |
| chb01 | pedestrian_spectral_latent_relationship_selected | 0.072 | 0.195 | 0.000 | 0.000 | 10.173 | 0.950 |
| chb01 | pedestrian_spectral_latent_selected | 0.072 | 0.200 | 0.000 | 0.000 | 7.094 | 0.977 |
| chb01 | pedestrian_spectral_relationship | 0.071 | 0.186 | 0.000 | 0.000 | 12.716 | 0.958 |
| chb01 | pedestrian_spectral_relationship_selected | 0.071 | 0.186 | 0.000 | 0.000 | 12.716 | 0.958 |
| chb01 | pedestrian_spectral_selected | 0.071 | 0.180 | 0.000 | 0.000 | 8.165 | 0.984 |
| chb01 | relationship_only | 0.153 | 0.573 | 0.000 | 0.000 | 0.535 | 0.955 |
| chb01 | relationship_only_selected | 0.093 | 0.396 | 0.000 | 0.000 | 0.000 | 0.830 |
| chb01 | spectral_latent | 0.105 | 0.415 | 0.000 | 0.000 | 0.669 | 0.972 |
| chb01 | spectral_latent_selected | 0.098 | 0.417 | 0.000 | 0.000 | 0.000 | 0.952 |
| chb02 | late_fusion_logistic | 0.038 | 0.331 | 0.108 | 1.000 | 55.402 | 0.883 |
| chb02 | late_fusion_mean | 0.083 | 0.694 | 0.000 | 0.000 | 0.000 | 0.885 |
| chb02 | latent_only | 0.052 | 0.586 | 0.014 | 0.500 | 2.385 | 0.788 |
| chb02 | latent_only_selected | 0.089 | 0.592 | 0.041 | 1.000 | 1.192 | 0.794 |
| chb02 | pedestrian | 0.224 | 0.684 | 0.216 | 1.000 | 4.311 | 0.978 |
| chb02 | pedestrian_spectral | 0.176 | 0.592 | 0.149 | 0.500 | 3.027 | 0.944 |
| chb02 | pedestrian_spectral_latent | 0.176 | 0.596 | 0.149 | 0.500 | 2.752 | 0.941 |
| chb02 | pedestrian_spectral_latent_relationship | 0.240 | 0.605 | 0.405 | 0.500 | 18.345 | 0.982 |
| chb02 | pedestrian_spectral_latent_relationship_selected | 0.037 | 0.579 | 0.000 | 0.000 | 0.000 | 0.980 |
| chb02 | pedestrian_spectral_latent_selected | 0.048 | 0.564 | 0.000 | 0.000 | 0.000 | 0.978 |
| chb02 | pedestrian_spectral_relationship | 0.263 | 0.594 | 0.351 | 0.500 | 14.584 | 0.989 |
| chb02 | pedestrian_spectral_relationship_selected | 0.039 | 0.566 | 0.000 | 0.000 | 0.000 | 0.974 |
| chb02 | pedestrian_spectral_selected | 0.022 | 0.350 | 0.000 | 0.000 | 0.000 | 0.988 |
| chb02 | relationship_only | 0.098 | 0.536 | 0.095 | 1.000 | 2.201 | 0.879 |
| chb02 | relationship_only_selected | 0.029 | 0.425 | 0.014 | 0.500 | 1.559 | 0.695 |
| chb02 | spectral_latent | 0.037 | 0.556 | 0.000 | 0.000 | 1.834 | 0.893 |
| chb02 | spectral_latent_selected | 0.089 | 0.657 | 0.054 | 1.000 | 0.826 | 0.770 |
| chb03 | late_fusion_logistic | 0.106 | 0.457 | 0.015 | 0.286 | 0.645 | 0.866 |
| chb03 | late_fusion_mean | 0.097 | 0.458 | 0.065 | 0.429 | 20.267 | 0.756 |
| chb03 | latent_only | 0.227 | 0.436 | 0.255 | 0.429 | 45.568 | 0.949 |
| chb03 | latent_only_selected | 0.225 | 0.504 | 0.255 | 0.857 | 33.821 | 0.930 |
| chb03 | pedestrian | 0.235 | 0.520 | 0.285 | 0.429 | 20.138 | 0.975 |
| chb03 | pedestrian_spectral | 0.191 | 0.465 | 0.285 | 0.714 | 32.530 | 0.987 |
| chb03 | pedestrian_spectral_latent | 0.184 | 0.535 | 0.535 | 0.857 | 75.775 | 0.973 |
| chb03 | pedestrian_spectral_latent_relationship | 0.252 | 0.610 | 0.570 | 0.857 | 79.647 | 0.925 |
| chb03 | pedestrian_spectral_latent_relationship_selected | 0.152 | 0.475 | 0.310 | 0.714 | 42.470 | 0.993 |
| chb03 | pedestrian_spectral_latent_selected | 0.152 | 0.475 | 0.310 | 0.714 | 42.470 | 0.993 |
| chb03 | pedestrian_spectral_relationship | 0.246 | 0.608 | 0.465 | 0.857 | 54.088 | 0.940 |
| chb03 | pedestrian_spectral_relationship_selected | 0.113 | 0.454 | 0.250 | 0.857 | 31.627 | 0.985 |
| chb03 | pedestrian_spectral_selected | 0.191 | 0.465 | 0.285 | 0.714 | 32.530 | 0.987 |
| chb03 | relationship_only | 0.101 | 0.501 | 0.040 | 0.286 | 8.133 | 0.894 |
| chb03 | relationship_only_selected | 0.101 | 0.501 | 0.040 | 0.286 | 8.133 | 0.894 |
| chb03 | spectral_latent | 0.237 | 0.429 | 0.260 | 0.571 | 77.582 | 0.929 |
| chb03 | spectral_latent_selected | 0.186 | 0.461 | 0.245 | 0.571 | 41.824 | 0.956 |

## Interpretation Guardrails

- This does not claim domain-general prediction. It asks whether generic latent and relationship structure contributes to a simple downstream predictor when labels exist.
- Leave-one-subject-out prevents random-window leakage, but the cohort is still small.
- CHB-MIT labels are file-local here; cross-file preictal continuity is not modeled.
- Temporal label shifts test seizure-relative timing, not whether recurring relationship features reflect neurological state, acquisition context, or recording artifact.
