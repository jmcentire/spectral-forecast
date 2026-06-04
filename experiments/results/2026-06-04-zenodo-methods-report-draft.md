# Latent Structure Discovery in Multichannel Time Series Using Spectral Residuals and Stigmergic Aggregation

Draft date: 2026-06-04

## Superseded Draft

Do not release this draft as written. The CDIP multi-buoy-coherence framing was
later invalidated by a trajectory-regroup control that preserves complete score
trajectories while breaking original buoy membership. Use
`2026-06-04-cross-domain-quality-assessment.md` and the domain validation-ladder
assessments as the current source of truth before preparing a replacement
methods report.

## Status

This is a draft methods report for a Zenodo-style artifact release. It is not
a prediction paper. Prediction-style labels are used only as downstream
information-content checks after latent structure has already been surfaced.

## Abstract

Many coupled systems fail, reorganize, or enter unusual operating regimes
without exposing a single decisive signal in any one component. We evaluate a
domain-agnostic observer that first models ordinary spectral behavior in each
series, then uses stigmergic aggregation to lift coherent residual structure
across components. The observer is not told domain labels, known predictors, or
event definitions. It produces candidate structure; domain-specific attribution
is performed afterward.

We report results across oceanographic buoy data, scalp EEG, and PMU-shaped
synthetic controls. In CDIP ocean data, the observer recovered coherent
multi-buoy structure that survived timing-permutation nulls, tide/high-pass
controls, dominant spectral-bin masking, held-out group splits, date controls,
and geography controls. Post-hoc spectral attribution indicates that the
surfaced structure is organized, longer-period, narrower-band sea-state
organization rather than confused or multi-modal sea. In CHB-MIT EEG, the same
label-free workflow surfaced cross-channel structure in three subjects; seizure
labels used only afterward showed seizure-phase enrichment. A separate
supervised ablation then tested whether the discovered feature layer encoded
usable information: selected latent/stigmergic features produced the best
aggregate PR-AUC among the tested models, while naive full feature
concatenation masked the signal. In pmuBAGE synthetic PMU data, the acceptance
gates rejected surfaces that failed transfer or support checks, demonstrating
useful negative controls.

The contribution is a methods artifact: a reproducible pipeline and result set
for label-agnostic latent structure discovery in noisy multicomponent systems.
It does not claim rogue-wave prediction, seizure prediction, grid-outage
prediction, or clinical/operational deployment readiness.

## Core Claim

The defensible claim is:

> A spectral residual observer with stigmergic cross-component aggregation can
> surface coherent latent structure in multichannel data without domain labels,
> and downstream ablations show that the surfaced latent features encode
> information not captured by pedestrian or direct spectral features alone.

The non-claim is equally important:

> This is not an event predictor. Labels are used for post-hoc attribution or
> supervised information-content tests, not as proof that the detector predicts
> a named event.

Operationally, "coherent" means that cross-component residual emissions exceed
domain-preserving nulls while satisfying support and saturation gates. It does
not mean that the method has inferred a causal mechanism. It means the observed
multi-component trace is stronger than the same aggregate statistic under
controls that preserve local score distributions or domain geometry while
breaking the candidate coordination structure.

"Stigmergic" is used in the engineering sense: individual components emit
local residual signals into a shared decayed trace, and later aggregate state is
read from that accumulated trace rather than from a centralized domain model.
The trace acts as the medium through which weak distributed emissions become a
candidate structure.

## Method

The pipeline has four layers.

1. Per-series spectral observation.
   Each component receives a past-only frozen nominal and a local adaptive
   nominal. The observer emits frozen residual scores, sliding residual scores,
   conditional drift, state drift, and decomposition-state summaries.

2. Stigmergic aggregation.
   Cross-component emissions are gated by score threshold and minimum active
   support. A decayed accumulation creates a pheromone-like trace of coherent
   residual activity. This is meant to lift distributed weak structure that may
   not be decisive in any single channel.

3. Label-free autotune.
   Candidate settings are selected by coherence above domain-preserving nulls
   while penalizing saturation, sparse support, and fragile threshold behavior.
   Labels are not used in the selection step.

4. Post-hoc attribution or supervised ablation.
   Domain quantities and labels are introduced only after structure is found.
   Attribution asks what the surfaced structure appears to be. Supervised
   ablation asks whether the surfaced feature layer carries information.

## Reporting Rules

- Report empirical p-values as Monte Carlo bounds with null repeat counts and
  p floors.
- Do not convert large z effects into normal-theory p-values.
- Report unique null totals where relevant.
- Keep aggregate lift separate from event/window support.
- Treat labels as post-hoc attribution unless a run explicitly uses labels.
- Keep negative controls in the report.

## Experiment 1: CDIP Ocean Calibration

The CDIP result is the known-physics calibration case. The observer was given
raw buoy displacement time series, not oceanographic quantities. It surfaced
coherent multi-buoy residual structure that survived several controls.

Representative masked reference runs:

| Run | Windows | Groups | Delta/window | z effect | Empirical p floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full 70-file masked reference | 5491 | 1024 | 0.5302 | 11.70 | 0.000999 |
| Heldout groups 512-1023 | 2722 | 512 | 0.4970 | 7.79 | 0.000999 |
| March 2026 date control | 245 | 58 | 0.8342 | 3.50 | 0.000999 |
| West Coast geography control | 434 | 67 | 0.9579 | 5.65 | 0.000999 |

Direct spectral attribution on heldout high-delta windows versus
month-matched baseline showed narrower and more concentrated spectra:

| Metric | High mean | Baseline mean | Difference | p |
| --- | ---: | ---: | ---: | ---: |
| Bandwidth Hz | 0.0765 | 0.0821 | -0.00564 | 0.012 |
| 90% energy width Hz | 0.2178 | 0.2388 | -0.02095 | 0.0248 |
| Hm0 estimate | 1.848 | 1.625 | +0.223 | 0.025 |
| Mean period s | 7.054 | 6.535 | +0.519 | 0.0296 |
| Peak count mean | 2.225 | 2.335 | -0.110 | 0.385 |
| Multimodal candidate mean | 0.572 | 0.601 | -0.029 | 0.428 |

Interpretation:

The detector recovered organized coherent sea-state structure from raw ocean
data without being told what swells, tides, spectra, or wave parameters are.
This is a calibration result, not a rogue-wave result.

Primary artifact:

- `experiments/results/2026-06-03-cdip-oceanographic-result.md`

## Experiment 2: CHB-MIT Label-Free EEG Structure

CHB-MIT tests a different data geometry: many channels on one physiological
system rather than many spatially separated buoys. The autotune workflow was
run independently on three subjects. Seizure labels were used only after the
label-free detector surfaced coherent windows.

Held-out validation summary:

| Subject | Files | Val accepted | Delta | z effect | p floor | Config | Ictal pos rate | Interictal pos rate |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| chb01 | 14 | yes | 937.3 | 15.73 | 0.0099 | 1024/256 t=2.5 min=3 | 15/16 = 0.938 | 159/1166 = 0.136 |
| chb02 | 14 | yes | 1127.9 | 4.98 | 0.0099 | 1024/256 t=3.0 min=3 | 5/6 = 0.833 | 69/1420 = 0.049 |
| chb03 | 14 | yes | 1526.6 | 8.27 | 0.0099 | 1024/256 t=3.0 min=3 | 10/14 = 0.714 | 522/1234 = 0.423 |

Interpretation:

The observer surfaces coherent cross-channel structure in a second domain, and
post-hoc labels reveal seizure-phase enrichment. This does not make the system
a seizure predictor. It shows that label-free structure is not ocean-specific.

Primary artifact:

- `experiments/results/2026-06-03-chbmit-autotune-cross-subject-summary.md`

## Experiment 3: CHB-MIT Information-Content Ablation

A separate supervised harness asks whether the latent/stigmergic features
encode information. This is intentionally not a deployment claim. It is an
information-content check: if the surfaced features are empty, they should not
help a simple downstream model on held-out subjects.

The harness used leave-one-subject-out splits across chb01, chb02, and chb03.
It produced 6,626 windows: 535 preictal positives and 6,091 interictal
negatives. Ictal, postictal, and near-seizure ambiguity gaps were excluded.
Regularization and optional feature counts were selected by inner
leave-one-subject-out on training subjects only.

Aggregate metrics:

| Model | Features | PR-AUC | ROC-AUC | Recall | Event recall | False alarms/hr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pedestrian | 79 | 0.115 | 0.396 | 0.061 | 0.143 | 3.038 |
| pedestrian_spectral | 134 | 0.080 | 0.346 | 0.080 | 0.302 | 11.460 |
| latent_only | 66 | 0.148 | 0.539 | 0.147 | 0.206 | 7.641 |
| spectral_latent | 121 | 0.164 | 0.533 | 0.182 | 0.302 | 12.448 |
| pedestrian_spectral_latent | 200 | 0.070 | 0.330 | 0.085 | 0.254 | 23.603 |
| pedestrian_spectral_selected | 134 | 0.072 | 0.322 | 0.059 | 0.190 | 8.403 |
| latent_only_selected | 66 | 0.178 | 0.595 | 0.065 | 0.270 | 1.409 |
| spectral_latent_selected | 121 | 0.158 | 0.618 | 0.061 | 0.159 | 1.977 |
| pedestrian_spectral_latent_selected | 200 | 0.105 | 0.306 | 0.116 | 0.254 | 21.333 |
| late_fusion_mean | 3 | 0.116 | 0.557 | 0.024 | 0.143 | 7.873 |
| late_fusion_logistic | 3 | 0.106 | 0.483 | 0.000 | 0.000 | 0.000 |

Interpretation:

Latent/stigmergic features contain usable information in this small ablation.
The best model is `latent_only_selected` by PR-AUC. Naive full feature
concatenation performs worse than pedestrian+spectral, which shows that
integration strategy matters. Train-only feature selection makes the full stack
beat the selected pedestrian+spectral baseline, but it still trails
latent-only selected. This supports a methods claim about latent information,
not a predictor claim.

Primary artifacts:

- `experiments/results/2026-06-04-chbmit-predictor-integration-summary.md`
- `experiments/chbmit_predictor_ablation.py`

## Experiment 4: PMU-Shaped Synthetic Controls

pmuBAGE is synthetic PMU-shaped data. It is not real grid evidence, but it is a
useful infrastructure-shaped control. The autotune gates rejected surfaces that
looked too saturated or too weakly supported under held-out validation.

| Surface | Events | Calibration | Validation | Read |
| --- | ---: | --- | --- | --- |
| frequency | 8 | accepted, delta 305.8, z 2.97, support 0.50, saturation 0.790 | rejected, delta -177.2, z -1.86, support 0.25, p 0.9505 | non-transfer |
| voltage magnitude | 20 | accepted, delta 3791.8, z 10.61, support 0.80, saturation 0.114 | rejected, delta 609.6, z 5.10, support 0.20, p floor 0.0099 | sparse support |

Interpretation:

The method can reject attractive but non-general surfaces. This is important
for infrastructure applications, where a single loud episode should not be
mistaken for generalized drift.

Primary artifact:

- `experiments/results/2026-06-03-pmubage-autotune-summary.md`

## Organizational And Leadership Expansion

The next replication queue should move beyond physical and physiological
signals into organizational theory and leadership behavior. The key is to
choose datasets with distributed behavior, not static survey tables.

Recommended five-dataset queue:

| Priority | Dataset | Organizational lens | First question |
| ---: | --- | --- | --- |
| 1 | SNAP `email-Eu-core-temporal` | Internal institutional email | Do communication-regime shifts surface across global/top-node flow? |
| 2 | SocioPatterns workplace contacts | Face-to-face workplace interaction | Do department coupling and meeting rhythms surface without labels? |
| 3 | Enron email corpus/core | Corporate crisis communication | Do known crisis-period communication structures surface post hoc? |
| 4 | GH Archive / curated open-source org data | Distributed maintainership and coordination | Do release crunches, bottlenecks, or contributor churn surface? |
| 5 | MAEC earnings calls | Executive/leadership communication | Do leadership communication regimes surface before market/risk labels? |

Primary artifact:

- `experiments/results/2026-06-04-organizational-leadership-dataset-scout.md`

## Limitations

- The current CHB-MIT run uses only three subjects.
- The supervised CHB-MIT ablation is an information-content test, not clinical
  evidence.
- The ocean attribution currently names a plausible structure, not a complete
  oceanographic mechanism.
- PMU evidence is synthetic only.
- Organizational/leadership datasets require new adapters, especially for
  text, communication networks, and open-source event streams.
- Static survey/HR datasets are not first-class tests for this method unless
  they can be converted into meaningful time-indexed or relational signals.

## Reproducibility Artifacts

Relevant code:

- `spectral_forecast/observation.py`
- `spectral_forecast/autotune.py`
- `experiments/cdip_batch.py`
- `experiments/cdip_sharded_run.py`
- `experiments/chbmit_autotune.py`
- `experiments/chbmit_predictor_ablation.py`
- `experiments/pmubage_autotune.py`

Relevant result summaries:

- `experiments/results/2026-06-03-cdip-oceanographic-result.md`
- `experiments/results/2026-06-03-cross-domain-structure-discovery-addendum.md`
- `experiments/results/2026-06-03-chbmit-autotune-cross-subject-summary.md`
- `experiments/results/2026-06-04-chbmit-predictor-integration-summary.md`
- `experiments/results/2026-06-03-pmubage-autotune-summary.md`
- `experiments/results/2026-06-04-organizational-leadership-dataset-scout.md`

## Recommended Zenodo Framing

Release this as:

> Version 0.1 technical methods artifact.

Use language like:

- "latent structure discovery"
- "label-agnostic observation"
- "post-hoc attribution"
- "information-content ablation"
- "organizational/leadership replication queue"

Avoid language like:

- "predict rogue waves"
- "predict seizures"
- "predict outages"
- "clinical result"
- "infrastructure warning system"
- "leadership predictor"
