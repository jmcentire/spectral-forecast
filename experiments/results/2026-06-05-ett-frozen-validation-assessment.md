# ETT Frozen Generic-Observer Validation Assessment

## Question

Can the generic spectral/stigmergic observer find transferable multichannel
structure in a new numeric time-series domain, and does that aggregate
structure resolve into stable exact relationships?

No event labels or domain quantities were used.

## Calibration

The first 8,192 hourly ETTh1 observations were used for calibration only.

The initial fixed threshold grid (`2.5` through `4.0`) rejected every candidate.
The best candidate was saturated:

- accepted: no;
- quality: `-0.112`;
- saturation penalty: `0.949`; and
- block-null empirical p: `0.175`.

This does not show that ETT lacks structure. It shows that the fixed threshold
scale could not resolve it.

A second calibration pass derived numeric threshold candidates from
calibration-only observer-score quantiles. The selected configuration was:

- baseline: `1024`;
- adaptive window: `256`;
- stride: `128`;
- threshold: `13.6296957494`;
- decay: `0.75`; and
- minimum active series: `3`.

It cleared the calibration block-permutation null with `0/199` exceedances,
`p=0.005`, and low saturation (`0.048`). Its even/odd-series stability score
was `0.0`, which was an explicit warning that the aggregate might not be
carried by stable arbitrary channel subsets.

## Frozen Aggregate Validation

The exact selected configuration and numeric threshold were frozen. No
threshold or configuration was re-estimated on validation data.

| Segment | Null | Delta | z effect | Exceedances | Empirical p | Accepted |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| later ETTh1 | permutation | 1041.7 | 4.02 | 0/999 | 0.001 | yes |
| later ETTh1 | circular shift | 1029.1 | 2.78 | 3/999 | 0.004 | yes |
| later ETTh1 | block permutation | 1013.8 | 2.89 | 5/999 | 0.006 | yes |
| later ETTh2 | permutation | 4995.3 | 2.52 | 0/999 | 0.001 | yes |
| later ETTh2 | circular shift | 5119.9 | 2.02 | 0/999 | 0.001 | yes |
| later ETTh2 | block permutation | 5679.7 | 2.43 | 0/999 | 0.001 | yes |

The aggregate observer therefore detects cross-channel concurrent residual
timing that transfers across both time and asset under three timing controls.

## Exact Relationship Audit

The aggregate requires at least three active channels but does not preserve
which three channels carry an emission. To test relationship identity, every
exact three-channel subset was scored under the same frozen configuration.

Each segment tested all 35 triplets against 4,999 block-permutation nulls.
Benjamini-Yekutieli correction was applied within each segment.

| Segment | Corrected exact triplets | Strongest triplet | Strongest p | Strongest BY q |
| --- | ---: | --- | ---: | ---: |
| later ETTh1 | 0 | LUFL, LULL, OT | 0.0020 | 0.2903 |
| later ETTh2 | 1 | HULL, MULL, OT | 0.0002 floor | 0.0290 |

Exact triplets replicated across both segments: **0**.

## Interpretation

The positive aggregate result is real at the level tested: multiple channels
become unusually active together more often than the timing controls produce.

The experiment does not establish a stable reusable **exact-triplet**
relationship identity. The current aggregate stigmergic statistic
intentionally collapses component identity into active count and excess. It can
therefore detect concurrent distributed activity while failing to classify
which exact relationship carries it.

Zero exact-triplet replication does not rule out a coarser recurring structure,
such as stable roles, motifs, or interchangeable channel classes. Those require
different representations and controls.

This is:

- a positive validation of the generic aggregate observer;
- a positive validation of calibration-only quantile thresholds;
- a negative validation of stable exact-triplet discovery; and
- evidence that exact identity, structural equivalence, and relation change
  require explicit downstream layers rather than inference from aggregate
  coherence.

## Primary Artifacts

- `experiments/results/2026-06-05-etth1-calibration-block199.json`
- `experiments/results/2026-06-05-etth1-calibration-quantile-block199.json`
- `experiments/results/2026-06-05-etth1-heldout-frozen-*.json`
- `experiments/results/2026-06-05-etth2-late-external-frozen-*.json`
- `experiments/results/2026-06-05-ett-frozen-triplet-block4999.json`
- `experiments/csv_subset_audit.py`
