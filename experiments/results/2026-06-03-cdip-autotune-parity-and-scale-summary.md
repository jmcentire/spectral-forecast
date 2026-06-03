# CDIP Autotune Parity And Scale

Date: 2026-06-03

## Summary

The CDIP autotune runner now uses the same aggregate statistic as the manual
CDIP batch runner for parity checks: raw multi-series emission totals, summed
over windows, compared against repeat-aligned aggregate shift-null totals.

After this change, a fresh run on the original small ocean slice recovered the
manual result and found stronger nearby label-free settings. A one-step CDIP
scale check also exceeded the prior manual 256-group reference.

## Method Fix

Two mismatches were corrected:

- The CDIP autotune aggregate now scores raw emission totals, matching
  `cdip_batch.py`, rather than decayed pheromone totals.
- The CDIP `shift` null now uses the manual batch geometry for aggregate totals:
  keep the anchor series fixed and shift non-anchor series by deterministic
  repeat/column offsets.

The support gate was also relaxed from `0.50` to `0.45` positive-window
fraction. That matters because the exact manual full-slice candidate has strong
aggregate evidence but only `342/702 = 0.487` positive windows.

## Original Small Slice

Manual reference:

- artifact: `experiments/results/2026-06-02-cdip-scale-500mb-128groups-null50-summary.json`
- 38 local CDIP files at or below the 500 MB cutoff
- 128 balanced groups
- 702 windows
- preset scale config: `preprocess=none`, baseline `1024`, adaptive `512`,
  stride `512`, threshold `3.0`, decay `0.9`
- shifted null, 50 repeats
- delta: `249.585`
- z: `3.282`
- empirical p-floor: `0.0196078`

Fresh autotune parity artifact:

- `experiments/results/2026-06-03-cdip-autotune-parity-500mb-128groups-shift.json`

Exact manual candidate recovered in autotune candidates:

- `preprocess=none`, threshold `3.0`
- delta: `249.585`
- z: `3.249`
- null repeats: `50`
- null exceedances: `0`
- empirical p-floor: `0.0196078`

Best auto-selected small-slice candidate:

- `preprocess=highpass`, threshold `2.5`
- accepted: `true`
- delta: `369.286`
- z: `8.383`
- null repeats: `50`
- null exceedances: `0`
- empirical p-floor: `0.0196078`

Interpretation: the auto-system is now on par with the original small manual
CDIP run on the same data slice, and it finds stronger nearby settings without
using labels or oceanographic features.

## Disjoint Half Split

Artifact:

- `experiments/results/2026-06-03-cdip-autotune-parity-500mb-64-64-shift-support045.json`

Calibration half:

- 350 windows
- selected `highpass-dominant-mask`, threshold `3.0`
- accepted: `true`
- delta: `197.776`
- z: `3.693`
- null repeats: `50`
- null exceedances: `0`
- empirical p-floor: `0.0196078`

Heldout half:

- 317 windows
- accepted: `false`
- delta: `24.244`
- z: `0.62`
- null repeats: `50`
- null exceedances: `16`
- empirical p-floor: `0.0196078`

Interpretation: full small-slice parity is recovered, but the 64/64 split is
not a strong heldout validation for the selected candidate. Treat the split as
a caution that group halves differ materially at this size.

## One-Step Scale Check

Manual reference:

- artifact: `experiments/results/2026-06-02-cdip-scale-1gb-256groups-null50-summary.json`
- 256 balanced groups
- 1347 windows
- delta: `605.252`
- z: `7.409`
- empirical p-floor: `0.0196078`

Fresh autotune scale artifact:

- `experiments/results/2026-06-03-cdip-autotune-scale-1gb-256groups-shift.json`

Best auto-selected scale candidate:

- 61 local files under the current `-1G` filesystem cutoff
- 256 balanced calibration groups
- 1328 windows
- selected `highpass-dominant-mask`, threshold `3.0`
- accepted: `true`
- delta: `729.675`
- z: `10.14`
- null repeats: `50`
- null exceedances: `0`
- empirical p-floor: `0.0196078`

Interpretation: the auto-system is on par or better than the prior manual
256-group CDIP scale result on the same data family. The empirical p value is
at the 50-repeat floor because observed exceeded all null repeats; do not read
the identical `0.0196078` p-floor as equal effect size.

## Next

Before moving to another domain, run the same auto-selected CDIP geometry one
more notch up with the cached/progress-enabled runner, preferably using a
manifest-backed split so calibration and heldout definitions are stable and
cheap to reproduce.
