# CDIP Autotune Validation

Date: 2026-06-03

## Summary

The label-free autotune mechanism was wired into bounded CDIP aligned buoy
windows and tested against heldout windows. The result is a useful rejection:
the current autotune grid did not validate on CDIP, so it should not be promoted
to larger CHB-MIT, PMU, BGP, or seismic runs yet.

This is the intended gate behavior. The tuner found calibration candidates with
some positive observed-vs-null lift, but the selected candidates did not clear
the stronger aggregate acceptance gate and failed heldout validation.

## Implementation

New runner:

- `experiments/cdip_autotune.py`

The runner:

1. Discovers balanced aligned CDIP buoy groups using the existing CDIP batch
   window machinery.
2. Scores candidate preprocessing and observation settings with the generic
   label-free autotune core.
3. Aggregates candidate quality over calibration windows.
4. Freezes the top candidate.
5. Scores the next aligned windows as heldout validation.

Labels, wave metrics, spectra attribution, and manual CDIP result knowledge are
not used in selection.

## Aggregate Gate

The first small run exposed a weak acceptance rule: a candidate could have small
positive aggregate quality and still fail heldout. The CDIP aggregate gate was
tightened so an accepted candidate must satisfy:

- positive aggregate quality,
- observed total greater than null mean total,
- accepted-window fraction at least `0.5`,
- aggregate z effect at least `1.5`,
- non-maximal saturation,
- positive mean null lift.

This prevents "one or two good windows" from promoting a candidate.

## Six-Group Run

Artifact:

- `experiments/results/2026-06-03-cdip-autotune-bounded-validation.json`

Setup:

- 12 local CDIP files.
- 8 candidates.
- 6 balanced calibration windows.
- 6 heldout validation windows.
- 12 timing-permutation null repeats per window.

Best calibration candidate:

- preprocess: `highpass-dominant-mask`
- baseline: `768`
- adaptive window: `384`
- stride: `256`
- threshold: `2.5`
- decay: `0.9`
- accepted: `false`
- calibration delta: `63.526`
- calibration z: `1.01`
- accepted windows: `2/6`

Heldout validation:

- accepted: `false`
- delta: `-47.843`
- z: `-0.49`
- accepted windows: `3/6`

## Twelve-Group Run

Artifact:

- `experiments/results/2026-06-03-cdip-autotune-12group-validation.json`

Setup:

- 12 local CDIP files.
- 24 candidates.
- 12 balanced calibration windows.
- 12 heldout validation windows.
- 12 timing-permutation null repeats per window.

Best calibration candidate:

- preprocess: `highpass-dominant-mask`
- baseline: `1024`
- adaptive window: `384`
- stride: `256`
- threshold: `3.0`
- decay: `0.9`
- accepted: `false`
- calibration delta: `40.493`
- calibration z: `0.62`
- accepted windows: `5/12`

Heldout validation:

- accepted: `false`
- delta: `-186.755`
- z: `-2.12`
- accepted windows: `2/12`

## Interpretation

Autotune is implemented, but not validated.

The current CDIP autotune grid can find local calibration lift, especially in
the highpass-dominant-mask family, but that lift is weak and does not survive
heldout windows. That means this version should not drive larger cross-domain
experiments yet.

The manual CDIP scale result remains valid as a separate manually fixed
experiment. This validation only says the new automatic tuning mechanism has
not yet reproduced that reliability from bounded CDIP data.

## Next Step

Improve the autotune mechanism before scaling:

1. Cache observation score matrices so thresholds, decay, and min-active sweeps
   do not refit the spectral observer repeatedly.
2. Tune at the group/window-manifest level with more calibration windows, then
   validate against disjoint groups rather than only next windows.
3. Add alternative null families from the CDIP controls: phase surrogate,
   spatial/sensor shuffle, and known-component masking.
4. Promote to larger cross-domain runs only after CDIP autotune produces an
   accepted calibration candidate that also remains accepted on heldout CDIP.
