# CDIP Autotune Hardening

Date: 2026-06-03

## Summary

The autotune gate was hardened before any larger cross-domain runs:

- expensive observer score matrices are now cached and reusable,
- matrix-local nulls now include `permute`, `shift`, and `block-permute`,
- CDIP can run an external `phase-surrogate` null that rebuilds Fourier
  phase-randomized observations,
- CDIP validation can use disjoint heldout buoy groups instead of only the next
  windows from calibration groups.

The bounded CDIP rerun still rejected the selected candidate. That is the right
outcome for this gate: the machinery is stronger, but the current small CDIP
autotune search still has not earned promotion to larger EEG, PMU, BGP, or
seismic runs.

## Bounded Run

Artifact:

- `experiments/results/2026-06-03-cdip-autotune-hardened-bounded.json`

Setup:

- 12 local CDIP files.
- 4 calibration groups and 4 disjoint validation groups.
- 4 candidates: `highpass` and `highpass-dominant-mask`, thresholds `2.5` and
  `3.0`.
- 6 null repeats per null family.
- Null families: `permute`, `shift`, `phase-surrogate`.
- Persistent matrix cache: `/tmp/spectral-forecast-cdip-autotune-cache`.

The first run wrote 132 cached matrices. An immediate rerun loaded 132 matrices
from disk with 0 misses and 0 writes, reproducing the same result.

## Result

Best calibration candidate:

- preprocess: `highpass-dominant-mask`
- baseline: `768`
- adaptive window: `384`
- stride: `256`
- threshold: `3.0`
- decay: `0.9`
- accepted: `false`
- combined quality: `0.2258`
- combined delta: `73.636`
- combined z: `1.14`
- accepted windows: `3/4`
- worst null family: `permute`

Calibration by null family:

- `permute`: accepted `false`, delta `73.636`, z `1.14`
- `shift`: accepted `true`, delta `109.371`, z `2.07`
- `phase-surrogate`: accepted `true`, delta `195.695`, z `2.88`

Disjoint heldout validation:

- accepted: `false`
- combined quality: `0.0186`
- combined delta: `-61.580`
- combined z: `-0.81`
- accepted windows: `1/4`
- worst null family: `permute`

Heldout by null family:

- `permute`: accepted `false`, delta `1.982`, z `0.03`
- `shift`: accepted `false`, delta `1.177`, z `0.02`
- `phase-surrogate`: accepted `false`, delta `-61.580`, z `-0.81`

## Interpretation

Caching works and the stricter null stack works. The selected small-grid CDIP
candidate does not validate on disjoint groups. The next useful step is not to
scale other domains yet; it is to run a broader CDIP autotune search with the
same hard gate and cached matrices, then require the selected candidate to
remain accepted on disjoint heldout groups.
