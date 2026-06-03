# Autotune Method

Date: 2026-06-03

## Goal

The CDIP work used manually chosen observation settings. The next requirement is
automatic tuning without labels, post-hoc target leakage, or domain-specific
positive examples.

The tuning objective is:

```text
maximize reproducible structure above a domain-preserving null,
while penalizing saturation, threshold fragility, and insufficient information
```

This is not ordinary supervised hyperparameter tuning. The tuner is not trying
to match rogue waves, seizures, outages, or any known positive label. It is
trying to find settings under which the system surfaces stable, nontrivial
cross-element structure that is not produced by the null.

## Implemented Core

New module:

- `spectral_forecast/autotune.py`

Main API:

- `AutoTuneConfig`
- `score_autotune_config()`
- `tune_observation()`
- `default_autotune_configs()`

Experiment CLI:

- `experiments/autotune_observe.py`

The CLI currently supports numeric CSV columns and bounded sample windows:

```bash
python3 experiments/autotune_observe.py data/ETTh1.csv --columns HUFL HULL MUFL MULL \
  --sample-limit 512 \
  --baselines 96,128 \
  --adaptive-windows 48,64 \
  --strides 24 \
  --thresholds 2.5,3.0 \
  --decays 0.75 \
  --min-active-series 2 \
  --null-repeats 12 \
  --top 4 \
  --output experiments/results/2026-06-03-ett-autotune-smoke.json
```

## Scoring Terms

Each candidate is scored with:

- `readiness_score`: finite-window information readiness from entropy deficit,
  peak surprise, and usable spectral bins.
- `null_lift_score`: observed decayed multi-series emission above a timing
  permutation null that preserves each series' score distribution.
- `stability_score`: overlap between top windows discovered from even and odd
  series splits.
- `compression_score`: whether active cross-series patterns repeat instead of
  fragmenting into one-off noise.
- `residual_activity_score`: whether there is enough activity to inspect.
- `saturation_penalty`: penalizes settings where too many windows or cells are
  active.
- `fragility_penalty`: penalizes settings whose top windows collapse under
  small threshold changes.

The important design change from the first draft is that null lift is a gate,
not just another additive term. A saturated config with high activity but no
observed-vs-null separation should not rank as good.

## Accepted Config Gate

A config is marked `accepted=true` only when:

- quality is positive,
- null lift is positive, and
- saturation is not maximal.

This means the tuner can return a "best among bad options" while still saying
the run found no acceptable automatic setting.

## First Smoke Result

The bounded ETT smoke intentionally produced no acceptable config:

- best quality: `-0.2712`
- accepted: `false`
- observed total: `822.282`
- null mean: `850.294`
- z effect: `-2.39`
- exceedances: `12/12`
- saturation penalty: `0.904`

That is a useful result. The tuner did not treat abundant activity as signal
when the timing null produced equal or stronger structure.

## Next Work

1. Add a domain adapter so CDIP grouped windows can call the same autotune core.
2. Add cached observation matrices so large candidate grids do not refit the
   spectral observer for every threshold/decay/min-active combination.
3. Add domain-specific null families: timing permutation, phase surrogate,
   spatial/sensor shuffle, known-component masking.
4. Add an expansion rule: increase data until readiness and stability converge,
   then stop expanding unless the accepted score remains unstable.
