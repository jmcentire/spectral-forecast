# Observation Experiments

This directory is for bounded observational runs, not claim validation.

The CDIP runner follows a deliberately narrow protocol:

1. Load raw CDIP displacement samples.
2. Apply generic data-quality masking only: non-finite values, fill values, and excluded primary flags.
3. Select a contiguous clean segment and keep channel alignment intact.
4. Run `spectral_forecast.observe_series` with past-only frozen and adaptive baselines.
5. Emit only detector-state terms into the optional Stigmergy mesh.

The mesh input does not include rogue-wave labels or published ocean-wave predictors.
Raw score values, sample indexes, and timestamps are retained as metadata for audit,
but the routed signal content is built from the observer's own output vocabulary:
`frozen-*`, `sliding-*`, `drift-*`, `state-*`, residual direction, station, and channel.

Example:

```bash
python3 experiments/cdip_observe.py data/cdip/045p1_xy.nc \
  --channels z x y \
  --baseline 4096 \
  --adaptive-window 2048 \
  --stride 512 \
  --sample-limit 32768 \
  --mesh
```
