# Observation Experiments

This directory is for bounded observational runs, not claim validation.

The CDIP runner follows a deliberately narrow protocol:

1. Load raw CDIP displacement samples.
2. Apply generic data-quality masking only: non-finite values, fill values, and excluded primary flags.
3. Select a contiguous clean segment and keep channel alignment intact.
4. Estimate finite-window information readiness from spectral entropy gap,
   peak surprise, and a minimum usable-bin gate.
5. Run `spectral_forecast.observe_series` with past-only frozen and adaptive baselines.
6. Emit only detector-state terms into the optional Stigmergy mesh.

Readiness answers a narrow question: when does the prefix have enough
concentrated structure above finite-sample noise to begin spectral extraction?
It does not claim the window is long enough to establish a stable anomaly
baseline.

The mesh input does not include rogue-wave labels or published ocean-wave predictors.
Raw score values, sample indexes, and timestamps are retained as metadata for audit,
but the routed signal content is built from the observer's own output vocabulary:
`frozen-*`, `sliding-*`, `drift-*`, `state-*`, residual direction, station, and channel.

Post-hoc research metrics can be reported for comparison only. These metrics
are computed after observations are emitted and are not available to the
observer or mesh.

The optional mesh uses neutral series priming by default. This gives the ART
mesh initial station/channel categories so it does not begin as several empty,
indistinguishable workers.

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

Multiple CDIP files can be passed in one run. The runner selects a common
clean UTC interval and resamples onto the lowest participating sample rate:

```bash
python3 experiments/cdip_observe.py \
  data/cdip/045p1_xy.nc \
  data/cdip/196p1_xy.nc \
  --channels z \
  --baseline 4096 \
  --adaptive-window 2048 \
  --stride 512 \
  --sample-limit 32768 \
  --mesh
```

For broader observation, use the batch runner. It discovers aligned buoy
groups, runs multiple clean windows, and compares multi-buoy co-emission
against a shifted-index null:

```bash
python3 experiments/cdip_batch.py data/cdip/*p1_xy.nc \
  --channels z \
  --group-size 3 \
  --max-groups 8 \
  --max-windows-per-group 3 \
  --baseline 1024 \
  --adaptive-window 512 \
  --stride 512 \
  --window-samples 4096 \
  --window-step 2048 \
  --null-repeats 5
```

If observed multi-buoy emission does not beat the shifted null, treat the
aggregate run as exploratory and inspect only the windows/groups that do beat
their local null.

Do not tune parameters against those local positive windows. For scale runs,
hold the preset fixed and expand coverage. The scale preset uses balanced
group selection so early filename order does not dominate the observed buoy
combinations:

```bash
python3 experiments/cdip_batch.py data/cdip/*p1_xy.nc \
  --preset scale \
  --channels z \
  --format text
```

Long scale runs should write progress to stderr and checkpoint state so they
can be resumed:

```bash
python3 experiments/cdip_batch.py data/cdip/*p1_xy.nc \
  --preset scale \
  --channels z \
  --max-groups 512 \
  --null-repeats 50 \
  --top-windows 100000 \
  --checkpoint /tmp/cdip-scale-checkpoint.json \
  --progress-every 30 \
  --checkpoint-every 60 \
  --format json > /tmp/cdip-scale-report.json
```

If interrupted, rerun the same command with `--resume`. The reported
`p_emp_ge` is the shifted-null empirical exceedance estimate and is resolution
limited by `1 / (null_repeats + 1)`. Use `p_floor` to see that limit and
`p_norm` as a separate normal approximation from the aggregate null z score.

For sharded local runs, do not launch every shard process in the shell at once.
Each `cdip_batch.py` shard loads the selected CDIP record set independently, so
`--shard-count 16` can be a deterministic partitioning choice while local
concurrency stays at 2 or 3 workers. Use the bounded launcher:

```bash
python3 experiments/cdip_sharded_run.py \
  --run-dir /tmp/cdip-scale-shards \
  --shard-count 16 \
  --max-workers 2 \
  -- \
  data/cdip/*p1_xy.nc \
  --preset scale \
  --channels z \
  --max-groups 1024 \
  --read-window-manifest /tmp/cdip_1p5gb_1024groups_windows.json \
  --null-mode permute \
  --null-repeats 1000 \
  --top-windows 100000
```

The launcher writes one report, log, and checkpoint per shard, prints aggregate
checkpoint progress, refuses unsafe memory estimates by default, and merges only
the shard report files into `merged.json`.

Known-structure controls should be run as preprocessing modes, not as detector
features. `--preprocess highpass` removes sub-cutoff low-frequency content before
observation. `--preprocess highpass-phase-randomize` first applies the high-pass
control, then preserves each window's Fourier magnitudes while deterministically
randomizing phase. Use it to test whether coherence depends on original
phase/time structure or survives as shared spectral/score-distribution
organization:

```bash
python3 experiments/cdip_sharded_run.py \
  --run-dir /tmp/cdip-phase-surrogate-shards \
  --shard-count 16 \
  --max-workers 2 \
  -- \
  data/cdip/*p1_xy.nc \
  --preset scale \
  --channels z \
  --max-groups 1024 \
  --read-window-manifest /tmp/cdip_1p5gb_1024groups_windows.json \
  --null-mode permute \
  --null-repeats 1000 \
  --preprocess highpass-phase-randomize \
  --phase-surrogate-seed 20260602 \
  --top-windows 100000
```
