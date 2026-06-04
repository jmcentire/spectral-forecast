# CDIP Big Ocean Fresh-Naive Autotune

Date: 2026-06-03

## Scope

This run tested the autotune system on the large local CDIP ocean corpus without
training from the small parity run. The runner received the neutral candidate
grid:

- preprocess: `none`, `highpass`, `highpass-dominant-mask`
- baseline: `1024`
- adaptive window: `512`
- stride: `512`
- thresholds: `2.5`, `3.0`
- decay: `0.9`
- min active series: `2`

The run did not use labels, sea-state products, known swell events, or the
previous small-run winner as detector inputs.

## Discovery/Progress Fix

The first big run attempt exposed a quiet pre-scoring bottleneck in balanced
window discovery: `_discover_windows` recomputed SHA1 digests and rescanned all
remaining buoy triples for every selected group. The helper now precomputes
combo metadata, uses a lazy heap for the same deterministic rank rule, and emits
discovery progress. Focused CDIP tests passed after the change:

- `pytest tests/test_cdip_batch.py tests/test_cdip_autotune.py -q`
- result: `18 passed`

## Full 1024-Group Fresh-Naive Shift Run

Artifact:

- `experiments/results/2026-06-03-cdip-autotune-big-ocean-naive-1024groups-shift.json`

Run shape:

- 70 local CDIP files
- 1024 balanced calibration groups
- 5491 calibration windows
- 1 disjoint validation window, kept only as a command-shape sanity check
- shift null, 50 repeats
- elapsed: `6727.0s`
- peak RSS: about `34.3GB`

Best selected candidate:

- preprocess: `highpass`
- threshold: `3.0`
- delta: `3924.719`
- z effect: `11.278`
- null repeats: `50`
- unique null totals: `6`
- null exceedances: `0`
- empirical p floor: `0.0196078`

Manual big-run reference:

- artifact: `experiments/results/2026-06-02-cdip-scale-1p5gb-1024groups-8shards-null50-summary.json`
- records: `70`
- groups: `1024`
- windows: `5491`
- preset: manual scale config, effectively `preprocess=none`, threshold `3.0`
- delta: `3913.050`
- z effect: `11.193`
- null repeats: `50`
- null exceedances: `0`
- empirical p floor: `0.0196078`

The exact manual-like candidate appeared as the second-ranked fresh-naive
candidate:

- preprocess: `none`
- threshold: `3.0`
- delta: `3913.050`
- z effect: `11.081`
- null repeats: `50`
- unique null totals: `6`
- null exceedances: `0`

Interpretation: the fresh-naive autotune pass recovered the manual big-ocean
result and selected a slightly stronger adjacent setting. The p value is still a
50-repeat shifted-null floor, and because the aggregate shifted null has only
six unique totals, report z as standardized effect size rather than
normal-theory significance.

## 512/512 Disjoint Heldout Shift Check

Artifact:

- `experiments/results/2026-06-03-cdip-autotune-big-ocean-naive-512-512-heldout-shift.json`

Run shape:

- 512 calibration groups, 2769 windows
- 512 disjoint validation groups, 2722 windows
- shift null, 50 repeats
- disk matrix cache reused from the 1024-group run
- elapsed after cache: `12.9s`

Selected candidate:

- preprocess: `highpass`
- threshold: `3.0`

Calibration:

- delta: `1883.905`
- z effect: `7.769`
- null repeats: `50`
- unique null totals: `6`
- null exceedances: `0`

Validation:

- delta: `2040.814`
- z effect: `17.428`
- null repeats: `50`
- unique null totals: `6`
- null exceedances: `0`

Interpretation: the fresh selected setting survives a large disjoint heldout
split under the same shifted-null diagnostic. The same cyclic shifted-null
caveat applies.

## 512/512 Disjoint Shift+Permute Check

Artifact:

- `experiments/results/2026-06-03-cdip-autotune-big-ocean-naive-512-512-heldout-shift-permute50.json`

Run shape:

- 512 calibration groups, 2769 windows
- 512 disjoint validation groups, 2722 windows
- null modes: `shift,permute`
- null repeats: `50`
- worst null mode: `permute`
- disk matrix cache reused from the 1024-group run
- elapsed after cache: `20.5s`

Selected candidate:

- preprocess: `none`
- threshold: `3.0`

Calibration:

- delta: `1663.710`
- z effect: `7.751`
- null repeats: `50`
- unique null totals: `50`
- null exceedances: `0`
- empirical p floor: `0.0196078`

Validation:

- delta: `1835.315`
- z effect: `11.832`
- null repeats: `50`
- unique null totals: `50`
- null exceedances: `0`
- empirical p floor: `0.0196078`

Interpretation: this is the cleaner fresh-naive big-ocean validation result. It
keeps the label-free candidate search, uses a disjoint group split, and survives
the stricter timing permutation null on heldout groups. The p value is still
floor-limited by only 50 repeats, but unlike the shifted-only run, the aggregate
null totals are not cyclic-collapsed.

## Bottom Line

The auto-system is at parity with the manual big-ocean result and passes a
large disjoint heldout check. The strongest defensible statement is:

> A fresh label-free autotune pass on the 70-file CDIP corpus rediscovers the
> manual large-ocean coherent-structure result, selects the same threshold
> family, and the selected setting survives a 512/512 disjoint heldout split
> under a permutation null.

Next step: move to another domain with the same discipline: neutral candidate
grid, explicit predictable-layer controls, disjoint validation where available,
and empirical-null reporting that includes repeat count, exceedances, p floor,
and unique null totals.
