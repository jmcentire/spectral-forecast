# CHB-MIT fresh autotune expanded pass

Date: 2026-06-03

## Purpose

Run the label-free spectral + stigmergy autotune system on a fresh non-ocean
domain, then inspect CHB-MIT seizure labels only after candidate selection.
This tests whether the agnostic observer surfaces coherent cross-channel
structure before any label-aware interpretation.

## Inputs

- Dataset: CHB-MIT Scalp EEG Database, subject `chb01`
- Files used: 14 readable EDF files
- Excluded local partial: `chb01_10.edf`
- Seizure-bearing files included: `chb01_03`, `chb01_04`, `chb01_15`,
  `chb01_16`, `chb01_18`, `chb01_21`, `chb01_26`
- Channels: default 6-channel subset from `experiments/chbmit_observe.py`
- Labels: not used during candidate scoring or selection

The downloader initially left `chb01_05`, `chb01_06`, and `chb01_07` as
truncated-but-large files. The expanded run used EDF loader validation instead
of file-size validation so partial files were excluded or repaired before
compute.

## Command

```bash
FILES=$(cat /tmp/chb01_ok_files.txt)
python3 experiments/chbmit_autotune.py $FILES \
  --summary data/chbmit/chb01/chb01-summary.txt \
  --validation-split alternate \
  --baselines 512,1024 \
  --adaptive-windows 256,512 \
  --strides 256 \
  --thresholds 2.5,3.0 \
  --min-active-channels 2,3 \
  --null-modes permute \
  --null-repeats 100 \
  --top-candidates 8 \
  --top-per-file 12 \
  --progress \
  --progress-every 30 \
  --checkpoint /tmp/chbmit_chb01_expanded_autotune.checkpoint.json \
  --checkpoint-every 60 \
  --output experiments/results/2026-06-03-chbmit-autotune-chb01-expanded.json
```

## Result

Selected config:

- `baseline_size=1024`
- `adaptive_window=256`
- `stride=256`
- `emission_threshold=2.5`
- `min_active_series=3`
- `decay=0.9`

Calibration:

- Files: 7
- Accepted: `true`
- Delta above permuted null: `950.625`
- Standardized effect size: `z=11.38`
- Positive file fraction: `0.857`
- Empirical p is floor-limited: `p_ge=0.00990` with 100 null repeats
- Unique null totals: `100`

Held-out validation:

- Files: 7
- Accepted: `true`
- Delta above permuted null: `937.333`
- Standardized effect size: `z=15.73`
- Positive file fraction: `1.000`
- Empirical p is floor-limited: `p_ge=0.00990` with 100 null repeats
- Unique null totals: `100`

The important statistical read is not the normal-theory p implied by the
z-score. Report the empirical bound and use z as a standardized effect size.

## Post-hoc label inspection

Validation anchor counts:

- Ictal: 16
- Preictal: 151
- Postictal: 135
- Interictal: 1166

Validation positive-window rates:

- Ictal: `15 / 16 = 0.938`
- Preictal: `42 / 151 = 0.278`
- Postictal: `35 / 135 = 0.259`
- Interictal: `159 / 1166 = 0.136`

Validation top-window counts, using top 12 emitted windows per file:

- Ictal: 9
- Preictal: 9
- Postictal: 8
- Interictal: 58

Interpretation: the selected label-free configuration surfaces coherent
cross-channel structure that survives a timing-permutation null on held-out
files. After labels are overlaid, ictal windows are strongly enriched among
positive windows, and pre/postictal windows are elevated relative to interictal
baseline. This is not a seizure predictor claim: top windows remain mostly
interictal because most available anchors are interictal and the detector is
surfacing generic coherent structure, not optimizing seizure proximity.

## Method changes

`experiments/chbmit_autotune.py` now supports:

- Candidate-level checkpoints with `--checkpoint`
- Resume with `--resume`
- Atomic checkpoint writes
- Invalid-grid skipping, so one impossible baseline/adaptive pair cannot kill a
  run before compute starts

