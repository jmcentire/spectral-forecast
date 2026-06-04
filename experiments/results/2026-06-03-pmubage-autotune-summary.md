# pmuBAGE PMU Autotune Result

Date: 2026-06-03

## Scope

This pass tested the fresh label-free autotune workflow on local pmuBAGE
synthetic PMU tensors. pmuBAGE is useful because it has the infrastructure
shape we care about: many PMU-like sensors, short high-rate time series, and
grid-event-like coherent signals. It is still synthetic, so this is an
adapter/control result, not validation on real utility disturbance records.

Source: https://github.com/NanpengYu/pmuBAGE

## Method

Each pmuBAGE event was treated as one multi-sensor observation:

- sensors: first 24 PMUs
- samples: 600 samples over 20 seconds
- sample rate: 30 Hz
- selection: aggregate cross-PMU emission lift on calibration events
- validation: freeze the selected candidate and score disjoint held-out events
- null: timing permutation within each PMU event
- labels/domain metadata: not used for selection

The wrapper added for this pass is `experiments/pmubage_autotune.py`.

## Frequency Tensor

Artifact:

- `experiments/results/2026-06-03-pmubage-autotune-frequency-bounded.json`

Run shape:

- tensors: `frequency_0.npy`, `frequency_1.npy`
- datatype: frequency
- events: 8 total, 4 calibration, 4 validation
- candidates: 36
- null repeats: 100

Selected config:

- baseline: `256`
- adaptive window: `128`
- stride: `32`
- threshold: `8.0`
- min active sensors: `12`

Calibration:

- accepted: `true`
- delta: `305.764`
- z effect: `2.97`
- positive event fraction: `0.50`
- saturation: `0.790`
- null exceedances: `0 / 100`
- empirical p floor: `0.0099`

Validation:

- accepted: `false`
- delta: `-177.237`
- z effect: `-1.86`
- positive event fraction: `0.25`
- saturation: `0.739`
- null exceedances: `95 / 100`
- empirical p: `0.9505`

Interpretation: the frequency surface is not useful validation for the current
aggregate coherence statistic. The calibration setting barely clears the gate
and does not transfer to held-out events. This is the synthetic version of a
saturation/control failure: the signal is globally structured enough that the
null often sees the same thing, and stricter gating does not produce stable
held-out support.

## Voltage Magnitude Tensor

Artifact:

- `experiments/results/2026-06-03-pmubage-autotune-voltage-bounded.json`

Run shape:

- tensor: `voltage_0.npy`
- datatype: voltage magnitude
- events: 20 total, 10 calibration, 10 validation
- candidates: 36
- null repeats: 100

Selected config:

- baseline: `128`
- adaptive window: `64`
- stride: `32`
- threshold: `12.0`
- min active sensors: `12`

Calibration:

- accepted: `true`
- delta: `3791.821`
- z effect: `10.61`
- positive event fraction: `0.80`
- saturation: `0.114`
- null exceedances: `0 / 100`
- empirical p floor: `0.0099`

Validation:

- accepted: `false`
- delta: `609.631`
- z effect: `5.10`
- positive event fraction: `0.20`
- saturation: `0.037`
- null exceedances: `0 / 100`
- empirical p floor: `0.0099`

Interpretation: voltage magnitude is more interesting than frequency because
it avoids the saturation problem and retains positive aggregate lift on
held-out events. It still fails the validation gate because only 2 of 10
held-out events have positive event-level support. The support gate is doing
real work here: it prevents a few loud synthetic events from being reported as
generalized structure.

## What pmuBAGE Establishes

Established:

- The PMU adapter works with the same label-free candidate/validation workflow
  used for CDIP and CHB-MIT.
- Synthetic frequency events expose a saturation/non-transfer failure mode.
- Synthetic voltage events expose a sparse-support failure mode: aggregate lift
  can remain positive while event-level support is too thin to accept.
- The validation gate correctly rejects both bounded PMU passes.

Not established:

- Real PMU disturbance validity.
- Failure precursor detection.
- Grid outage prediction.

The important result is negative-but-useful: PMU-shaped synthetic data is now a
control domain that demonstrates why support, saturation, and held-out
validation gates are necessary before moving to real infrastructure telemetry.

