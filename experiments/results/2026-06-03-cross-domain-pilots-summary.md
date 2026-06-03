# Cross-Domain Pilot Summary

Date: 2026-06-03

## Summary

After the CDIP calibration result, two additional dataset families were checked:

1. CHB-MIT scalp EEG, using four one-hour `chb01` EDF files.
2. pmuBAGE synthetic PMU events, using one frequency tensor event and one voltage-magnitude tensor event.

Both pilots used the same discipline as CDIP: the observer saw only sensor time series. Labels and dataset metadata were used only after detection or for access/scout interpretation.

## CHB-MIT EEG

Source: https://physionet.org/content/chbmit/1.0.0/

Pilot files:

- `chb01_01.edf`: no labeled seizure.
- `chb01_02.edf`: no labeled seizure.
- `chb01_03.edf`: one seizure, 2996-3036 seconds.
- `chb01_04.edf`: one seizure, 1467-1494 seconds.

Command:

```bash
python3 experiments/chbmit_observe.py \
  data/chbmit/chb01/chb01_01.edf \
  data/chbmit/chb01/chb01_02.edf \
  data/chbmit/chb01/chb01_03.edf \
  data/chbmit/chb01/chb01_04.edf \
  --summary data/chbmit/chb01/chb01-summary.txt \
  --null-repeats 200 \
  --top 12 \
  --output experiments/results/2026-06-03-chbmit-chb01-4file-pilot.json
```

Settings:

- Six EEG channels: `FP1-F7`, `T7-P7`, `FZ-CZ`, `CZ-PZ`, `FP2-F8`, `P8-O2`.
- Resampled to 16 Hz for a bounded first pass.
- Frozen baseline: 1024 samples.
- Sliding window: 512 samples.
- Stride: 256 samples.
- Coherent emission requires at least two active channels.
- Timing-permutation null preserves each channel's score distribution and breaks cross-channel timing.

Results:

| File | Labeled seizures | Observed | Null mean | z effect | Exceedances | Empirical p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chb01_01.edf` | 0 | 201.724 | 76.030 | 5.96 | 0/200 | 0.00498 floor |
| `chb01_02.edf` | 0 | 376.483 | 255.109 | 4.77 | 0/200 | 0.00498 floor |
| `chb01_03.edf` | 1 | 510.603 | 458.054 | 3.85 | 0/200 | 0.00498 floor |
| `chb01_04.edf` | 1 | 649.605 | 626.185 | 1.27 | 19/200 | 0.0995 |

Top coherent phase counts, using labels only after detection:

| File | Top coherent phase counts |
| --- | --- |
| `chb01_01.edf` | `interictal: 12` |
| `chb01_02.edf` | `interictal: 12` |
| `chb01_03.edf` | `interictal: 8`, `postictal: 4` |
| `chb01_04.edf` | `interictal: 7`, `postictal: 4`, `ictal: 1` |

Interpretation:

- The detector surfaces cross-channel EEG structure in all four files.
- The non-seizure files also beat their timing-permutation nulls, so cross-channel coherence alone is abundant background structure in EEG.
- The labeled seizure files show some top coherent rows near or after labeled seizures, especially postictal rows, but this pilot does not establish seizure-specific structure.
- The useful next question is label enrichment under fixed settings: expand to more `chb01` seizure files, then patient-heldout files, and ask whether top coherent rows overrepresent preictal, ictal, or postictal intervals versus matched interictal windows.

## pmuBAGE Synthetic PMU

Source: https://github.com/NanpengYu/pmuBAGE

Scout result:

- The repo is accessible and contains real `.npy` tensors.
- It is synthetic and about 2.5 GB locally after clone.
- Frequency tensors have shape `4 x 4 x 100 x 600`.
- Voltage tensors have shape `20 x 4 x 100 x 600`.
- Axes are event, datatype `PQVF`, PMU index, and time.
- There are 600 time samples over 20 seconds, so the implied sample rate is 30 Hz.

Smoke commands:

```bash
python3 experiments/pmubage_observe.py data/pmuBAGE/data/frequency/frequency_0.npy \
  --event-index 0 \
  --datatype-index 3 \
  --sensor-limit 24 \
  --null-repeats 200 \
  --output experiments/results/2026-06-03-pmubage-frequency0-event0-smoke.json

python3 experiments/pmubage_observe.py data/pmuBAGE/data/voltage/voltage_0.npy \
  --event-index 0 \
  --datatype-index 2 \
  --sensor-limit 24 \
  --null-repeats 200 \
  --output experiments/results/2026-06-03-pmubage-voltage0-event0-smoke.json
```

Results:

| Tensor | Datatype | Observed | Null mean | z effect | Exceedances | Unique null totals | Empirical p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frequency_0.npy` | frequency | 36059.242 | 36059.242 | 0.00 | 184/200 | 4 | 0.920 |
| `voltage_0.npy` | voltage magnitude | 3962.037 | 3962.037 | 0.00 | 112/200 | 5 | 0.562 |

Interpretation:

- The detector sees enormous all-sensor structure in the synthetic PMU events.
- The timing-permutation null also sees the same aggregate structure because nearly every PMU is active across nearly every observed window.
- That makes this useful adapter rehearsal, but a weak validation dataset for the current coherence statistic.
- Real PMU disturbance records remain the better grid/infrastructure bridge.

## Current Cross-Domain Read

CDIP, CHB-MIT, and pmuBAGE now give three different calibration lessons:

- CDIP: the agnostic detector recovered physically meaningful ocean structure that survives several controls.
- CHB-MIT: the detector readily surfaces cross-channel EEG structure, but label enrichment must be tested before interpreting event relevance.
- pmuBAGE: synthetic globally coherent events can saturate the statistic and the null together, exposing a limit of aggregate cross-sensor coherence as a validation signal.

The next strong experiment is still CHB-MIT expansion, because it has real labels and real multi-sensor physiology. pmuBAGE should stay as an adapter smoke test while real PMU event data is scouted.
