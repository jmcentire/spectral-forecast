# CHB-MIT EEG Validation-Ladder Assessment

## Questions

The EEG work now separates two different questions:

1. **Within-brain deviance:** does a recording contain cross-channel structure
   relative to that brain's own frozen nominal?
2. **Population deviance:** does a brain contain persistent cross-channel
   spectral organization that differs from a nominal built from other brains?

The second question is required because a persistently unusual brain can define
itself as normal under a purely self-referential baseline.

## Within-Brain Frozen Observer

Each subject independently completed the full label-free candidate grid on its
own calibration files. No `chb01` candidate was imposed on `chb02` or `chb03`.
All three independently selected the same baseline/adaptive/stride/min-channel
geometry; `chb01` selected threshold `2.5`, while `chb02` and `chb03` selected
`3.0`.

The channel-preserving recording-regroup null keeps every complete observer
score trajectory and electrode identity, but rebuilds synthetic recordings
using channels from different files. It therefore tests whether channels from
the same recording contain extra organization.

| Subject | Calibration regroup delta | Calibration p | Validation regroup delta | Validation p | Replicates? |
| --- | ---: | ---: | ---: | ---: | --- |
| chb01 | 2081.72 | 0.0010 | 580.28 | 0.0010 | yes |
| chb02 | 485.04 | 0.0010 | 2275.09 | 0.0020 | yes |
| chb03 | -270.64 | 0.4675 | 6496.23 | 0.0050 | no |

Result: recording-specific within-brain cross-channel structure replicates for
two of three subjects. `chb03` is unstable across its calibration/validation
file split and must not be counted as a within-brain replication.

Primary artifact:

- `experiments/results/2026-06-04-chbmit-observer-group-null1000.json`

## Cross-Fitted Population Nominal

### Method

The population experiment uses a separate generic feature surface:

- per-recording robust location/scale removal as acquisition nuisance control;
- scale-free spectral-shape features in evenly divided frequency bins;
- generic spectral entropy, centroid, bandwidth, peak concentration,
  flatness, and lag-1 correlation;
- robust channel-feature population centers and scales;
- multi-channel population-deviance emissions.

The feature surface contains no EEG bands, seizure labels, patient labels, or
clinical features.

One common candidate was developed on aggregate data by its conservative
performance across all three cross-fitted development folds. It passed all
three label-free development gates:

- window: `256` samples = `16` seconds;
- stride: `256` samples;
- generic spectral bins: `8`;
- feature-deviance quantile: `0.9`;
- emission threshold: `2.5`;
- minimum active channels: `3`;
- minimum development-fold quality: `12.77`;
- minimum development-fold group delta: `236.65`;
- worst development-fold empirical p: `0.0099`.

The common instrument was then run on all three subjects. For each subject, the
population nominal was fit only on the other two subjects.

This makes subject scores comparable under one instrument, while keeping each
subject out of its own nominal. Because all three subjects contributed to
common-candidate development in other folds, a fourth unseen subject is still
required for strict external generalization.

### Population Result

| Held-out subject | Population-deviant row fraction | Cell-score median | Cell-score q95 | Recording-regroup delta | Regroup z effect | Regroup p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chb01 | 0.49% | 1.149 | 2.301 | 16.69 | 5.89 | 0.0020 |
| chb02 | 1.64% | 1.287 | 2.727 | 66.27 | 9.04 | 0.0010 |
| chb03 | 22.22% | 1.629 | 4.090 | 1228.47 | 9.05 | 0.0010 |

All three held-out subjects show recording-specific population-deviance
organization under the common instrument. The magnitude is not similar:
`chb03` is persistently unlike the `chb01+chb02` population.

That difference is broad rather than a single-electrode artifact:

- `chb03` channel positive fractions are high in `FP1-F7` (28.8%),
  `T7-P7` (31.3%), `FZ-CZ` (23.0%), and `FP2-F8` (36.6%);
- its strongest generic deviations are spectral peak concentration, spectral
  entropy, and several spectral-bin fractions.

This establishes a persistent population difference under the current generic
representation. It does **not** establish medical abnormality. The corpus
contains three epilepsy subjects and no healthy controls; the difference could
reflect physiology, pathology, acquisition, montage behavior, or persistent
artifact.

### Post-Hoc Seizure Attribution

Seizure labels were introduced only after the common instrument and population
nominals were frozen.

| Subject | Interictal positive | Preictal positive | Ictal positive | Postictal positive | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| chb01 | 12/2550 = 0.47% | 2/247 = 0.81% | 0/27 = 0% | 1/246 = 0.41% | little population deviation |
| chb02 | 40/2805 = 1.43% | 0/82 = 0% | 4/11 = 36.36% | 5/86 = 5.81% | event-associated population deviation |
| chb03 | 569/2627 = 21.66% | 47/234 = 20.09% | 3/27 = 11.11% | 81/262 = 30.92% | broad persistent difference, not seizure-only |

`chb02` is the cleanest event-associated population result: ictal population
deviation is about 25 times its interictal rate, with a smaller postictal
increase. Counts are still small and temporally dependent, so this is
descriptive attribution, not a seizure predictor or a clinical result.

`chb03` is different in another way: the population-deviant regime is already
common during interictal windows. A self-referential baseline largely hides
that kind of stable difference; the population nominal surfaces it.

Primary artifact:

- `experiments/results/2026-06-04-chbmit-population-nominal.json`

## Validation Ladder

| Layer | Status | Assessment |
| --- | --- | --- |
| Information sufficiency | bounded but usable | 42 recordings across three subjects; enough repeated windows for controls, too few subjects for population claims. |
| Known-answer validation | passed for generic primitive | Synthetic tests confirm robust population nominals surface persistent held-out feature shifts. |
| Canonical surface | explicit | Six common EEG channels, generic spectral-shape feature tensor. |
| Frozen observer | mixed within-brain | Recording-specific structure replicates for chb01/chb02, not chb03. |
| Population observer | promising | One common aggregate-developed instrument surfaces cross-fitted subject differences. |
| Relationship null | passed for population result | Channel-preserving recording regroup is exceeded for all three common-instrument held-out runs. |
| Post-hoc attribution | differentiated | chb02 is event-associated; chb03 is broadly population-different; chb01 is mostly quiet. |
| External validity | open | Requires more subjects, healthy or otherwise appropriate controls, and a fully unseen subject after common-instrument freeze. |

## General Method Learning

Self-referential and population nominals answer orthogonal questions:

- self-referential nominal: **is this entity changing?**
- population nominal: **is this entity persistently unlike its peers?**

A generic structure-discovery system should retain both axes. Collapsing them
causes stable abnormality to disappear into the entity's own definition of
normal.

## Frozen External Extension

The common population instrument was subsequently frozen and run without
retuning on checksum-verified `chb04`, `chb05`, and `chb06` recordings. Each
external subject exceeded the channel-preserving recording-regroup null, and
its signed population-deviation fingerprint replicated across disjoint,
sequence-balanced recording halves.

The external subjects exposed different geometries:

- `chb04`: persistent population difference dominates self-change;
- `chb05`: ordered later self-change dominates; and
- `chb06`: self-change dominates while population deviation remains rare.

This is external replication of an instrument response, not identification of
medical abnormality or a prediction result. See
`2026-06-04-chbmit-frozen-external-assessment.md`.
