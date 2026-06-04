# CHB-MIT Frozen External Structure-Discovery Assessment

## Question

Does the common label-free population instrument developed on `chb01-03`
surface reproducible structure on genuinely unseen `chb04-06` recordings
without retuning?

This is a structure-discovery experiment, not a seizure predictor experiment.
Seizure labels and subject metadata were withheld until after all structure
surfaces and controls were frozen.

## Frozen Protocol

- Development subjects: `chb01`, `chb02`, `chb03`
- External subjects: `chb04`, `chb05`, `chb06`
- Recordings per subject: `14`
- External EDF inputs: verified against PhysioNet's published SHA-256 values
- Frozen source report:
  `experiments/results/2026-06-04-chbmit-population-nominal.json`
- Frozen candidate:
  - `16`-second window and stride
  - `8` generic spectral bins
  - feature-deviance quantile `0.9`
  - emission threshold `2.5`
  - minimum `3` active channels

The three development subjects alone fit the population nominal. External
subjects contributed to neither candidate selection nor population fitting.

For each external subject, the experiment constructed three views:

1. **Population view:** difference from the frozen development population.
2. **Balanced self view:** difference from alternating recordings of the same
   subject.
3. **Ordered self view:** later recordings compared with an earlier-recording
   prefix.

The operational structure test is a channel-preserving recording-regroup null.
It preserves complete channel-score trajectories and channel identity, but
rebuilds synthetic recordings from channels belonging to different files. A
surplus therefore means channels from the same recording contain additional
organization under the frozen observer. It does not establish the cause or
medical meaning of that organization.

## External Population Result

| Subject | Population-positive rows | Recording-regroup delta | Regroup z effect | Regroup empirical p |
| --- | ---: | ---: | ---: | ---: |
| chb04 | 14.21% | 1729.65 | 7.69 | 0.0010 |
| chb05 | 2.70% | 187.80 | 16.65 | 0.0010 |
| chb06 | 1.64% | 1396.83 | 9.67 | 0.0010 |

All three external subjects contain same-recording organization under the
frozen population instrument. The magnitude differs substantially, so this is
not one generic "external EEG looks anomalous" result.

## Within-Subject Replication

Each subject's files were split into two disjoint, sequence-balanced halves.
Both halves were scored against the same frozen population nominal.

| Subject | Half A positive | Half A p | Half B positive | Half B p | Signed fingerprint cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| chb04 | 15.31% | 0.0310 | 13.34% | 0.0010 | 0.920 |
| chb05 | 2.54% | 0.0010 | 2.86% | 0.0010 | 0.968 |
| chb06 | 0.70% | 0.0010 | 2.60% | 0.0010 | 0.989 |

The population-deviation direction repeats within each unseen subject. This is
the strongest current external result: a frozen instrument surfaced
recording-specific organization and a stable signed channel-feature geometry
on data that did not participate in development.

It remains an instrument-response replication. It does not identify the
structure's cause.

## Different Structural Geometries

The population and self views separate three different patterns:

| Subject | Population all | Balanced self | Ordered self | Dominant contrast |
| --- | ---: | ---: | ---: | --- |
| chb04 | 14.21% | 5.32% | 4.12% | persistent population difference dominates self-change |
| chb05 | 2.70% | 1.97% | 7.62% | later within-subject change dominates |
| chb06 | 1.64% | 8.16% | 4.83% | within-subject change while remaining near the population |

On comparable validation rows:

- `chb04` is mostly population-only: `8.09%` population-only versus `0.06%`
  self-only in the balanced view.
- `chb05` changes with ordering: the later view is `6.60%` self-only versus
  `0.57%` population-only.
- `chb06` is mostly self-only: `5.57%` self-only versus `0.02%`
  population-only in the balanced view.

This directly confirms the methodological reason to keep population and
self-referential nominals separate. One asks whether an entity differs from
peers; the other asks whether it changes relative to itself.

## Signed Relationships

Absolute anomaly magnitude is unsuitable for structural comparison because
opposite shifts can look similar. The external runner therefore uses signed
median channel-feature deviations for subject fingerprints while retaining
absolute deviation for anomaly emission.

| Pair | Signed fingerprint cosine | RMS difference |
| --- | ---: | ---: |
| chb05 / chb06 | 0.546 | 0.534 |
| chb04 / chb05 | -0.097 | 0.643 |
| chb04 / chb06 | -0.436 | 0.842 |

The external subjects do not share one deviation direction. `chb05` and
`chb06` are moderately aligned, while `chb04` is nearly orthogonal to `chb05`
and opposed to `chb06`. With three subjects, these are relationships to
investigate, not clusters to name.

## Population-Reference Sensitivity

The frozen candidate was also scored after rebuilding the population nominal
while omitting each original development subject.

| Subject | Omission positive-fraction range | Minimum fingerprint cosine to full nominal | All omission regroup p |
| --- | ---: | ---: | ---: |
| chb04 | 11.83%-23.35% | 0.891 | 0.0010 |
| chb05 | 2.73%-4.57% | 0.875 | 0.0010 |
| chb06 | 1.37%-2.43% | 0.971 | 0.0010 |

The exact amount of population deviance depends on who defines the population,
especially for `chb04`. The same-recording organization and general signed
fingerprint direction survive every omission. Omitting `chb03` increases
positive fractions for all three external subjects, consistent with the
previous finding that `chb03` broadens the small development population.

## Post-Hoc Attribution

Labels were applied only after the structure results were frozen.

| Subject | Interictal positive | Preictal positive | Ictal positive | Postictal positive | Descriptive read |
| --- | ---: | ---: | ---: | ---: | --- |
| chb04 | 14.08% | 25.68% | 30.00% | 19.74% | broad population difference with some event-phase enrichment |
| chb05 | 2.03% | 4.76% | 71.43% | 14.67% | rare structure strongly concentrated around recorded events |
| chb06 | 1.72% | 0% | 0% | 0% | surfaced structure is not aligned with recorded event phases |

The contrast matters. The frozen instrument did not merely rediscover seizures
in every subject. It found event-associated structure in `chb05`, broader
population structure in `chb04`, and non-event-aligned self structure in
`chb06`.

The CHB-MIT summary parser was corrected during this run to support numbered
multi-seizure lines such as `Seizure 1 Start Time`. That correction changed only
post-hoc attribution, not structure scoring or candidate selection.

## What Is Supported

- A frozen label-free instrument developed on three subjects surfaces
  same-recording multi-channel organization on three unseen subjects.
- The signed population-deviation geometry replicates across disjoint
  recording halves within each unseen subject.
- Population and self-reference expose materially different structures.
- Ordered and sequence-balanced self views expose different aspects of change.
- Signed fingerprints preserve relationships that absolute anomaly magnitude
  destroys.
- The result survives small-development-population composition changes, though
  magnitude is baseline-sensitive.

## What Is Not Supported

- Medical abnormality or diagnosis.
- A seizure predictor.
- A claim that every surfaced structure is neural rather than acquisition,
  developmental, medication, montage, or other corpus structure.
- Cross-corpus generalization.
- A named explanation for the three external structural geometries.

All six subjects are from the same epilepsy corpus, and the development
population is small. Post-hoc subject metadata also spans markedly different
ages, making developmental structure an obvious candidate explanation that
requires a larger, properly stratified experiment.

## Next Experiment

Expand the frozen external run to more CHB-MIT subjects without retuning, then
grade:

1. within-subject split replication;
2. sensitivity to development-population composition;
3. signed fingerprint stability and recurring relationship families; and
4. attribution against subject metadata only after structural families are
   frozen.

The relation-change feature task remains separate. The current result compares
signed channel-feature structure; it does not yet directly model changes in
cross-channel relationships.

Primary artifact:

- `experiments/results/2026-06-04-chbmit-population-external.json`
