# Cross-Domain Latent-Structure Instrument Quality Assessment

## Scope

This assessment applies one explicit validation ladder to ocean, EEG, and
social/organizational data. The instrument is not one statistic. It is a stack:

1. canonical or constructed multichannel surface;
2. frozen/self-referential or population spectral observer;
3. thresholded stigmergic accumulation;
4. relationship-mechanism diagnostics; and
5. relationship-specific nulls and second-level controls.

The purpose is to learn where the instrument finds defensible structure, where
it loses structure, and where an attractive aggregate is an artifact.

## Current Results

| Domain/result | Quality read | What survives | What does not |
| --- | --- | --- | --- |
| Social: SocioPatterns | positive | group and graph observer coactivation replicates and beats repeated random event orders | group ordering is label-aligned, not autonomously discovered |
| Social: Email-Eu | interpretable negative | raw feature-surface coactivation and phase exist | selected observer geometry does not preserve them |
| Social: Enron | unresolved | none claimed | current representation cannot construct the frozen observer |
| Ocean: CDIP | downgraded by stronger control | reproducible anchor-aligned observer-score structure; high-score windows are swell-like post hoc | original buoy-group-specific coherence does not replicate under trajectory regroup |
| EEG: within-brain | mixed positive | recording-specific structure replicates for chb01 and chb02 | chb03 fails calibration/validation replication |
| EEG: population nominal | promising, bounded | one common aggregate-developed instrument surfaces recording-specific population deviance in all three held-out runs | three epilepsy subjects cannot establish medical abnormality or external generalization |

## Sober Interpretation

The system is not declaring structure everywhere:

- it produces positive replications;
- it produces calibrated transformation losses;
- it reports under-resolved data;
- and it invalidated its own strongest prior ocean framing when given a better
  relationship-specific null.

That pattern is a good sign for the instrument-validation process. It is not
proof that every surfaced structure is useful or meaningful.

The strongest current generic result is the separation between
self-referential and population nominals in EEG:

- self-referential observation finds changes within a brain;
- population observation finds stable differences between a brain and peers;
- the same brain can be quiet on one axis and unusual on the other.

Under one common population instrument, `chb02` shows rare but strongly
seizure-associated deviations, while `chb03` shows a broad persistent
population difference. That is the kind of distinction a purely within-entity
baseline cannot make.

The most important negative result is CDIP. Timing permutation, tide removal,
spectral masking, and held-out groups were insufficient to establish
group-specific coherence. A trajectory-regroup null preserved the shared score
template while breaking only buoy membership, and the claimed buoy-group
surplus did not replicate. The earlier ocean claim is therefore superseded.

## Required Rules Going Forward

- Define the relationship being claimed before choosing the null.
- Use trajectory regrouping before calling a grouped aggregate
  group-specific coherence.
- Keep self-referential and population nominals as separate axes.
- Independently calibrate whether the observed matrix shape can detect a
  mechanism before interpreting its absence.
- Treat aggregate sign as operator output, not mechanism identity.
- Freeze label-free results before post-hoc attribution.
- Require a genuinely unseen entity after aggregate/common-instrument
  development before claiming external generalization.

## Next Highest-Value Experiments

1. Expand the common population EEG instrument to additional CHB-MIT subjects.
   Freeze the current common candidate first; use new subjects as true external
   tests before any retuning.
2. Add suitable non-epilepsy or healthy-control EEG if acquisition geometry is
   comparable. That is required before discussing abnormality.
3. Add relation-change features to population scoring so the instrument can
   distinguish channel-level spectral difference from altered cross-channel
   organization.
4. Apply trajectory-regroup and population/self-reference separation to PMU
   and other infrastructure-shaped datasets.
5. Revisit ocean only if a new representation or control targets a specific
   relationship beyond the invalidated aggregate group-coherence claim.

## Primary Artifacts

- `experiments/results/2026-06-04-social-observer-layer-assessment.md`
- `experiments/results/2026-06-04-cdip-validation-ladder-assessment.md`
- `experiments/results/2026-06-04-chbmit-validation-ladder-assessment.md`
- `experiments/results/2026-06-04-cdip-observer-group-null1000.json`
- `experiments/results/2026-06-04-chbmit-observer-group-null1000.json`
- `experiments/results/2026-06-04-chbmit-population-nominal.json`
