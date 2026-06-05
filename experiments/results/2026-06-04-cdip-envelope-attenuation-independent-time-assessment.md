# CDIP Envelope-Attenuation And Independent-Time Assessment

Date: 2026-06-04

## Question

Does the explicit, label-blind relationship instrument recover only ordinary
ocean spectral geography, or does pair-specific structure remain as the
corpus-wide spectral envelope is progressively removed?

This is a latent-relationship discovery experiment. It is not a rogue-wave
detector, event predictor, propagation claim, or causal model.

## Preregistered Test

The corpus spectral envelope was attenuated at fixed fractions selected before
examining outcomes:

- `0.00`: ordinary normalized spectral power.
- `0.25`, `0.50`, and `0.75`: progressively attenuated corpus median
  log-spectrum.
- `1.00`: positive ratio-to-corpus-envelope endpoint.
- Signed residual control: corpus-median log-spectrum subtraction followed by
  per-frequency robust MAD scaling.

Every waveform view calibrated independently. Benjamini-Hochberg FDR correction
was applied jointly across every supported pair, waveform view, and tried
spectral representation, separately for the domain-regroup and
temporal-regroup controls. Intermediate attenuation fractions therefore cannot
win merely because several fractions were tried.

The widened pass used 437 aligned CDIP windows from 128 groups in each source
segment, 492 unique repeatedly observed buoy-pair identities, three waveform
views, and 1,476 pair/view rows. Each control family corrected 8,856
pair/view/profile hypotheses.

No coordinates, station names, oceanographic quantities, or event labels
participated in discovery. Geography entered only in the post-hoc audit.

## Method Corrections

Three implementation defects were caught and corrected before interpreting the
widened result:

1. Null RNG streams originally depended on hypothesis iteration order. Adding
   attenuation profiles could change another unchanged profile's empirical
   p-value. Null streams are now deterministically independent by layer,
   profile, control, and pair. Adding a hypothesis can change FDR multiplicity,
   but it cannot change another hypothesis's null samples.
2. Pair temporal nulls repeatedly sampled tiny finite permutation spaces. A
   pair with two possible temporal arrangements could be reported as thousands
   of redraws. Small permutation spaces are now enumerated exactly once; larger
   spaces are sampled with their space size reported.
3. Pair-domain bootstraps and temporal controls were unnecessarily serial.
   Vectorization reduced the frozen 222-window corpus audit from 860 seconds to
   575 seconds without changing any domain-regroup summary.

`2026-06-04-cdip-relationship-envelope-ladder-graph64.json` was generated
before the independent-null-stream correction and must not be used.

## Frozen-Scale Validation

The corrected 222-window run tested 273 unique pair identities.

- Ordinary raw profile: 11 FDR-significant unique identities.
- Full positive envelope attenuation: 5 unique identities.
- Signed robust residual: 6 unique identities.
- Independent-date replication: 2 discovered identities across any tier.
- FDR-significant temporal concurrence: none.

The ladder therefore rejected the earlier inference that complete envelope
removal necessarily erased every relationship. That negative result was not
stable to corrected null sampling.

## Widened Result

All small-run discoveries at every positive attenuation fraction survived at
the same fraction in the widened run. Four of six small signed-residual
identities survived the wider FDR family.

| Spectral representation | Unique pair identities | Independent-time replicated identities | Median distance km | Within 250 km | Matched geography p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary raw profile | 22 | 3 | 101.5 | 14 / 22 | < 0.00001 |
| 25% envelope attenuation | 23 | 4 | 134.0 | 13 / 23 | < 0.00001 |
| 50% envelope attenuation | 23 | 3 | 134.0 | 14 / 23 | < 0.00001 |
| 75% envelope attenuation | 19 | 3 | 451.4 | 8 / 19 | 0.00007 |
| Full positive envelope attenuation | 16 | 4 | 1,368.0 | 4 / 16 | 0.03961 |
| Signed robust residual | 15 | 4 | 662.9 | 4 / 15 | 0.01101 |

Across all tiers, 10 discovered pair identities repeat across time clusters
separated by at least 24 hours. No profile or view has FDR-significant
moment-specific temporal concurrence.

## What Fell Out

The ordinary and partially attenuated graph is strongly geographic. Without
being told that location exists, the instrument preferentially surfaces nearby
buoy pairs. This independently validates that the pair graph contains real
organizing structure rather than arbitrary pair selection.

As more of the corpus-wide spectral envelope is removed, the graph becomes
less dominated by local geography. Median detected-pair distance rises from
101.5 km in the raw profile to 1,368 km at full positive attenuation. Local
enrichment remains present but is much weaker.

The post-hoc matched metadata audit identifies what largely replaces geography
in the deeper tiers:

| Spectral representation | Sample-rate matches | Matched p | Processing-family matches | Matched p |
| --- | ---: | ---: | ---: | ---: |
| Ordinary raw profile | 12 / 22 | 0.94695 | 10 / 22 | 0.88372 |
| 50% envelope attenuation | 16 / 23 | 0.43358 | 15 / 23 | 0.31640 |
| 75% envelope attenuation | 19 / 19 | 0.00035 | 18 / 19 | 0.00059 |
| Full positive envelope attenuation | 16 / 16 | 0.00139 | 14 / 16 | 0.02392 |
| Signed robust residual | 15 / 15 | 0.00093 | 15 / 15 | 0.00028 |

The ordinary graph does not preferentially connect stations with the same
acquisition or processing class. After aggressive envelope removal, those
hidden classes dominate the discovered graph. The signed residual tier exactly
recovers same-sample-rate and same-processing-family pairs.

Examples of independently repeated relationships include:

- San Pedro, California--Santa Cruz Basin, California: 116.8 km, three
  independent time clusters, survives every positive attenuation fraction in
  all three waveform views.
- Humboldt Bay North Spit, California--Hilo, Hawaii: 3,727 km, two independent
  time clusters, survives only the signed residual representation in all three
  waveform views.
- Point Sur, California--Jeffreys Ledge, New Hampshire: 4,438 km, two
  independent time clusters, survives the signed residual only in the raw
  waveform view.

The first relationship is a strong local persistent relationship. The latter
two are nonlocal residual candidates explained at least in part by shared
acquisition and processing class, not established physical coupling.

## Interpretation

The instrument recovers at least two distinguishable pair-profile layers from
raw ocean displacement data:

1. A robust local spectral-neighborhood layer that dominates ordinary and
   partially attenuated profiles.
2. A smaller, more nonlocal acquisition/processing-system layer that becomes
   dominant after aggressive corpus-envelope removal and sometimes repeats
   across independent dates.

The deeper layer is detectable structure with a strong post-hoc attribution:
sample rate and source processing family. For an ocean-physics question this
layer is a confound to control or stratify away. For a generic latent-structure
instrument, recovering an unprovided acquisition/processing classification
from raw waveforms is a successful discovery. The current controls establish
pair-profile specificity relative to the tested corpus and reject
moment-specific concurrence. They do not establish causation, propagation, or
oceanographic novelty.

The absence of temporal-concurrence discoveries is itself informative. The
instrument is finding persistent pair identity and profile similarity, not
simultaneous events.

## Artifacts

- Corrected frozen-scale result:
  `2026-06-04-cdip-relationship-envelope-ladder-stable-nulls-graph64.json`
- Widened independent-time result:
  `2026-06-04-cdip-relationship-envelope-ladder-graph128-independent-time.json`
- Per-profile post-hoc geography audit:
  `2026-06-04-cdip-relationship-envelope-ladder-graph128-geography-audit.json`
- Per-profile post-hoc acquisition/processing metadata audit:
  `2026-06-04-cdip-relationship-envelope-ladder-graph128-metadata-audit.json`

## Next Controls

1. Rerun the deep residual tiers within homogeneous sample-rate and processing
   families to ask what remains after removing the newly identified known
   layer.
2. Attribute any within-class residual pairs after discovery using published
   CDIP sea-state and station metadata.
3. Repeat the fixed ladder on a disjoint date range or collection and
   grade recurrence of relationship classes, not exact pair identities.
4. Generalize the explicit relationship/control ladder to another domain only
   after preserving the same independent-null and multiplicity rules.
