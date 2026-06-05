# CDIP Explicit Relationship Discovery Assessment

Date: 2026-06-04

## Follow-Up Status

The first broad relationship graph below remains the historical starting
point, but its conclusion that complete envelope removal left no
FDR-significant pair was not stable after correcting hypothesis-order-dependent
null RNG streams. The preregistered attenuation ladder and widened
independent-time result supersede that specific negative conclusion:
`2026-06-04-cdip-envelope-attenuation-independent-time-assessment.md`.

## Question

Can a label-agnostic instrument recover meaningful relationship structure from
canonical ocean waveforms when it is told only which mathematical relationship
families to measure?

This is not a rogue-wave predictor. No wave labels, coordinates, platform names,
or published oceanographic indicators participate in discovery.

## Method

The final broad pass uses:

- 222 aligned 4,096-sample CDIP windows selected from 64 calibration groups and
  64 validation groups.
- 273 unique buoy-pair identities and 819 pair/view hypotheses.
- Three non-destructive views: raw, 50% dominant-bin attenuation, and 100%
  dominant-bin attenuation.
- Explicit relationship families: common mode, spectral alignment, phase
  coherence, lagged dependence, and relation change.
- Exact-geometry known-answer calibration performed independently for each
  transformation layer.
- Relationship-specific local nulls, corpus domain-regroup controls,
  identity-preserving temporal-regroup controls, and Benjamini-Hochberg FDR
  correction across supported pair/view hypotheses.
- Independent temporal replication defined as observations separated by at
  least 24 hours.
- Geographic coordinates used only after discovery as an external audit.

The final machine-readable relationship result is
`2026-06-04-cdip-relationship-discovery-graph64-null4999.json`. The post-hoc
audit is `2026-06-04-cdip-relationship-geography-audit.json`.

## Corrections Caught

The stricter system rejected several initially attractive artifacts:

1. Raw-geometry calibration cannot license residualized views. Joint
   common-mode subtraction manufactured spectral and phase relationships. Its
   own transformed-view calibration rejected those families.
2. Frequency-bin permutation established shared wave-band structure but did not
   establish pair specificity. Pair regrouping showed which identities were
   unusual relative to the corpus.
3. Nominal pair p-values were inadequate across hundreds of hypotheses. FDR
   correction removed all apparent beyond-envelope discoveries.
4. Calibration/validation group labels were not necessarily independent in
   time. A 24-hour separation rule removed duplicate-time "replication."

## Result

The instrument recovered a real first-level latent relationship graph:

- 12 unique buoy-pair identities have FDR-significant pair-specific ordinary
  spectral shape.
- 8 of the 12 survive attenuation of dominant spectral bins; 6 survive in all
  three views.
- San Pedro--Santa Cruz Basin survives all three views and repeats across three
  independent dates: March 13, May 18, and May 23, 2026.
- No pair has FDR-significant temporal-concurrence evidence.
- No pair has FDR-significant evidence after complete removal of the robust
  corpus spectral envelope.

The post-hoc geography audit strongly validates that the first-level graph is
not arbitrary:

- Median distance among all 273 tested pair identities: 3,731 km.
- Median distance among the 12 discovered identities: 84.8 km.
- 11 of 12 discovered identities are within 250 km, versus 28 of 273 tested
  identities.
- A 100,000-repeat geography null matched on pair occurrence count and
  independent-time-cluster count produced zero equally strong enrichments:
  empirical `p < 1e-5`.
- The unmatched hypergeometric tail is `1.89e-11`, but the matched empirical
  result is the defensible significance statement because pair identities do
  not all have equal observation opportunity.

The lone long-distance discovered identity is Point Sur, California--Ritidian
Point, Guam. It survives all three spectral views but occurs at effectively the
same June 1 time in both group partitions, so it is not independently
replicated. It is a follow-up candidate, not a finding about long-distance
coupling.

## Interpretation

The system recovered pair-specific spectral-neighborhood structure without
being told that geography exists. Most surfaced relationships map back to
nearby buoys after the fact. That is evidence that explicit, label-agnostic
relationship discovery can recover a real organizing structure from raw noisy
data. Longitudinal persistence across independent dates is established for one
pair, not for the complete discovered graph.

The discovered structure is persistent spectral similarity, plausibly
geography and shared wave climate. It is not event propagation, moment-specific
co-movement, prediction, or evidence of deeper residual structure beneath the
ordinary domain envelope.

Complete envelope removal may be too aggressive and can erase meaningful
location-specific structure along with the obvious domain shape. The current
negative result therefore says that no beyond-envelope structure survived this
specific residualization and control ladder at this scale. It does not establish
that no deeper structure exists.

## Next

1. Add a preregistered envelope-attenuation scan between raw and complete
   envelope removal, with FDR correction across all fractions.
2. Expand pair/time coverage and require multiple independent time clusters for
   stronger relationship grades.
3. Investigate the Point Sur--Guam outlier for shared event, instrument, or
   source-data artifacts.
4. Apply the same explicit relationship/control ladder to the next domain,
   preserving the distinction between ordinary structure and structure that
   survives removal of the domain-wide known layer.
