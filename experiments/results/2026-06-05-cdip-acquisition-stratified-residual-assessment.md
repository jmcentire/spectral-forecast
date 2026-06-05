# CDIP Acquisition-Stratified Residual Assessment

Date: 2026-06-05 UTC

## Subsequent Resolution

The exact-context identifiability limit documented below was subsequently
attacked by collapsing overlapping selected triangles with identical sampling
geometry into canonical multi-buoy contexts. The stricter result survives an
exact-context additive-buoy model and dependency-preserving node-label null,
but does not establish both-half chronological replication.

See `2026-06-05-cdip-exact-context-dyadic-assessment.md`.

## Question

After the label-blind relationship instrument recovered CDIP sample-rate and
processing-family classes, does any deeper pair-profile structure remain when
those newly known layers are explicitly controlled?

This is a latent-structure discovery experiment. It is not a rogue-wave
detector, event predictor, propagation claim, or causal ocean model.

## Controls

The 437-window corpus was rerun with the previously discovered acquisition
layer treated as known:

- Corpus spectral envelopes were estimated independently inside each
  sample-rate and processing-family stratum.
- Candidate pairs were restricted to the same acquisition stratum.
- Individual pair controls used only complete alternatives from the exact same
  segment, group, window, and acquisition stratum.
- Every individual pair/view/profile hypothesis was corrected with
  arbitrary-dependence-safe Benjamini-Yekutieli FDR.
- Aggregate pair-identity persistence was tested by permuting complete local
  score-rank sets only inside exact contexts.
- A nested residual control removed exact-context intercepts plus either
  global or calibration/validation-segment additive buoy effects, then used
  within-context Freedman-Lane residual permutations.
- Aggregate controls were run both for all repeated pairs and for pairs
  recurring in at least two time clusters separated by 24 hours.
- Full, early, and late chronological scopes were corrected jointly.

The confirmatory run used 4,999 null permutations. Every testable aggregate
null contained 4,999 unique values. Empirical p-values are reported at their
finite null resolution; z-scores are standardized effect sizes, not
normal-theory p-values.

## Individual Pair Result

There were 945 candidate pair/view rows and 642 testable rows. No individual
pair survived BY correction in any waveform view or spectral representation.

This is partly an information limit. Exact matched-occurrence spaces were:

| Exact assignment space | Pair/view rows |
| ---: | ---: |
| Untestable | 303 |
| 9 assignments | 564 |
| 81 assignments | 78 |

The exact p-value floors of `1/9` and `1/81` make individual-pair FDR
discoveries unreachable on this corpus geometry. More random null repeats
cannot fix an exact finite-space resolution limit.

## Aggregate Independent-Time Result

Seventeen pair identities recur in at least two time clusters separated by 24
hours. On the canonical raw waveform's two deepest residual representations,
their aggregate pair-profile persistence survives both global and
calibration/validation-segment additive buoy effects:

| Residual representation | Buoy-effect model | Pair identities | z effect | Null exceedances | Empirical p floor | BY q |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full positive envelope attenuation | Global | 17 | 5.78 | 0 / 4,999 | 0.0002 | 0.00113 |
| Full positive envelope attenuation | Segment-specific | 17 | 7.61 | 0 / 4,999 | 0.0002 | 0.00113 |
| Signed robust envelope residual | Global | 17 | 6.37 | 0 / 4,999 | 0.0002 | 0.00113 |
| Signed robust envelope residual | Segment-specific | 17 | 7.74 | 0 / 4,999 | 0.0002 | 0.00113 |

The weaker aggregate identity-rank test independently shows the same
attenuation pattern. For the 17 independent-time pairs, ordinary raw and 25%
attenuated profiles do not survive BY correction, while every waveform view at
50%, 75%, full attenuation, and signed residual does.

The result therefore does not reduce to adjacent-window continuity, ordinary
spectral shape, the recovered acquisition classes, or stable additive buoy
signatures.

## Identifiability Failure

The strictest dyadic control cannot be run on this corpus.

Every exact usable context is a three-buoy triangle. A triangle supplies three
pair scores, and an additive three-buoy incidence model has rank three. It
therefore leaves zero residual degrees of freedom. Inside one exact context,
any apparent pair pattern can be represented perfectly as buoy effects.

This is not evidence that the aggregate signal is absent. It is a design
boundary: the current observation geometry cannot prove whether the persistent
structure is intrinsic to a pair or belongs to its local triadic context.

## Replication Boundary

The all-repeated-pair aggregate signal replicates in both chronological halves.
The stricter independent-time surface does not establish half-by-half
replication:

- Full corpus: 17 eligible independently recurring pair identities.
- Early half: 4 eligible identities.
- Late half: 2 eligible identities.

The full independent-time result survives. The early-half result does not, and
the late-half result is based on only two pairs. Treat this as insufficient
replication power, not a passed replication claim.

## Post-Hoc Attribution

A QAP-like post-hoc audit kept the 17 discovered edge identities and residual
effects fixed while permuting station geography and depth labels. Twelve
hypotheses tested signed and absolute residual effect against geographic
distance, depth difference, and mean depth in both deep representations.

No hypothesis survived BY correction. The strongest uncorrected association
was signed robust residual versus absolute depth difference:

- Spearman rho: `-0.539`
- Node-label permutation p: `0.01965`
- BY q across the attribution family: `0.732`

This audit does not establish that geography or depth are irrelevant. It says
that no simple distance/depth attribution is supported on the small
confirmatory surface after multiplicity correction.

## Interpretation

The generic instrument found a residual relationship class after its first
hidden-class discovery was explicitly removed. The surviving class is:

- spectral rather than moment-concurrent;
- visible only after substantial corpus-envelope attenuation;
- persistent across at least two dates for 17 pair identities;
- stronger than nulls preserving exact context and additive buoy effects;
- not simply attributed to distance or water depth by the tested audit.

That is evidence of persistent latent pair-profile structure in this corpus.
It is not evidence for a particular oceanographic mechanism, an individual
pair discovery, intrinsic dyadic physics, event prediction, or causation.

## What The Rigor Changed

The controls progressively narrowed the result:

1. Deep residual structure initially recovered acquisition and processing
   classes, not ocean physics.
2. Conditioning on those classes eliminated every individual pair discovery.
3. Aggregate persistence remained, but most repeated pairs were adjacent in
   time.
4. Requiring 24-hour-separated recurrence reduced the surface to 17 pairs and
   retained the full-corpus aggregate effect.
5. Additive buoy controls did not erase that aggregate effect.
6. Exact-context dyadic attribution proved mathematically unidentifiable.
7. Simple geography/depth attribution did not survive corrected post-hoc
   testing.

The remaining finding is smaller and more defensible than the original pair
graph.

## Next Decisive Experiment

The next ocean collection should break the triangle constraint. It needs
contexts containing at least four same-stratum buoys, or repeated observation
of the same pair across multiple distinct third-buoy contexts. That creates
residual degrees of freedom for a strict dyadic control.

Run that design on a disjoint date range and geography with the current
configuration frozen. The decision rule is straightforward:

- If deep independent-time residual persistence survives exact-context buoy
  effects, the evidence supports genuinely dyadic latent structure.
- If it collapses, the current result was persistent node/triad structure.

## Artifacts

- Confirmatory control:
  `2026-06-05-cdip-acquisition-stratified-residual-control-null4999.json`
- Post-hoc residual attribution:
  `2026-06-05-cdip-acquisition-stratified-residual-attribution.json`
- Prior envelope-ladder assessment:
  `2026-06-04-cdip-envelope-attenuation-independent-time-assessment.md`
