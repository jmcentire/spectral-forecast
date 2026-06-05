# CDIP Exact-Context Dyadic Assessment

Date: 2026-06-05 UTC

## Question

Does the acquisition-stratified residual relationship result survive a control
that can distinguish pair-specific structure from per-context buoy effects
while preserving the complete dependency geometry among pairs sharing buoys?

This remains a label-free latent-structure experiment. It does not identify an
oceanographic mechanism, individual significant buoy pair, event predictor, or
causal relation.

## Geometry Reconstruction

The prior 437 selected three-buoy windows contained overlapping observations at
identical timestamps. Those overlaps were collapsed into canonical raw-profile
contexts only when segment, start, end, sample count, and target sample rate
all matched.

The sampling-geometry gate matters. One timestamp contained the same buoy both
at native 2.56 Hz and downsampled to 1.28 Hz; those profiles disagreed and were
correctly kept separate.

The reconstruction produced:

- 353 canonical contexts from 437 selected triangles;
- 56 contexts containing overlapping source triangles;
- 109 duplicate raw profiles verified equal before deduplication;
- 46 same-acquisition-stratum contexts with at least four buoys and therefore
  residual degrees of freedom after exact-context additive buoy effects.

Only canonical raw spectral profiles were merged. Context-dependent residual
layers from the original triangles were not reused.

## Strongest Control

The strict control:

1. normalizes spectral envelopes within sample-rate and processing-family
   strata;
2. uses only same-stratum pairs;
3. linearly standardizes complete pair scores within each exact context;
4. removes an exact-context intercept plus additive buoy-incidence effects
   independently inside every context;
5. requires pair recurrence in at least two time clusters separated by 24
   hours;
6. requires recurrence under at least two distinct surrounding-buoy sets; and
7. randomly relabels buoy nodes inside each complete exact-context pair graph.

The node-label null preserves every local pair score, complete graph geometry,
and shared-buoy dependency. It breaks only persistent assignment of that
geometry to buoy-pair identities. Benjamini-Yekutieli FDR is applied jointly
across the declared dyadic-control family.

The confirmatory run used 4,999 unique null draws.

## Result

Thirty-three pair identities satisfy both the independent-time and
changed-neighborhood gates inside the 46 identifiable contexts.

| Residual representation | Observed | Null mean | Delta | z effect | Exceedances | Empirical p | BY q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full positive envelope attenuation | 0.3053 | 0.1736 | 0.1316 | 3.09 | 24 / 4,999 | 0.0050 | 0.0396 |
| Signed robust envelope residual | 0.4362 | 0.1829 | 0.2532 | 5.54 | 0 / 4,999 | <0.0002 | 0.0022 |

No individual pair survives BY correction. The supported result is aggregate
pair-identity persistence after strict context and dependency controls.

## Replication And Influence

The full-corpus result does not establish significance in both chronological
halves:

| Residual representation | Early half | Late half |
| --- | --- | --- |
| Full positive envelope attenuation | null | null |
| Signed robust envelope residual | null | positive, p=0.0006, BY q=0.0056 |

For the 16 strict-gate pairs present in both halves:

- full-envelope early/late residual correlation is `0.417`, with `68.8%` sign
  agreement;
- signed-residual early/late correlation is `0.693`, with `75.0%` sign
  agreement.

The signed strict-gate effect is not carried by one pair, but it is not diffuse
across all 33 pairs either:

- top five pairs contribute `50.7%` of the aggregate statistic;
- the contribution-based effective pair count is `14.4`.

These diagnostics support positive cross-half consistency while leaving
half-by-half replication unproven.

## What The Attack Changed

The stronger design did not erase the residual relationship result, but it
changed its boundary:

- The prior exact-context dyadic effect was unidentifiable because every
  original context was a triangle.
- Overlap reconstruction supplied identifiable multi-buoy contexts without
  inventing or averaging incompatible measurements.
- A dependency-preserving node-label null materially weakened the effect
  relative to unrestricted residual permutation.
- Requiring different companion sets did not remove the full-corpus result.
- The early/late split prevents a temporal-replication claim.

The defensible finding is:

> A generic, label-free spectral relationship instrument surfaces aggregate
> pair-identity structure in this selected CDIP corpus that survives recovered
> acquisition classes, exact-context additive buoy effects, changed local
> neighborhoods, and a complete-graph node-label null.

It remains unknown what the structure represents and whether it generalizes to
a disjoint date range, geography, or independently selected context surface.

## Next Decisive Ocean Test

Freeze the current representation, gates, and null family. Run them on a
disjoint date range and independently selected multi-buoy contexts.

- Survival would establish external temporal replication of the latent
  relationship class.
- Collapse would show that the current result is specific to the selected
  corpus or period.

## Artifacts

- Confirmatory result:
  `2026-06-05-cdip-exact-context-node-relabel-companion-null4999.json`
- Prior acquisition-stratified assessment:
  `2026-06-05-cdip-acquisition-stratified-residual-assessment.md`
- Runner:
  `../cdip_relationship_stratified_control.py`
