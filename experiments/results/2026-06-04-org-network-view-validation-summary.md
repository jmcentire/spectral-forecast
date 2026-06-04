# Organizational Network View Validation

Date: 2026-06-04

## Scope

This pass resumed from the organizational structure assessor and view probe.
It added the missing bridge from preflight structure detection to defensible
downstream analysis:

- the probe can now explicitly refuse all candidate views;
- downstream runs can select a feature family and enforced event order;
- transformed bins map back to original events, entities, pairs, and groups;
- repeated observer matrices are reused instead of recomputed for top windows;
- block-permute nulls automatically shrink their block size when the requested
  size would leave fewer than two blocks.

The core question was:

> Does a view that contains detectable latent structure also provide
> exploitable cross-series anomaly coherence to the current
> spectral/stigmergic observer?

For the selected SocioPatterns views, the answer depends on what counts as
coherence. They do not provide the positive coactivation surplus that the
current stigmergic objective rewards. They do show coactivation deficits
relative to the null, which may encode exclusion, succession, phase separation,
or an artifact of the enforced order. This pass does not distinguish those
explanations.

## Implementation

Updated:

- `experiments/org_network_view_probe.py`
- `experiments/org_network_autotune.py`
- `tests/test_org_network_view_probe.py`
- `tests/test_org_network_autotune.py`

### Probe refusal state

The view probe now emits a global recommendation:

- `broad_candidate_set`
- `selective_candidate_set`
- `controls_too_strong`
- `none_of_these`

The regenerated reports classify:

- Email-Eu: `broad_candidate_set`
- SocioPatterns Workplace: `selective_candidate_set`
- Enron Email Simplices: `broad_candidate_set`

For SocioPatterns, only `group_multilayer` and `relationship_graph` earned a
larger run. `partial_order_or_hyperedge` was rejected because its control
dominated. `relation_churn` and `temporal_total_order` remain unresolved.

### View-specific downstream runner

The organizational autotune runner now accepts:

- `--surface-mode full|relation|graph|group`
- `--event-order real-time|timestamp|timestamp_bucket_partial_order|first_seen_pair|degree_descending|group_then_time|stable_id_or_alphabetic|hash_order|random_order|silly_proxy_order`
- `--event-bin-size`

For non-real-time orders, the runner retains an analysis-edge to canonical-edge
mapping. Top windows report:

- analysis and canonical edge indices;
- original timestamps;
- sampled original events;
- top entities, pairs, and groups;
- anchor-bin and adaptive-history context;
- cross-group event counts.

### Null validity guard

The first selected-view runs exposed a degenerate control:

- ordered surfaces had four observation anchors;
- requested block-permute size was eight;
- each series therefore had one block;
- all null repeats were identical.

The runner now bounds effective block size to at most half the anchor count.
The repeated runs used effective block size `2` and produced 98-100 unique
null totals out of 100 repeats.

The shared autotune null summary now reports upper-tail, lower-tail, and
two-sided empirical p-values. The positive-coactivation tuner still accepts
only surplus coherence, but it no longer hides a potentially informative
deficit behind `p_ge` near one.

The invalid pre-fix result artifacts were removed from the working tree.

## Canonical Back-Mapping Validation

Artifact:

- `experiments/results/2026-06-04-sociopatterns-canonical-context-validation.json`

The full real-time SocioPatterns surface produced top windows with bounded
canonical context. Calibration had positive observed-minus-null lift but was
not accepted:

- calibration: `z=2.25`, delta `6806.12`, empirical `p=0.04`, rejected by the
  full quality gate;
- validation: `z=1.32`, delta `1204.38`, empirical `p=0.12`, accepted by the
  current quality gate but weak and not statistically decisive.

This run validates the back-mapping machinery. It does not establish a finding.

## Selected SocioPatterns Views

### Group/multilayer under group-then-time order

Artifact:

- `experiments/results/2026-06-04-sociopatterns-group-then-time-autotune-expanded.json`

Preflight structure remained strong:

- calibration readiness score: `0.700`
- validation readiness score: `0.700`

Positive coactivation failed against a valid block-permute null, while the
validation deficit was unlikely under that null:

- calibration: deficit `-152.37`, `z=-2.02`, lower-tail `p=0.0396`,
  two-sided `p=0.0792`
- validation: deficit `-524.70`, `z=-2.57`, lower-tail `p=0.0099`,
  two-sided `p=0.0198`
- 100 unique null totals in both segments

### Graph surface under degree-descending order

Artifact:

- `experiments/results/2026-06-04-sociopatterns-degree-graph-autotune-expanded.json`

Preflight structure remained strong:

- calibration readiness score: `1.000`
- validation readiness score: `0.700`

Positive coactivation also failed. The calibration deficit cleared a two-sided
0.05 threshold; validation was directionally similar but weaker:

- calibration: deficit `-5.92`, `z=-2.47`, lower-tail `p=0.0198`,
  two-sided `p=0.0396`
- validation: deficit `-9.49`, `z=-1.98`, lower-tail `p=0.0396`,
  two-sided `p=0.0792`
- 100 unique calibration null totals and 98 unique validation null totals

## Honest Interpretation

The assessor and probe found structural concentration in the selected
SocioPatterns views. That structure did not become a positive cross-series
coactivation surplus under the current spectral/stigmergic objective.

This distinction matters:

- `structure_readiness` asks whether a surface has non-random organization;
- the view probe asks whether a candidate view exceeds artifact controls;
- autotune currently rewards whether the observer produces simultaneous
  positive emissions above a domain-preserving null.

Passing the first two does not imply passing the third.

Observed emissions were below the independently block-permuted null. That is
evidence against positive coactivation for these views and configurations. It
is not evidence that the underlying structure is absent, nor does it prove that
serialization is the cause. The deficit may mean the enforced views organize
activity into mutually exclusive or sequential blocks, which the current
positive-resonance objective treats as a failure.

## Next

1. Do not scale the current SocioPatterns `group_then_time` or
   `degree_descending` positive-coactivation objective.
2. Diagnose the deficit directly: test exclusion, succession, lagged
   activation, and phase-offset statistics against the same null.
3. Use synthetic graph/group structures with known answers to determine whether
   the deficit comes from real organization or from the enforced order.
4. Treat graph-native and multilayer-native observers as candidates to test,
   not as conclusions already established.
5. Keep the assessor and view probe as preflight filters.
6. Preserve canonical back-mapping and valid-null checks as hard requirements
   for every future view.
