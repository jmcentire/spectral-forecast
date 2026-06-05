# pmuBAGE Relationship-Dynamics Assessment

## Scope

The identity, structural-role, and global-motif step/drift audit was applied to
synthetic pmuBAGE grid events without using event labels or generator classes.

Each event contains 600 samples from 24 PMU channels. Relationship graphs used
64-sample windows with stride 16. Null block length was estimated independently
from each event's graph autocorrelation, and null permutations were unique.
Benjamini-Yekutieli correction covered the six step/drift tests within each
relationship family.

## Voltage Magnitude

Two disjoint 20-event tensors were tested with correlation graphs.

| Tensor | Exact step | Role step | Motif step | Any drift |
| --- | ---: | ---: | ---: | ---: |
| voltage 0 | 1/20 | 0/20 | 0/20 | 0/20 |
| voltage 1 | 0/20 | 0/20 | 0/20 | 0/20 |

One isolated detection in 40 events is compatible with the corrected false
positive rate. There is no recurring voltage relationship-change result.

## Frequency

Four disjoint four-event tensors were tested under three declared relationship
families:

- contemporaneous correlation;
- normalized spectral-profile alignment; and
- maximum lagged dependence.

The raw family-specific results contained scattered event-level detections. A
second Benjamini-Yekutieli correction then covered all 18 family/mode/metric
tests within each event. After that correction:

- no family/mode/metric appeared in at least two events within any tensor;
- the fourth tensor had no detections at all; and
- no recurring mechanism replicated across tensors.

## Interpretation

This is a negative result for the current relationship-dynamics layer on
pmuBAGE. The method can recover known-answer synthetic graph changes, but it
does not identify a stable step or drift signature across these synthetic grid
events.

The result does not imply that pmuBAGE contains no event structure. It says the
tested structure is not consistently expressed as change in pair identity,
interchangeable node roles, or global weighted-graph motifs at this temporal
geometry. The earlier aggregate PMU observer also failed held-out support, so
the two negative results are consistent rather than contradictory.

## Artifacts

- `experiments/pmubage_relationship_dynamics.py`
- `experiments/merge_relationship_families.py`
- `experiments/results/2026-06-05-pmubage-voltage0-relationship-dynamics.json`
- `experiments/results/2026-06-05-pmubage-voltage1-relationship-dynamics.json`
- `experiments/results/2026-06-05-pmubage-frequency0-all-families-by.json`
- `experiments/results/2026-06-05-pmubage-frequency1-all-families-by.json`
- `experiments/results/2026-06-05-pmubage-frequency2-all-families-by.json`
- `experiments/results/2026-06-05-pmubage-frequency3-all-families-by.json`
