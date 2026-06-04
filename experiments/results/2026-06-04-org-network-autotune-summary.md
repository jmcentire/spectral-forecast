# Organizational Network Autotune Smoke Results

Date: 2026-06-04

## Scope

This pass added a temporal organizational-network adapter and ran bounded smoke
tests on the first two organizational datasets:

- SNAP `email-Eu-core-temporal`
- SocioPatterns workplace contacts, 2nd deployment

The adapter converts timestamped edges into synchronized numeric series and
then uses the existing label-free spectral+stigmergy autotune core. The first
feature family is intentionally plain:

- global edge volume;
- unique source/target/active-node counts;
- source and target concentration;
- reciprocity;
- top-node send/receive/total activity;
- when labels are valid, group activity/internal/external contact series.

Labels are not used as positives. For SocioPatterns, department metadata is
used to define group-level series. For Email-Eu, department labels are not used:
SNAP notes that the static department-label graph and the temporal graph do not
share node IDs. The four Email-Eu department temporal files should be treated
as separate cohorts, not labels on the full temporal graph.

## New Artifacts

Code:

- `experiments/org_network_autotune.py`
- `tests/test_org_network_autotune.py`

Result JSON:

- `experiments/results/2026-06-04-email-eu-org-network-autotune-smoke.json`
- `experiments/results/2026-06-04-email-eu-org-network-autotune-permute-smoke.json`
- `experiments/results/2026-06-04-sociopatterns-workplace-org-network-autotune-smoke.json`
- `experiments/results/2026-06-04-sociopatterns-workplace-org-network-autotune-permute-smoke.json`

Data sources:

- SNAP Email-Eu temporal: https://snap.stanford.edu/data/email-Eu-core-temporal.html
- SocioPatterns workplace contacts: https://sociopatterns.org/datasets/test/

## Email-Eu Full Temporal Graph

Run shape:

- edges: 332,334
- bins: 804 daily bins
- calibration/validation split: 402 / 402 bins
- series: 37
- features: global and top-node only
- top nodes: `987`, `168`, `629`, `586`, `356`, `178`, `912`, `915`, `98`, `746`

### Block-Permutation Null

| Segment | Accepted | Delta | z effect | Empirical p_ge | Unique null totals |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | no | -66.349 | -0.74 | 0.7451 | 50 |
| Validation | no | -1993.185 | -5.31 | 1.0000 | 50 |

### Permutation Null

| Segment | Accepted | Delta | z effect | Empirical p_ge | Unique null totals |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | no | -120.722 | -1.14 | 0.8235 | 50 |
| Validation | no | -2630.898 | -11.28 | 1.0000 | 50 |

Interpretation:

The first Email-Eu feature surface is rejected. The largest observed windows
are dominated by global volume and active-node count bursts, but those bursts
do not exceed timing nulls. In validation they are substantially weaker than
the null. This does not mean Email-Eu has no organizational structure. It means
this plain volume/top-node feature family is not yet the right surface for this
detector.

## SocioPatterns Workplace

Run shape:

- edges loaded: 78,249
- labels loaded: 232 person-to-department mappings
- feature surface: global and department-level contact activity
- selected departments: `DMI`, `DMCT`, `DST`, `DCAR`, `DSE`, `DISQ`, `SRH`,
  `SFLE`

### Block-Permutation Null

Small bounded run:

- bins: 180 ten-minute bins
- included edges: 17,140
- calibration/validation split: 90 / 90 bins
- series: 33

| Segment | Accepted | Delta | z effect | Empirical p_ge | Unique null totals |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | no | ~0.000 | -0.98 | 1.0000 | 1 |
| Validation | no | ~0.000 | 0.98 | 1.0000 | 1 |

The block null was degenerate on this bounded slice: all null totals collapsed
to a single value. Treat this run as a null-design diagnostic, not as evidence
for or against structure.

### Permutation Null

Non-degenerate bounded run:

- bins: 240 ten-minute bins
- included edges: 20,416
- calibration/validation split: 120 / 120 bins
- series: 33

| Segment | Accepted | Delta | z effect | Empirical p_ge | Unique null totals |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | no | -305.106 | -2.79 | 1.0000 | 50 |
| Validation | no | -886.802 | -6.92 | 1.0000 | 50 |

Interpretation:

The first SocioPatterns feature surface is also rejected under the
non-degenerate permutation null. The strongest windows are again mostly global
activity and concentration bursts. Those are real workplace rhythms, but this
surface does not produce cross-series residual structure stronger than the
null.

## Current Read

The adapter works, but the first organizational feature family did not surface
accepted latent structure. That is a useful negative control:

- the pipeline did not accept ordinary communication volume as signal;
- the validation split rejected the frozen selected configurations;
- unique-null-total reporting caught a degenerate block null on the small
  SocioPatterns slice;
- the result is consistent across Email-Eu and SocioPatterns for this feature
  family.

The likely issue is not "organizational data has no structure." It is that
simple volume/concentration series are too pedestrian and too schedule-driven.
The nulls can reproduce or exceed the same aggregate structure.

## Next Adapter Step

The next feature surface should move from volume to relational change:

1. Novel dyads per bin and returning-dyad ratio.
2. Edge persistence and churn.
3. Ego-network turnover for top nodes.
4. Cross-group bridge churn where valid group labels exist.
5. Temporal motif counts, such as repeated A-to-B contact, reciprocal reply
   latency buckets, and triadic closure bursts.
6. Detrended or difference features, so the detector sees change in structure
   rather than raw meeting/email volume.

Then rerun Email-Eu and SocioPatterns before moving to Enron. If relational
change also fails, that is a stronger boundary on this method for
organizational communication streams. If it succeeds, Enron becomes the right
third dataset because it has a known external crisis timeline for post-hoc
attribution.
