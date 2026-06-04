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
- `experiments/results/2026-06-04-email-eu-relation-only-autotune-permute-smoke.json`
- `experiments/results/2026-06-04-enron-email-simplices-relation-autotune-permute-smoke.json`
- `experiments/results/2026-06-04-enron-email-simplices-relation-only-autotune-permute-smoke.json`
- `experiments/results/2026-06-04-sociopatterns-workplace-org-network-autotune-smoke.json`
- `experiments/results/2026-06-04-sociopatterns-workplace-org-network-autotune-permute-smoke.json`
- `experiments/results/2026-06-04-sociopatterns-workplace-relation-only-autotune-permute-smoke.json`

Data sources:

- SNAP Email-Eu temporal: https://snap.stanford.edu/data/email-Eu-core-temporal.html
- SocioPatterns workplace contacts: https://sociopatterns.org/datasets/test/
- Cornell Enron temporal higher-order email: https://www.cs.cornell.edu/~arb/data/email-Enron/

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

## Relation-Change Follow-Up

After the volume/concentration surface failed, the adapter was expanded with
relational-change features:

- novel pairs;
- returning pairs;
- persistent and lost pairs;
- pair churn and pair Jaccard to the previous bin;
- active-node churn;
- top-node neighbor count, new neighbors, lost neighbors, and neighbor churn;
- group internal/external pair churn where valid group labels exist.

The script also gained `--relation-only`, which drops plain activity/count
features and tests only relationship change.

### Email-Eu Relation-Only

| Segment | Accepted | Delta | z effect | Empirical p_ge | Unique null totals |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | no | +208.927 | 0.55 | 0.3137 | 50 |
| Validation | no | -8845.283 | -12.38 | 1.0000 | 50 |

Read: weak calibration positivity did not transfer. This is a bad sign for the
current Email-Eu relation-only surface.

### SocioPatterns Relation-Only

| Segment | Accepted | Delta | z effect | Empirical p_ge | Unique null totals |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | no | +6.956 | 0.12 | 0.3922 | 50 |
| Validation | no | -107.083 | -0.28 | 0.6667 | 50 |

Read: relation-only features are more semantically plausible than volume, but
they still do not show a stable surface on this bounded workplace slice.

### Cornell Enron Email Simplices

Cornell's `email-Enron` data is a temporal higher-order dataset: each simplex
is the sender plus recipients among core Enron employees. This adapter projects
each simplex to unordered co-participation pairs. That means the run tests
changing co-participation structure, not directed sender-to-recipient flow.

Mixed activity + relation features:

| Segment | Accepted | Delta | z effect | Empirical p_ge | Unique null totals |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | no | -2131.165 | -3.33 | 1.0000 | 50 |
| Validation | no | +252.972 | 0.39 | 0.4118 | 50 |

Relation-only features:

| Segment | Accepted | Delta | z effect | Empirical p_ge | Unique null totals |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | no | -1467.498 | -3.05 | 1.0000 | 50 |
| Validation | no | +244.026 | 0.39 | 0.4706 | 50 |

Read: Enron relation-change/co-participation did not produce an accepted
surface. The validation slice is weakly positive, but it is nowhere near enough
to trust, especially because calibration was below null.

## Current Honest Read

The organizational-network adapter is operational, and the relation-change
features are the right direction conceptually. But the current organizational
network experiments are negative:

- volume/concentration features failed;
- relation-change features also failed;
- mild calibration positives did not transfer;
- Enron co-participation churn did not beat nulls in calibration;
- SocioPatterns block-permutation showed one degenerate null case, which the
  unique-null-total instrumentation caught.

This does not prove there is no exploitable organizational structure. It does
show that these first edge-stream representations are not enough. The method is
behaving conservatively, which is good, but the organizational result itself is
not yet promising.

## Next Adapter Step

The next feature surface should move from generic edge churn to richer
organizational structure:

1. Temporal motifs, such as repeated A-to-B contact, reciprocal reply latency
   buckets, and triadic closure bursts.
2. Detrended or difference features, so the detector sees change in structure
   rather than raw meeting/email volume.
3. Role-aware features for datasets that actually have roles, titles, or
   reliable groups.
4. Thread/conversation features where message text or subject lines are
   available.
5. Explicit event-time attribution after discovery, especially Enron's public
   crisis timeline, but not as a tuning target.

The next dataset should probably not be another bare temporal edge list. Use a
dataset with richer semantics, such as GitHub project activity or MAEC earnings
calls, because this method may need a more meaningful component vocabulary than
"edge happened" to find organizational structure.
