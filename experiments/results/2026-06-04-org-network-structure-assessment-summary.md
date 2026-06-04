# Organizational Network Structure Assessment

Date: 2026-06-04

## Scope

This pass added a data-first structure assessor for organizational edge streams.
It runs before expensive observer/autotune passes and asks:

> Given the canonical dataset, which structural views and enforced orders are
> plausible enough to test?

The assessor does not identify anomalies. It does not use labels as positives.
It profiles the canonical edge/simplicial data, builds cheap feature surfaces,
runs structure readiness, and grades candidate views.

## New Artifact

Code:

- `experiments/org_network_assess.py`
- `tests/test_org_network_assess.py`

Result JSON:

- `experiments/results/2026-06-04-email-eu-structure-assessment.json`
- `experiments/results/2026-06-04-sociopatterns-workplace-structure-assessment.json`
- `experiments/results/2026-06-04-enron-email-simplices-structure-assessment.json`

## Assessment Axes

The assessor keeps the canonical data stable and scores multiple possible
views:

- `temporal_total_order`: timestamp order as a sequence.
- `relationship_graph`: adjacency/topology without forcing total order.
- `relation_churn`: new, returning, persistent, and lost ties.
- `partial_order_or_hyperedge`: simultaneity and higher-order events without
  arbitrary tie-breaking.
- `group_multilayer`: group/department layers when labels are canonical and
  available at analysis time.

It also lists order candidates and controls:

- timestamp order;
- first-seen-pair order;
- degree-descending order;
- group-then-time order when group labels exist;
- stable ID / alphabetic order as artifact control;
- hash order as artifact control;
- random order as null control;
- silly proxy order, such as string length or digit frequency, as an artifact
  probe;
- timestamp-bucket partial order when simultaneity is substantial.

These scores are triage heuristics, not claims of discovery.

Important caveat: the rank order is not yet validated as predictive of
downstream anomaly usefulness. Until that validation exists, the assessor
should be read as "candidate heuristic pending validation," not as a proven
view selector.

## Email-Eu

Profile:

- edges: 332,334
- nodes: 986
- unique pairs: 24,929
- timestamps: 207,880
- timestamp resolution: 0.626
- simultaneity fraction: 0.495
- repeated-edge fraction: 0.925
- returning-pair fraction: 0.663
- degree Gini: 0.691
- canonical group labels: unavailable for the full temporal graph

View ranking:

| View | Grade | Score |
| --- | --- | ---: |
| temporal_total_order | strong | 0.807 |
| relation_churn | strong | 0.781 |
| relationship_graph | strong | 0.716 |
| partial_order_or_hyperedge | promising | 0.522 |
| group_multilayer | poor | 0.140 |

Read:

Email-Eu is not a single-view dataset. Temporal order, relationship churn, and
graph topology are all plausible. The high simultaneity fraction also says a
partial-order control is necessary; arbitrary tie-breaking could distort the
signal. Group/multilayer analysis should not be run on the full temporal graph
because the static department labels do not map to the temporal node IDs.

## SocioPatterns Workplace

Profile:

- edges: 78,249
- nodes: 217
- unique pairs: 4,274
- timestamps: 18,488
- timestamp resolution: 0.236
- simultaneity fraction: 0.962
- repeated-edge fraction: 0.945
- returning-pair fraction: 0.669
- degree Gini: 0.375
- label coverage: 1.000
- groups: 12
- cross-group fraction: 0.237

View ranking:

| View | Grade | Score |
| --- | --- | ---: |
| group_multilayer | strong | 0.930 |
| relation_churn | strong | 0.786 |
| partial_order_or_hyperedge | promising | 0.687 |
| relationship_graph | promising | 0.660 |
| temporal_total_order | promising | 0.617 |

Read:

SocioPatterns should not be treated primarily as a flat time series. Its
strongest candidate view is group/multilayer, followed by relation churn.
Because simultaneity is extremely high, timestamp-bucket partial order matters:
forcing a total order inside contact intervals is likely to add arbitrary
structure. The correct first expensive run should compare group-multilayer and
relation-churn observers against timestamp, bucketed partial-order, random, and
silly-order controls.

## Enron Email Simplices

Profile:

- projected edges: 28,867
- nodes: 143
- unique pairs: 1,800
- timestamps: 10,366
- timestamp resolution: 0.359
- simultaneity fraction: 0.730
- repeated-edge fraction: 0.938
- returning-pair fraction: 0.721
- degree Gini: 0.562
- canonical group labels: unavailable in this projected dataset

View ranking:

| View | Grade | Score |
| --- | --- | ---: |
| relation_churn | strong | 0.918 |
| partial_order_or_hyperedge | strong | 0.864 |
| relationship_graph | strong | 0.807 |
| temporal_total_order | promising | 0.694 |
| group_multilayer | weak | 0.200 |

Read:

Enron simplices should be treated as higher-order co-participation first, not
as a simple timestamped edge list. Relation churn, partial order/hyperedge, and
graph topology are the first-class candidates. Time order is plausible but
secondary. Group/multilayer is weak because this dataset does not provide
canonical role/group labels in the current adapter.

## Honest Interpretation

This assessment explains the earlier mixed results better:

- The organizational datasets are not empty.
- The first spectral-residual sequence observer was probably the wrong
  instrument for several of these views.
- SocioPatterns wants group/multilayer and partial-order handling.
- Enron wants hyperedge/co-participation and relation churn.
- Email-Eu needs a comparison across temporal, graph, relation-churn, and
  partial-order views.

The next implementation should not be "run more of the same." It should add a
view runner that:

1. Starts from canonical events/entities.
2. Generates several view transforms.
3. Preserves a mapping from every transformed component back to canon.
4. Runs readiness and view-specific nulls.
5. Compares view sensitivity.
6. Uses artifact orders, random orders, and silly proxy orders as controls.

If a signal only survives under an arbitrary or silly order, treat it as an
artifact until a canonical explanation is found. If a signal survives across a
view and maps cleanly back to canonical entities/events, that is a much better
candidate for downstream feature detection or prediction.
