# Organizational Network View Probe

Date: 2026-06-04

## Scope

This pass tested whether the structure assessor can do useful work before a
larger analysis run. The new probe turns assessor-ranked views into cheap
candidate/control surfaces and runs structure readiness on each surface.

This is not anomaly detection and not prediction. It is a preflight filter:

> If a candidate view cannot beat random, stable-ID, hash, or nonsense-order
> controls at this stage, it has not earned an expensive downstream run.

## New Artifact

Code:

- `experiments/org_network_view_probe.py`
- `tests/test_org_network_view_probe.py`

Result JSON:

- `experiments/results/2026-06-04-email-eu-view-probe.json`
- `experiments/results/2026-06-04-sociopatterns-workplace-view-probe.json`
- `experiments/results/2026-06-04-enron-email-simplices-view-probe.json`

## Method

The probe starts from the canonical event stream, then materializes candidate
views and controls:

- real timestamp-binned surfaces;
- equal-event chronological surfaces;
- relation-churn surfaces ordered by first seen pair or timestamp;
- graph/topology surfaces ordered by degree;
- group/multilayer surfaces ordered by group then time when labels exist;
- random order null controls;
- stable ID and hash artifact controls;
- nonsense/proxy order artifact probes.

The underlying readiness metric remains bounded and conservative. For comparing
candidate surfaces against controls, the probe adds an unsaturated
`probe_score` based on positive log z effects from covariance, pattern, and
temporal-memory readiness. This was necessary because the bounded
`structure_score` saturated on these datasets and could not distinguish strong
candidate surfaces from strong controls.

The Enron simplicial timestamps are in a much larger unit scale than the other
two datasets. The probe now guards real-time binning and automatically widens
real-time bins when a default bin size would create too many bins. In this run,
Enron would have produced 1,307,177 real-time bins at `86400`; the probe widened
the bin to `56,498,255.63` to bound real-time surfaces at 2,000 bins.

## Results

### Email-Eu

All assessed candidate families separated from controls:

| View | Verdict | Best Candidate | Best Control | Gap |
| --- | --- | --- | --- | ---: |
| temporal_total_order | candidate separates | temporal_real_time 5.368 | temporal_silly_order_probe 3.803 | 1.565 |
| partial_order_or_hyperedge | candidate separates | partial_real_time_bucket 5.304 | partial_random_event_control 3.799 | 1.505 |
| relationship_graph | candidate separates | graph_degree_descending 5.456 | graph_hash_control 4.330 | 1.126 |
| relation_churn | candidate separates | relation_first_seen_pair 4.137 | relation_random_control 3.836 | 0.300 |

Read:

Email-Eu has exploitable structure under multiple views. The relation-churn
margin is narrow, so it should be treated as plausible but less decisive than
real-time/partial-order and graph-topology surfaces. Controls are still strong,
which means the dataset has broad structural concentration; the useful signal
is the candidate-over-control gap, not raw readiness.

### SocioPatterns Workplace

The probe is selective here:

| View | Verdict | Best Candidate | Best Control | Gap |
| --- | --- | --- | --- | ---: |
| group_multilayer | candidate separates | group_then_time 4.031 | group_silly_order_probe 3.633 | 0.397 |
| relationship_graph | candidate separates | graph_degree_descending 3.942 | graph_hash_control 3.250 | 0.693 |
| temporal_total_order | ambiguous | temporal_event_timestamp 3.678 | temporal_silly_order_probe 3.765 | -0.087 |
| relation_churn | ambiguous | relation_timestamp 3.295 | relation_random_control 3.373 | -0.078 |
| partial_order_or_hyperedge | control dominates | partial_real_time_bucket 2.621 | partial_random_event_control 3.323 | -0.701 |

Read:

SocioPatterns should not get a generic "more of the same" run. The assessor's
group/multilayer instinct held up, and graph/topology also looks worth testing.
Flat temporal order, relation churn, and partial-order handling did not separate
cleanly from controls in this probe. The partial-order result is especially
useful because it says high simultaneity alone is not enough; preserving
simultaneity is not automatically the right enforced structure.

### Enron Email Simplices

All assessed candidate families separated from controls after bounding real-time
bins:

| View | Verdict | Best Candidate | Best Control | Gap |
| --- | --- | --- | --- | ---: |
| partial_order_or_hyperedge | candidate separates | partial_real_time_bucket 5.304 | partial_random_event_control 2.705 | 2.599 |
| temporal_total_order | candidate separates | temporal_real_time 5.343 | temporal_event_random_control 2.790 | 2.554 |
| relationship_graph | candidate separates | graph_degree_descending 3.351 | graph_stable_id_control 2.531 | 0.821 |
| relation_churn | candidate separates | relation_timestamp 2.996 | relation_silly_order_probe 2.698 | 0.299 |

Read:

Enron's strongest signal is in real-time/higher-order bucket structure, with
graph topology also meaningful. Relation churn separates, but only barely. This
supports the earlier assessment that Enron should be handled as
co-participation/hyperedge-like data first, with relation churn as a secondary
surface.

## Honest Take

This is a good sign, but not a victory lap.

The useful result is not "everything has structure." Controls also have strong
readiness in these datasets, especially Email-Eu and SocioPatterns. The useful
result is that the probe can distinguish cases where the candidate structure
beats controls from cases where it does not.

Practical next moves:

1. Use Email-Eu and Enron for larger downstream runs on the separating views.
2. For SocioPatterns, run only group/multilayer and graph/topology first.
3. Do not spend large compute on SocioPatterns partial-order, flat temporal, or
   relation-churn views until a better transform is defined.
4. Add a "none of these" or "controls too strong" assessor state so the system
   can refuse to recommend a view when candidate/control gaps are too small.
5. Add canonical back-mapping for top probe windows before naming any surfaced
   organizational behavior.

The assessor-plus-probe pair now does something defensible: it narrows the
search space before expensive analysis and exposes when an enforced structure
looks like an artifact.
