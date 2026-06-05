# Organizational-Network Directional Quality

## Subsequent Multiplicity Audit

The second-level order-null decision now uses Benjamini-Yekutieli FDR across
the four tested relationship mechanisms. Existing Email-Eu and SocioPatterns
results survive. Enron required a 999-order rerun and then retained only
coactivation.

See `2026-06-05-org-network-order-null-multiplicity-assessment.md`.

## Question

Can a label-free diagnostic distinguish several forms of multichannel
organization, and can selected organizational-network views reveal replicated
structure that is not produced by arbitrary event order?

Here, **quality** has four bounded meanings:

1. The diagnostic recovers known-answer synthetic mechanisms and does not
   invent structure in independent noise.
2. A mechanism separates from an independent per-series block-permutation null.
3. The mechanism repeats in held-apart calibration and validation segments.
4. The candidate view exceeds repeated random orderings of the same events on
   the exact candidate-selected feature names.

Quality does not mean that the structure is useful, causal, novel, or
predictive.

## Diagnostic

The new directional-quality layer tests four non-exclusive mechanisms:

- **coactivation**: anomalous components are active simultaneously more often
  than their independent rates imply;
- **exclusion**: components avoid simultaneous anomalous activity;
- **lagged succession**: one component's anomalous activity is followed by
  another's more than zero-lag coactivation explains;
- **phase offset**: continuous components align more strongly at a nonzero lag
  than at zero lag.

Each statistic aggregates a tail of pairwise effects rather than taking the
single best pair. Each is compared with an independent block-permutation null
that preserves local within-series runs while breaking cross-series alignment.

The known-answer suite passed all five cases: independent noise produced no
detected structure, and the intended strongest mechanism was recovered for
coactivation, exclusion, lagged succession, and phase offset.

## Results

The final order-null runs used 100 random reorderings. `p=0.0099` means zero of
100 random orders equaled or exceeded the candidate; it is the empirical
resolution floor, not a more precise probability. Reported z values are
standardized effect sizes against the random-order distribution, not
normal-theory p-values.

In this report, **replicated** means a mechanism separated from its block null
independently in both calibration and validation for that view. The
second-level order-null comparison statistic is the smaller of those two
observed-minus-block-null deltas. A candidate exceeds a random ordering when
that conservative delta is larger.

| Dataset and candidate view | Bins, calibration + validation | Replicated and order-sensitive mechanisms | Order-null effect sizes | Quality reading |
| --- | ---: | --- | --- | --- |
| SocioPatterns workplace, group surface ordered by group then time | 76 + 77 | coactivation, exclusion, phase offset | z=12.71, 7.51, 13.10; each p<=0.0099 | Broad selective structure |
| SocioPatterns workplace, graph surface ordered by degree | 76 + 77 | coactivation | z=20.46; p<=0.0099 | Narrow selective structure |
| Email-Eu, full surface in timestamp event order | 162 + 163 | coactivation, phase offset | z=16.81, 8.36; each p<=0.0099 | Selective structure across two mechanisms |
| Enron email simplices, full surface in timestamp event order | 28 + 29 | coactivation | z=10.63; p<=0.0099 | Narrow result; short segments limit confidence |

The order-null comparison freezes the feature names selected by each candidate
segment. The intended experimental difference is event order, not a new
high-variance channel selection for each control.

## Sober Interpretation

The diagnostic is working as an instrument on synthetic cases and is not merely
calling every mechanism present on every real view. Different views expose
different repeated mechanisms:

- SocioPatterns group order retains exclusion and phase-offset structure that
  random event order removes.
- SocioPatterns degree order retains only coactivation.
- Email-Eu timestamp order retains coactivation and phase offset, but not
  replicated exclusion or succession.
- Enron's tested equal-event timestamp view retains only coactivation.

That supports the multi-view premise: ordering and structural projection can
expose, suppress, or create measurable organization. There is no evidence here
for one universally correct representation.

Coactivation also appears in matched random-order controls, so coactivation
alone is not selective enough. The candidate views still produce larger,
replicated coactivation than 100 frozen-feature random orders, but its presence
in controls means feature construction and the event set contribute to it.

Real-time Email-Eu and Enron surfaces initially lit up all four mechanisms with
very large effects. Equal-event timestamp views and matched random-order
controls reduced that result substantially. Broad clock-time density, trends,
or bin occupancy therefore contribute to the real-time result. The narrower
equal-event findings are the more defensible current evidence.

The group-then-time SocioPatterns view uses known group labels to construct the
order. It demonstrates that an aligned order can reveal multiple mechanisms;
it is not evidence that the system autonomously discovered the group order.

## Important Layer Boundary

These runs diagnose the constructed organizational feature surfaces before the
spectral observer. They do not yet explain the earlier negative
positive-coactivation result on spectral observer scores. The two findings can
coexist: the input surface can contain simultaneous organization while the
observer's anomaly scores are mutually exclusive or phase-separated.

The direct next test is to run the same directional diagnostic on frozen
spectral-observer score matrices. That will determine whether the earlier
coactivation deficit resolves into exclusion, succession, phase offset, or
simply transformation loss.

## Artifacts

Core and runners:

- `spectral_forecast/directional.py`
- `experiments/org_network_directional_quality.py`
- `experiments/org_network_directional_compare.py`
- `experiments/org_network_directional_order_null.py`

Final second-level order-null reports:

- `2026-06-04-sociopatterns-group-directional-order-null100.json`
- `2026-06-04-sociopatterns-graph-directional-order-null100.json`
- `2026-06-04-email-eu-directional-order-null100.json`
- `2026-06-04-enron-directional-order-null100.json`
