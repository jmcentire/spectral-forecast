# ETT Relationship-Dynamics Assessment

## Question

Does the previously validated aggregate ETT observer contain recurring changes
in exact channel relationships, interchangeable structural roles, or global
graph motifs?

The test used no target labels or domain quantities. The observer configuration
was frozen from the earlier ETTh1 calibration, then applied unchanged to:

- the first 8,192 ETTh1 rows;
- the next 8,192 ETTh1 rows; and
- rows 8,192 through 16,383 of ETTh2.

## Method

Each segment produced 56 channel-score observations. Correlation graphs were
built over 32 observations with stride 2, leaving 13 graph windows per segment.

Six preregistered tests were evaluated:

- step change in exact channel identity;
- step change after optimal role substitution;
- step change in permutation-invariant global motifs;
- ordered drift in each of the same three representations.

One exact-identity CUSUM selected the step location. All three step metrics were
evaluated at that location against null rescans. Drift was measured as ordered
distance from the frozen early graph. Benjamini-Yekutieli correction covered all
six tests within each segment.

Null block length was estimated from graph autocorrelation. It selected three
graph windows, leaving five temporal blocks and only 119 non-identity block
permutations. All 119 were enumerated exactly; the empirical p-value floor was
therefore 1/120, not 1/1000.

## Result

| Segment | Corrected mechanisms | Nearest held-out result |
| --- | --- | --- |
| ETTh1 calibration | exact, role, and motif drift | calibration only |
| later ETTh1 | none | exact drift p=0.0083, BY q=0.1225 |
| later ETTh2 | none | exact drift p=0.0083, BY q=0.1225 |

An earlier fixed two-window block run appeared to retain exact-identity drift in
later ETTh1 and placed later ETTh2 just outside correction. That interpretation
does not survive the data-derived dependence block and is not the controlling
result.

## Interpretation

The role and motif mechanisms do not replicate. Exact channel relationships
move monotonically away from each segment's early graph, but 13 graph windows
provide too few dependence-preserving temporal arrangements to distinguish that
ordering after six-test correction.

This is a negative validation result, not evidence that ETT has no relational
structure. It establishes that the current observer geometry compresses an
8,192-row segment to too little independent relationship history for a rigorous
six-way step/drift audit.

The useful engineering result is the null accounting: null draws are now unique,
the finite permutation space is reported, and temporal block length is estimated
from the observed graph dependence. A requested repeat count can no longer imply
resolution the data cannot supply.

## Artifacts

- `experiments/csv_relationship_dynamics.py`
- `experiments/results/2026-06-05-etth1-calibration-relationship-dynamics-auto-block999.json`
- `experiments/results/2026-06-05-etth1-heldout-relationship-dynamics-auto-block999.json`
- `experiments/results/2026-06-05-etth2-external-relationship-dynamics-auto-block999.json`
