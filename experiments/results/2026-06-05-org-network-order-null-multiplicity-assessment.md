# Organizational Order-Null Multiplicity Assessment

Date: 2026-06-05 UTC

## Question

Do the organizational directional-quality results survive correction across
the four relationship mechanisms tested in each dataset?

The prior order-null runner required replication in calibration and validation
and comparison against repeated random event orders, but it used uncorrected
empirical p-values for the final mechanism decision. The decision rule now uses
Benjamini-Yekutieli arbitrary-dependence-safe FDR across coactivation,
exclusion, lagged succession, and phase offset.

## Reassessment Of Existing 100-Order Runs

| Dataset/view | Replicated mechanisms surviving BY | Read |
| --- | --- | --- |
| Email-Eu, timestamp order | coactivation, phase offset | survives, each p=0.0099 and BY q=0.0275 |
| SocioPatterns, group-then-time order | coactivation, exclusion, phase offset | survives, each p=0.0099 and BY q=0.0275 |
| SocioPatterns, degree graph order | coactivation | survives, p=0.0099 and BY q=0.0413 |
| Enron simplices, timestamp order | none at 100 repeats | coactivation p=0.0099 but BY q=0.0825 |

The Enron result was resolution-limited rather than disproven: its single
replicated mechanism could not survive conservative correction at a
`1/101` empirical p-floor.

## Enron Higher-Resolution Result

The Enron timestamp-order result was rerun against 999 random event orders with
the candidate-selected series and inner block-null procedure frozen.

| Mechanism | Candidate min delta | Random-order mean | z effect size | Exceedances | Empirical p | BY q | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Coactivation | 0.03504 | 0.00848 | 10.13 | 0 / 999 | 0.0010 | 0.0083 | survives |
| Exclusion | -0.01697 | -0.00525 | -5.30 | 999 / 999 | 1.000 | 1.000 | null |
| Lagged succession | -0.01527 | -0.00692 | -3.73 | 995 / 999 | 0.996 | 1.000 | null |
| Phase offset | -0.14462 | -0.12464 | -0.84 | 807 / 999 | 0.808 | 1.000 | null |

Enron therefore supports a narrow order-sensitive coactivation result, not a
broad organizational mechanism claim.

## Interpretation

The corrected cross-dataset pattern is selective:

- SocioPatterns group order preserves three generic relationship mechanisms.
- SocioPatterns graph order preserves only coactivation.
- Email-Eu timestamp order preserves coactivation and phase offset.
- Enron timestamp order preserves only coactivation.

These are statistical organization results. They do not identify leadership
quality, crisis causation, organizational health, or predictive utility.

## Artifacts

- `2026-06-05-enron-directional-order-null999-by.json`
- `2026-06-04-email-eu-directional-order-null100.json`
- `2026-06-04-sociopatterns-group-directional-order-null100.json`
- `2026-06-04-sociopatterns-graph-directional-order-null100.json`
- Runner: `../org_network_directional_order_null.py`
