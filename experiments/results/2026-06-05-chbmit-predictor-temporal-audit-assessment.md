# CHB-MIT Predictor Temporal Audit Assessment

## Question

Do the previously discovered generic latent/stigmergic features contain
time-local information about held-out preictal windows, or do their apparent
predictive gains merely correlate with subject or recording context?

The predictor is a supervised measurement instrument. It is not the latent
structure discovery system and this is not a clinical prediction claim.

## Control

The audit freezes every leave-one-subject-out model, feature selection decision,
and held-out prediction. It then circularly shifts labels within each held-out
EDF file 999 times.

This preserves:

- held-out subject and file membership;
- per-file prevalence; and
- contiguous positive-label blocks.

It breaks alignment between the fixed prediction and the event-relative timing.
Benjamini-Yekutieli correction is applied across all absolute-model and paired
feature-contribution tests.

## Result

No absolute model and no latent-feature contribution survives the corrected
temporal audit.

| Test | Observed PR-AUC or delta | Null mean | Empirical p | BY q |
| --- | ---: | ---: | ---: | ---: |
| latent-only selected | 0.178 | 0.143 | 0.091 | 0.715 |
| spectral + latent selected | 0.158 | 0.133 | 0.068 | 0.715 |
| raw latent addition | -0.010 | -0.009 | 0.586 | 1.000 |
| selected latent addition | +0.032 | +0.005 | 0.049 | 0.715 |
| latent-only versus pedestrian + spectral | +0.069 | +0.053 | 0.133 | 0.825 |

The selected latent addition has a nominal `p=0.049`, but that value does not
survive correction across the audit. Raw latent addition is slightly harmful.

## Interpretation

The current three-subject experiment does not establish that the generic
latent/stigmergic features encode time-local precursor information. The
uncorrected model rankings and nominal selected-feature gain are compatible
with file/subject context and chance across the tested surfaces.

This is a useful negative result:

- the predictor harness can now distinguish held-out correlation from
  time-aligned predictive information;
- the prior statement that latent features contain usable predictive
  information is superseded for this experiment; and
- future predictive tests need more subjects, an externally frozen feature
  instrument, and the same timing-preserving null discipline.

## Boundary

The audit does not show that the label-free features contain no structure. It
shows that this supervised harness has not demonstrated that the structure is
predictive of the chosen preictal labels at the chosen temporal resolution.

Primary artifacts:

- `experiments/results/2026-06-05-chbmit-predictor-temporal-shift-null999.json`
- `experiments/results/2026-06-05-chbmit-predictor-temporal-shift-null999-summary.md`
- `experiments/chbmit_predictor_ablation.py`
