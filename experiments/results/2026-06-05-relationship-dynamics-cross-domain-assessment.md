# Relationship-Dynamics Cross-Domain Assessment

## Purpose

The earlier observer could detect aggregate coactivation but could not identify
whether exact relationships, interchangeable roles, or graph organization had
changed. This pass added an explicit relationship-dynamics layer with three
representations:

- exact entity identity;
- structural role after optimal entity substitution; and
- permutation-invariant global graph motifs.

Each representation is tested for abrupt steps and ordered drift. Step location
selection is included in the null rescan. Null block length is estimated from
graph autocorrelation, null permutations are unique, finite permutation space is
reported, and Benjamini-Yekutieli correction covers the six tests.

## Instrument Controls

Known-answer tests pass for:

- actor substitution with preserved roles and motifs;
- matching-to-star topology change;
- gradual role and motif drift;
- raw noisy multichannel topology change; and
- a stationary negative graph sequence.

These controls establish that the implementation can distinguish identity
change from topology change under the tested synthetic conditions.

## ETT

The frozen ETT observer produced only 56 score rows and 13 relationship graphs
per 8,192-row segment. Autocorrelation selected a three-graph null block, leaving
119 distinct non-identity permutations.

Calibration showed exact, role, and motif drift. Neither later ETTh1 nor later
ETTh2 retained any corrected mechanism. The earlier positive result under a
fixed two-window block did not survive the dependence-aware null.

Grade: **needs revision / negative at this geometry**. The compression leaves
too little independent relationship history for the six-way audit.

## Synthetic PMU

Two 20-event voltage tensors produced one isolated exact-step detection in 40
events and no role, motif, or drift recurrence.

Four frequency tensors were tested under correlation, spectral alignment, and
lagged dependence. A second correction covered all 18 family/mode/metric tests
within each event. No mechanism recurred in at least half of any four-event
tensor, and none replicated across tensors.

Grade: **bad for this method/data pairing**. pmuBAGE may contain other event
structure, but the current relationship step/drift representations do not
recover a stable synthetic-grid signature.

## EEG Discovery

Six first recordings from six CHB-MIT subjects were restricted to equal one-hour
exposure. With the original six channels, exact relationship steps appeared in
4 of 6 recordings, while role and motif steps appeared in only 2 of 6.

Post-hoc attribution showed `FP2-F8` among the top three changed channels in all
four detections. Removing that channel did not remove the result. On the
five-channel development surface, exact, role, and motif steps each appeared in
3 of 6 recordings.

That five-channel surface was then frozen and tested on each subject's second
recording:

- exact step: 3 of 6;
- structural-role step: 3 of 6;
- global-motif step: 2 of 6; and
- role/motif drift: 3 of 6, but without development recurrence.

The subjects carrying the development and validation results differed except
for `chb06`. This is therefore not a stable subject fingerprint.

The signed edge-change direction did not align across runs:

- observed median signed cosine: `0.194`;
- channel-label permutation p: `0.0608`; and
- two-test BY q: `0.0912`.

The absolute pattern of which relationships changed did align:

- observed median absolute cosine: `0.7775`;
- null mean: `0.6075`;
- channel-label permutation p: `0.0060`; and
- two-test BY q: `0.0180`.

The narrow finding is that the same relationship coordinates tend to change in
magnitude across independent recordings, while the direction of change varies.
That is compatible with recurring state reorganization from different starting
states, but also with recurring acquisition or recording artifacts. The test
establishes stability of the affected coordinates, not their source. It does not
name the state or establish neurological or medical meaning.

Post-hoc seizure labels do not explain the discovery result. Most detected
recordings contain no seizure near the frozen split. One split was 102 seconds
after a short seizure; another seizure-bearing recording had no detected step.

Grade: **meh**. This is a held-out relationship-change recurrence result with a
channel-label permutation control, but it remains same-corpus, small-sample
evidence whose physical source is unresolved. Comparable established change
point and dynamic-network methods should be expected to recover at least this
much structure. This result does not justify further investment in the current
representation.

## Predictor Check

Forty generic relationship-change features were added as a separate family to
the existing leave-one-subject-out seizure predictor. The run used 6,384 rows,
507 preictal positives, and 999 within-file circular label shifts.

The raw pedestrian-plus-spectral model improved from PR-AUC `0.1457` to `0.1934`
when relationship features were added, a nominal gain of `0.0477`. The timing
null produced a mean gain of `0.0435`:

- paired empirical p: `0.420`;
- corrected BY q: `1.0`; and
- selected-feature gain: `-0.0202`, p=`0.256`, q=`1.0`.

Relationship-only prediction was the nearest nominal result at p=`0.065`, but it
also failed correction.

Grade: **disappointing as prediction, useful as a control**. The features may
encode file, subject, brain-state, or acquisition context, but this experiment
does not distinguish those alternatives and does not establish seizure-relative
timing information.

## Bottom Line

This layer is not a general-purpose latent-structure success yet.

It failed to replicate on ETT at low independent sample count and failed on
synthetic PMU events. On EEG it produced one defensible, narrow result: a
five-channel relationship-role step recurred in development and held-out
recordings, and the absolute affected-edge fingerprint survived channel-label
permutation. This is stable recurrence, not yet identified latent meaning; a
shared recording artifact remains a live alternative. The same features did
not improve time-aligned seizure prediction.

The current relationship-dynamics formulation should be retained as a tested
negative baseline, not advanced to another validation cycle. The next work
should replace the representation with stronger established multivariate change
point or dynamic-network methods and compare them under the same label-free,
held-out, permutation-controlled protocol.

## Primary Artifacts

- `spectral_forecast/relationship_dynamics.py`
- `experiments/results/2026-06-05-ett-relationship-dynamics-assessment.md`
- `experiments/results/2026-06-05-pmubage-relationship-dynamics-assessment.md`
- `experiments/results/2026-06-05-chbmit-six-subject-equal-hour-no-fp2f8-relationship-dynamics.json`
- `experiments/results/2026-06-05-chbmit-six-subject-second-hour-no-fp2f8-validation.json`
- `experiments/results/2026-06-05-chbmit-five-channel-fingerprint-alignment.json`
- `experiments/results/2026-06-05-chbmit-relationship-predictor-temporal-audit.json`
- `experiments/results/2026-06-05-chbmit-relationship-predictor-temporal-audit-summary.md`
