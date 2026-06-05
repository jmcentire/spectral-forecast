# Established Change-Point Benchmark

Date: 2026-06-05

## Question

Does a maintained, established multivariate change-point method outperform the
retired relationship-dynamics representation when the objective is stated
plainly: locate an abrupt distributional transition?

The benchmark uses
[`changeforest`](https://jmlr.org/papers/v24/22-0512.html), which reports strong
published simulation performance for nonparametric multivariate change-point
detection. Detection used no event labels. Known transition locations were
revealed only for scoring.

## Controls

One hundred independent trials were generated for each synthetic condition:

- stationary independent Gaussian observations;
- stationary AR(1) observations with coefficient `0.85`;
- a mean shift;
- a covariance/topology shift; and
- a same-scale Gaussian-to-heavy-tail distribution shift.

Two package methods were frozen: random forest and k-nearest neighbors. Each was
run on raw observations and on per-channel AR(1) residuals. Model-selection used
99 package permutations at alpha `0.05` and 50 random-forest estimators.

The real-data-shaped test used all 84 frequency and 620 voltage events in local
pmuBAGE. The dataset fixes each event onset at sample 300 of 600. The source
paper states that each window is 20 seconds and the event begins at 10 seconds.

## Synthetic Result

Random-forest changeforest was strong for its declared objective:

| Condition | Raw detection / root localization | AR(1) residual detection / root localization |
|---|---:|---:|
| stationary iid | 4% false positive | 7% false positive |
| stationary AR(1) | 100% false positive | 5% false positive |
| mean shift | 100% / 100% | 100% / 100% |
| covariance shift | 100% / 100% | 100% / 100% |
| heavy-tail shift | 99% / 90% | 100% / 92% |

The raw permutation test is invalid under strong serial dependence. Simple
AR(1) residualization restored the stationary AR(1) false-positive rate to its
nominal neighborhood without losing the injected mean or covariance changes.

KNN detected mean and covariance shifts but largely missed the heavy-tail
change. Method family matters; no single nonparametric score dominated every
alternative.

## pmuBAGE Result

Using the package significance decision directly was unacceptable. Raw random
forest localized the event with its primary split in every event, but also
declared a change in every 300-sample pre-event baseline. It recursively emitted
a median of 65 or 66 splits per full event. That is over-segmentation, not a
usable anomaly gate.

| Corpus | Method/view | Primary split within 15 samples | Pre-event package false positive |
|---|---|---:|---:|
| frequency, 84 events | random forest, raw | 100.0% | 100.0% |
| frequency, 84 events | random forest, AR(1) residual | 90.5% | 89.3% |
| voltage, 620 events | random forest, raw | 100.0% | 100.0% |
| voltage, 620 events | random forest, AR(1) residual | 97.9% | 92.6% |

The primary-split localization itself is real and materially better than the
retired relationship-dynamics result. The nominal package p-value is not.

## Exploratory Gain Gate

After the package significance failure was observed, an explicitly post-hoc
gate was tested. The first half of tensor files supplied only pre-event root
gains. Their 99th percentile was frozen as a threshold and evaluated on the
second half of files.

| Corpus | Held-out event detection | Held-out onset localization | Held-out pre-event false positive |
|---|---:|---:|---:|
| frequency, random forest raw | 100.0% | 100.0% | 4.5% |
| voltage, random forest raw | 100.0% | 100.0% | 0.6% |
| frequency, KNN raw | 100.0% | 63.6% | 2.3% |
| voltage, KNN raw | 100.0% | 93.8% | 0.9% |

This is **good exploratory evidence**, not a confirmatory result. The gate was
designed after seeing that nominal p-values failed, although its threshold and
evaluation files are separated. It should be reproduced on a new event corpus
before being treated as a validated detector.

Three additional boundaries are load-bearing:

- Every evaluated 600-sample window was already selected to contain an event.
  This does not establish performance on an unsegmented continuous stream.
- Held-out files remain synthetic samples from the same pmuBAGE generator
  family. They are not an external grid, operating condition, or utility.
- The root split identifies maximum multivariate distributional divergence. It
  need not identify the first causal action in a multi-stage disturbance.

## Sober Grade

- Explicit abrupt-change localization: **impressive within this narrow task**.
- Package significance on autocorrelated PMU series: **bad**.
- Empirically calibrated gain gate: **good, pending external confirmation**.
- Generic latent-structure discovery: **not addressed**.

This demonstrates the practical boundary. Established tools are much better
when the objective is named. `changeforest` can identify the dominant
distributional transition without knowing the physical domain. It does not
decide whether that transition is meaningful, anomalous, causal, or worth
acting upon, and this experiment does not establish continuous-stream event
detection.

## Reproduction

```bash
python3 -m pip install -e '.[research]'
python3 experiments/established_change_point_benchmark.py \
  --synthetic-repeats 100 \
  --permutations 99 \
  --estimators 50 \
  --jobs 8 \
  --checkpoint experiments/results/2026-06-05-established-change-point-checkpoint.jsonl \
  --output experiments/results/2026-06-05-established-change-point-benchmark.json
```

Primary artifacts:

- `experiments/established_change_point_benchmark.py`
- `experiments/results/2026-06-05-established-change-point-benchmark.json`
- `experiments/results/2026-06-05-established-change-point-checkpoint.jsonl`
