# Social/Organizational Frozen-Observer Layer Assessment

## Question

What did the earlier negative positive-coactivation aggregates mean after the
organizational feature surfaces passed directional-quality controls?

The possibilities were materially different:

- mutual exclusion between observer score channels;
- lagged succession or phase-separated score behavior;
- sparse observer sampling;
- threshold/decay aggregate behavior; or
- structure lost by the spectral-observer transformation.

## Method

The same frozen spectral baseline and adaptive-window geometry was applied to
the selected organizational views. Observer score channels use **positive-tail
activation** because a high score is anomalous; an unusually low anomaly score
is not silently treated as an active event.

The assessment separates three measurements:

1. **Positive-coactivation aggregate:** the prior thresholded, decayed emission
   objective.
2. **Directional score structure:** coactivation, exclusion, lagged succession,
   and phase offset measured on the observer score matrix.
3. **Repeated random-order control:** rebuild the feature surface and frozen
   observer for repeated random event orders while freezing candidate input and
   score feature names.

The exact prior SocioPatterns configuration produced only four observer anchors
per calibration or validation half. Its aggregate arithmetic was reproduced,
but four anchors were graded insufficient for directional diagnosis.

The same frozen baseline and adaptive window were then sampled with stride 1,
producing 28 and 29 score anchors. This changes observation resolution; it does
not refit the frozen nominal or change event order.

## Independent Resolution Calibration

“Adequate resolution” was not defined by whether the real-data result became
positive. Each observed score-matrix geometry was independently tested over 20
positive-valued synthetic observer-score trials per known mechanism plus noise.

At the 28–29 anchor geometries:

- noise false-positive rate was controlled at 5–10%;
- strong synthetic coactivation was recovered in 100% of trials;
- strong synthetic phase offset was recovered in 100% of trials;
- exclusion and lagged succession were not reliably recovered.

Therefore:

- coactivation and phase-offset presence or absence is interpretable at this
  resolution;
- exclusion and succession absence is underpowered and must remain unresolved;
- the four-anchor matrices are unusable for directional diagnosis.

## Results

### SocioPatterns Group Surface, Group-Then-Time Order

The exact four-anchor positive aggregate reproduced the earlier deficit:

| Segment | Exact aggregate delta | z effect | lower-tail empirical p |
| --- | ---: | ---: | ---: |
| Calibration | -160.70 | -2.52 | 0.0198 |
| Validation | -543.98 | -2.43 | 0.0198 |

With dense observer anchors, the aggregate sign flipped to surplus but did not
replicate as a significant effect. The directional result was clearer:

- observer-score **coactivation replicated** across calibration and validation;
- feature-surface coactivation survived the spectral observer;
- feature-surface phase offset was not observed downstream, and this absence is
  interpretable under the independent power calibration;
- feature-surface exclusion was not observed downstream, but exclusion is
  underpowered at this matrix shape and remains unresolved;
- succession appeared in calibration only and did not replicate.

Coactivation exceeded all 20 random event-order controls using the conservative
minimum-segment delta:

- candidate delta: `0.05337`;
- random-order mean: `0.00198`;
- standardized order-null effect: `z=25.21`;
- empirical `p=1/21=0.0476`.

The group-then-time order uses known group labels. This demonstrates that an
aligned group order preserves score-level organization; it is not autonomous
discovery of the group ordering.

### SocioPatterns Graph Surface, Degree Order

The exact four-anchor positive aggregate also reproduced a deficit:

| Segment | Exact aggregate delta | z effect | lower-tail empirical p |
| --- | ---: | ---: | ---: |
| Calibration | -5.75 | -2.22 | 0.0099 |
| Validation | -8.27 | -1.86 | 0.0297 |

At dense resolution:

- the positive aggregate became effectively null rather than coherently
  negative;
- observer-score **coactivation replicated** across calibration and validation;
- the feature-surface coactivation mechanism survived the observer;
- no exclusion mechanism was detected, but exclusion absence is underpowered.

Coactivation exceeded all 30 random event-order controls:

- candidate delta: `0.01757`;
- random-order mean: `0.00143`;
- standardized order-null effect: `z=3.03`;
- empirical `p=1/31=0.0323`.

### Email-Eu Timestamp Event Order

The feature surface contained replicated coactivation and phase offset. At the
observer-score layer:

- neither coactivation nor phase offset replicated;
- both absences are interpretable because the same 29-by-48 geometry recovered
  strong synthetic coactivation and phase offset in 100% of calibration trials;
- no observer mechanism beat 30 random event-order controls;
- the positive aggregate remained a replicated deficit (`z=-2.76` and
  `z=-2.68`), but that deficit did not resolve into exclusion, succession, or
  phase separation.

The defensible result is that this frozen observer geometry does not preserve
the raw Email-Eu coactivation or phase-offset organization. The aggregate
deficit is not itself a structural mechanism.

### Enron Timestamp Event Order

The current equal-event representation has only 28 and 29 input bins per half,
fewer than the frozen observer baseline of 48. It is unresolved at the observer
layer. The next Enron run needs a finer event-bin representation before any
downstream conclusion is possible.

## Conclusion

The old positive-coactivation deficit was not one thing:

- In both SocioPatterns views, the exact deficit was reproduced at four anchors,
  but dense measurement exposed replicated coactivation. The graph and group
  coactivation also survived repeated random-order controls.
- In Email-Eu, the aggregate deficit persisted at a matrix shape independently
  shown to recover strong synthetic coactivation and phase offset, while no
  directional mechanism replicated. The selected observer geometry did not
  preserve the interpretable feature-surface mechanisms.
- Enron remains under-resolved.

Therefore, a negative thresholded stigmergic aggregate must not be labeled
“exclusion” or “absence of structure.” The aggregate is the output of a
specific threshold, minimum-active-series, and decay operator. Its sign does
not identify which relationship among channels produced that output. A
mechanism claim requires a separate relationship diagnostic and an independent
resolution calibration.

## Reusable Artifacts

- `spectral_forecast/directional.py`
  - supports absolute-tail feature activation and positive-tail observer-score
    activation.
- `spectral_forecast/directional_resolution.py`
  - independently calibrates known-answer recovery and false positives for an
    observed matrix geometry.
- `experiments/org_network_observer_directional_quality.py`
  - reconstructs a feature surface, builds frozen observer scores, grades
    resolution, and compares surface-to-observer mechanisms.
- `experiments/org_network_observer_directional_order_null.py`
  - rebuilds observer scores under repeated random event orders with
    checkpoint/resume support.
- `experiments/directional_resolution_calibration.py`
  - standalone geometry calibration runner.

Primary retained results:

- `2026-06-04-sociopatterns-group-observer-dense-quality.json`
- `2026-06-04-sociopatterns-group-observer-order-null20.json`
- `2026-06-04-sociopatterns-graph-observer-dense-quality.json`
- `2026-06-04-sociopatterns-graph-observer-order-null30.json`
- `2026-06-04-email-eu-timestamp-observer-quality.json`
- `2026-06-04-email-eu-observer-order-null30.json`
- `2026-06-04-enron-timestamp-observer-resolution-audit.json`
