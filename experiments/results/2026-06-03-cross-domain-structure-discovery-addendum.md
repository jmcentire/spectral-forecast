# Cross-Domain Structure Discovery Addendum

Date: 2026-06-03

## Superseded Status

The CDIP multi-buoy-coherence interpretation and the broad thesis below are
superseded by `2026-06-04-cross-domain-quality-assessment.md`.

A later trajectory-regroup control showed that the CDIP aggregate is dominated
by group-independent/shared-template effects rather than replicated original
buoy-group coherence. The later assessment retains the narrower ocean result,
adds calibrated social positive/negative outcomes, and separates EEG
self-referential from population-nominal structure.

## Thesis

The experiments now support a stronger and more general claim than the original
ocean framing:

> A label-free spectral + stigmergy observer can discover coherent residual
> structure in unlabeled multi-sensor time series across unrelated domains; the
> meaning of that structure is then assigned by domain-specific post-hoc
> attribution.

This is not an event-prediction claim. It is a structure-discovery claim.

That distinction matters. Event prediction asks, "Will this specific event
happen at this specific time?" In nonstationary, coupled systems, that is often
the brittle question. Structure discovery asks, "What coherent organization is
present in the residual after ordinary predictable behavior is controlled?" That
is the reusable primitive. It works before labels are available, before the
operator knows which feature matters, and before the domain has a name for the
structure being surfaced.

## Method Boundary

The detector is domain-agnostic:

- It receives time series.
- It builds spectral/residual surprise per component.
- It uses stigmergic accumulation to lift coherent multi-component structure.
- It tunes candidate settings without labels.
- It validates against nulls that preserve local distributions while breaking
  the coherence being tested.
- It interprets nothing by itself.

The interpretation is domain-specific:

- Ocean labels and spectra name the surfaced structure as organized sea state.
- EEG seizure labels reveal phase enrichment after the structure is found.
- PMU synthetic event classes expose saturation and sparse-support failure
  modes.

The method discovers candidate structure. Domain evidence attributes meaning.

## CDIP: Known Physics Recovered From Raw Ocean Data

The CDIP experiment remains the strongest calibration case. The detector was
given raw buoy displacement time series and no oceanographic quantities. It
surfaced coherent multi-buoy residual structure that survived permutation nulls,
explicit tide/high-pass controls, dominant spectral-bin masking, held-out group
splits, date controls, and geography controls.

Post-hoc attribution using CDIP bulk parameters and direct spectra showed that
the high-delta windows were not best described as confused or multi-modal seas.
The strongest current read is organized coherent swell-like structure:
longer-period energy, narrower bandwidth, lower entropy, and higher
concentration.

Representative CDIP result:

| Run | Windows | Groups | Delta/window | z effect | Empirical p floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full 70-file masked reference | 5491 | 1024 | 0.5302 | 11.70 | 0.000999 |
| Heldout groups 512-1023 | 2722 | 512 | 0.4970 | 7.79 | 0.000999 |
| March 2026 date control | 245 | 58 | 0.8342 | 3.50 | 0.000999 |
| West Coast geography control | 434 | 67 | 0.9579 | 5.65 | 0.000999 |

What this confirms: the method can recover real physical organizing structure
without being told the domain's concepts. If the structure is known swell-scale
organization, that is not a deflation. It is calibration: known physics fell out
of the agnostic observer.

Primary artifact:

- `experiments/results/2026-06-03-cdip-oceanographic-result.md`

## CHB-MIT: Cross-Subject EEG Structure With Post-Hoc Seizure Enrichment

CHB-MIT tests a different domain shape: many channels on one physiological
system instead of many buoys across a spatial field. The same label-free
selection rule was run independently on three subjects. Seizure labels were
used only after the selected candidate surfaced coherent cross-channel windows.

All three subjects accepted on held-out validation:

| Subject | Files | Val accepted | Delta | z effect | p floor | Pos files | Config | Ictal pos rate | Interictal pos rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| chb01 | 14 | yes | 937.3 | 15.73 | 0.0099 | 1.00 | 1024/256 t=2.5 min=3 | 15/16 = 0.938 | 159/1166 = 0.136 |
| chb02 | 14 | yes | 1127.9 | 4.98 | 0.0099 | 1.00 | 1024/256 t=3.0 min=3 | 5/6 = 0.833 | 69/1420 = 0.049 |
| chb03 | 14 | yes | 1526.6 | 8.27 | 0.0099 | 0.86 | 1024/256 t=3.0 min=3 | 10/14 = 0.714 | 522/1234 = 0.423 |

What this confirms: the method generalizes beyond ocean data. It surfaces
coherent cross-channel physiological structure without labels, and post-hoc
labels show seizure-phase enrichment. It is still not a seizure predictor. Top
windows remain partly or mostly interictal because the detector is not
optimizing seizure proximity and most anchors are interictal.

Primary artifacts:

- `experiments/results/2026-06-03-chbmit-autotune-cross-subject-summary.md`
- `experiments/chbmit_autotune.py`

## pmuBAGE: PMU-Shaped Synthetic Controls And Failure Modes

PMU data is the infrastructure bridge: distributed sensors, predictable
electrical behavior, coherent residuals, and possible cascade structure.
The local pmuBAGE data is synthetic, so it is not a real-grid validation
dataset. It is still useful as a PMU-shaped control case.

The fixed smoke tests already showed that pmuBAGE can saturate a naive
aggregate coherence statistic. The fresh autotune pass sharpened that read:

| Surface | Events | Calibration | Validation | Validation read |
| --- | ---: | --- | --- | --- |
| frequency | 8 | accepted, delta 305.8, z 2.97, support 0.50, saturation 0.790 | rejected, delta -177.2, z -1.86, support 0.25, p 0.9505 | calibration does not transfer |
| voltage magnitude | 20 | accepted, delta 3791.8, z 10.61, support 0.80, saturation 0.114 | rejected, delta 609.6, z 5.10, support 0.20, p floor 0.0099 | aggregate lift survives, support too sparse |

What this confirms: the method has meaningful negative controls. pmuBAGE
frequency exposes saturation/non-transfer. pmuBAGE voltage exposes sparse
support: a few loud events can create aggregate lift, but held-out event-level
support stays too thin, so the validation gate rejects it.

That rejection is a feature. It shows the acceptance rule is stricter than
"aggregate delta is positive." For infrastructure work, this matters: a method
that accepts one loud incident as generalized drift is not trustworthy.

Primary artifacts:

- `experiments/results/2026-06-03-pmubage-autotune-summary.md`
- `experiments/pmubage_autotune.py`

## Cross-Domain Read

These results are not three versions of the same story. They are three
different forms of evidence:

1. CDIP is the known-physics calibration case. The method recovers meaningful
   physical structure from raw unlabeled ocean data.
2. CHB-MIT is the label-enrichment generalization case. The method transfers to
   physiology and post-hoc labels show seizure-phase enrichment.
3. pmuBAGE is the infrastructure-shaped control case. The method runs on
   PMU-like tensors and rejects synthetic surfaces that fail support or
   transfer gates.

Together they confirm the generic primitive:

- noisy multi-sensor data in,
- predictable or distribution-preserving controls around it,
- residual coherence surfaced without labels,
- domain attribution after discovery,
- rejection when saturation or sparse support makes the result non-general.

## What Class Of Problems This Solves

This is useful where event prediction is the wrong first question:

- infrastructure systems where exact failures are rare, coupled, and
  nonstationary;
- monitoring systems where labels are sparse, stale, missing, or retrospective;
- physical systems where known forcing dominates until it is controlled;
- operational systems where a weak signal is distributed across components and
  no single stream looks decisive;
- unknown-unknown discovery, where the point is to surface structure before
  the analyst knows what feature to request.

The method does not replace domain science. It changes the order of operations:
discover coherent residual structure first, then ask domain experts what it is.

## Reporting Rules

Use these constraints in every write-up:

- Do not convert z effects into normal-theory p-values.
- Report empirical p as a Monte Carlo bound with its null repeat count and
  floor.
- Report unique null totals where available.
- Distinguish aggregate lift from event/window support.
- Treat labels as post-hoc attribution unless a run explicitly used labels.
- Treat pmuBAGE as synthetic PMU-shaped control data, not real-grid evidence.

## Next Work

The next infrastructure step is not more pmuBAGE. It is a real PMU or
PMU-adjacent disturbance dataset with enough sensors and event context to test
whether coherent residual structure precedes or accompanies documented grid
events.

If real PMU access remains blocked, the next-best public bridge is another
multi-sensor infrastructure-like stream with event context, such as BGP routing
incidents or public outage telemetry. The method is now calibrated enough to
move from proving that it can find structure to asking what structure it finds
in operational systems.
