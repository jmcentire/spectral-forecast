# CDIP Ocean Validation-Ladder Reassessment

## Question

Does the previously reported CDIP aggregate establish structure specific to the
original multi-buoy groups, or can the same aggregate arise from score
trajectories shared across arbitrary groups?

This is stricter than asking whether the observed aggregate exceeds a
within-window timing-permutation null.

## New Relationship-Specific Control

The trajectory-regroup null:

1. preserves every complete per-buoy observer-score trajectory;
2. preserves each trajectory's score distribution and anchor-position profile;
3. preserves the number of groups, anchors, and series per group; and
4. destroys only original buoy group membership by rebuilding groups from
   unrelated trajectories.

The frozen candidate came from the fresh-naive 512/512 disjoint-group CDIP run.
No candidate setting was refit for this control. The control used 1,000 unique
regrouping repeats on calibration and validation for three preprocessing
surfaces.

## Result

The original anchor-permutation surplus replicates decisively. The
group-specific remainder does not.

| Preprocess | Segment | Anchor-permute delta | Anchor p | Regroup delta | Regroup z effect | Regroup p | Fraction of anchor surplus |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | calibration | 1680.75 | 0.0010 | 296.07 | 1.54 | 0.0649 | 17.6% |
| none | validation | 1876.72 | 0.0010 | 176.57 | 0.94 | 0.1828 | 9.4% |
| highpass | calibration | 1694.55 | 0.0010 | 298.15 | 1.56 | 0.0589 | 17.6% |
| highpass | validation | 1859.35 | 0.0010 | 168.88 | 0.88 | 0.1778 | 9.1% |
| highpass-dominant-mask | calibration | 1553.55 | 0.0010 | 361.13 | 1.93 | 0.0350 | 23.2% |
| highpass-dominant-mask | validation | 1355.59 | 0.0010 | -217.32 | -1.23 | 0.8891 | -16.0% |

The masked calibration result is positive under regrouping, but it reverses on
the disjoint validation groups. It therefore does not replicate.

## Revised Ocean Result

The phrase **coherent multi-buoy residual structure** was too strong.

What is established:

- the frozen observer produces reproducible anchor-aligned score structure;
- that structure is decisively above a null that independently permutes anchor
  timing within each trajectory;
- high-scoring windows have post-hoc swell-like spectral characteristics; and
- the validation ladder can expose a shared-template/group-independence failure
  mode that the earlier nulls missed.

What is not established:

- that the aggregate depends on the original buoy groups;
- that the aggregate represents physical cross-buoy coherence;
- a propagation field or directional ocean mechanism; or
- rogue-wave relevance.

The trajectory-regroup result indicates that most of the prior aggregate
surplus is explained by score trajectories and anchor-position structure that
remain effective after arbitrary regrouping. The post-hoc spectral attribution
still describes the kinds of ocean windows receiving high scores. It does not
turn those scores into evidence of original buoy-group coherence.

## Validation Ladder

| Layer | Status | Assessment |
| --- | --- | --- |
| Information sufficiency | mixed | 5,491 windows and 1,024 groups support aggregate controls; six observer anchors per window are under-resolved for directional mechanism diagnosis. |
| Known-answer validation | passed for new control | Synthetic tests distinguish a shared anchor template from true group-specific alignment. |
| Canonical surface | present, not separately graded | Raw z-channel displacement supplies the source surface. |
| Frozen observer | diagnosed | Observer scores contain a strong shared anchor-aligned component. |
| Stigmergic aggregate | positive but group-independent | Aggregate lift survives anchor permutation controls but not the stricter original-group test. |
| Within-layer nulls | passed as arithmetic, incomplete as meaning | Timing permutation, tide, masking, and phase controls are valid measurements but did not establish group specificity. |
| Second-level controls | failed original-group replication | Trajectory regrouping explains most surplus; the small remainder does not replicate. |
| Post-hoc attribution | retained with narrower meaning | High-score windows are narrower-band, longer-period, swell-like sea states; attribution does not establish cross-buoy coherence. |

## General Method Learning

An aggregate timing-permutation null can reward a shared observer template even
when original component grouping carries little information. A
relationship-specific second-level null must preserve the suspected shared
template while breaking only the relationship being claimed.

For grouped multichannel experiments, trajectory regrouping should become a
required post-selection control before describing aggregate lift as
group-specific coherence.

Primary artifact:

- `experiments/results/2026-06-04-cdip-observer-group-null1000.json`
