# CDIP Oceanographic Result

Date: 2026-06-03

## Summary

The CDIP experiment is now a defensible cross-domain calibration result:

1. An agnostic spectral+stigmergy detector, given only raw buoy displacement time series, surfaced coherent multi-buoy residual structure.
2. That structure survived stricter timing-permutation nulls, explicit high-pass tide controls, dominant spectral-bin masking, and held-out group splits.
3. Post-hoc attribution against CDIP-published bulk parameters and direct CDIP spectra indicates the surfaced structure is not best described as confused or multi-modal sea state.
4. The strongest current interpretation is organized coherent swell-like structure: longer-period energy, narrower spectral bandwidth, lower entropy, and higher concentration.

The result is the observation itself: a generic detector, with no oceanographic labels or predictors, recovered physically meaningful organizing structure from raw ocean data. If the surfaced structure is classical swell-scale organization, that is not a deflation of the result. It is the calibration case: the method found known physics without being told the domain.

## Why CDIP Was a Good Calibration Dataset

CDIP had the right shape for this method:

- Many spatially distributed sensors.
- High-frequency raw time series.
- A predictable layer that can be modeled, subtracted, or controlled.
- A coherent-residual question where structure across components is physically meaningful.
- External domain quantities that can be checked after detection.

The detector did not receive wave labels, significant wave height, peak period, spectral bandwidth, direction, rogue-wave annotations, or any CDIP-published sea-state field. Those quantities were used only after detection for attribution.

Source context:

- CDIP ERDDAP `wave_agg` publishes bulk wave parameters including significant wave height, peak period, average period, peak direction, peak PSD, and zero-crossing period: https://erddap.cdip.ucsd.edu/erddap/tabledap/wave_agg.html
- CDIP THREDDS spectra expose `waveTime`, `waveFrequency`, `waveEnergyDensity`, and related variables used for post-hoc direct spectral attribution: https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/spectrum_plot.html

## Detector Runs

All aggregate claims below use the fixed highpass-dominant-mask detector setup with deterministic permutation nulls. The empirical p-values are resolution floors from 1000 null repeats, so they should be read as `p < 0.001`, not as exact tail probabilities.

| Run | Windows | Groups | Delta/window | z effect | Empirical p floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full 70-file masked reference | 5491 | 1024 | 0.5302 | 11.70 | 0.000999 |
| Heldout groups 512-1023 | 2722 | 512 | 0.4970 | 7.79 | 0.000999 |
| March 2026 date control | 245 | 58 | 0.8342 | 3.50 | 0.000999 |
| West Coast geography control | 434 | 67 | 0.9579 | 5.65 | 0.000999 |

The heldout group-half run retained about 94% of the full masked run's per-window effect, which argues against the result being a single lucky group slice.

Primary artifacts:

- `2026-06-02-cdip-known-structure-masking-summary.json`
- `2026-06-02-cdip-heldout-groups512-1023-summary.json`
- `2026-06-02-cdip-seastate-attribution-and-alt-cohorts-summary.json`
- `2026-06-03-cdip-direct-spectra-attribution-summary.json`

## Controls And Failure Modes

### Tide

Tide was the obvious shared forcing to rule out. The experiment used high-pass preprocessing to remove slow sub-wave-band structure before observation. The aggregate signal survived, making tide an unlikely primary explanation.

This does not mean no residual tidal artifact exists anywhere. It means the aggregate finding is not well explained by slow astronomical forcing.

### Weak Nulls

An earlier shifted-null implementation looked stronger than it was because repeated shifted nulls collapsed to only a small number of unique totals. The instrumentation was fixed to report unique null totals, and the stronger permutation null was used thereafter.

The timing-permutation null preserves each series' score distribution while breaking cross-buoy timing coherence. The aggregate signal survived this stricter control.

### Known Spectral Structure

Dominant local FFT-bin masking removed the largest obvious per-window spectral modes before observation. The signal weakened but remained substantial:

- High-pass reference delta: 3550.66
- Highpass-dominant-mask delta: 2911.34
- Retained effect: about 82%

So the detector was not only rediscovering the single strongest spectral bin.

### Propagation Geometry

Spatial audits did not support a clean propagation claim. Mixed-region and bounded-speed patterns were weak or ordinary under spatial surrogates. The result is better framed as coherent spectral/residual organization than as a mapped swell-front velocity field.

## Attribution

The first attribution pass used CDIP ERDDAP `wave_agg` bulk parameters. That pass did not support the first guess that high-delta windows were multi-modal or confused seas. The proxies that would have pointed toward multi-modal structure, such as period gaps, cross-platform peak-direction spread, and across-platform peak-period spread, were not consistently higher.

The direct spectra pass was more decisive. It read CDIP THREDDS `waveEnergyDensity`, `waveFrequency`, and `waveBandwidth` after detection and computed true spectral shape metrics.

Heldout, high-delta versus month-matched baseline:

| Metric | High mean | Baseline mean | Difference | p |
| --- | ---: | ---: | ---: | ---: |
| Bandwidth Hz | 0.0765 | 0.0821 | -0.00564 | 0.012 |
| 90% energy width Hz | 0.2178 | 0.2388 | -0.02095 | 0.0248 |
| Hm0 estimate | 1.848 | 1.625 | +0.223 | 0.025 |
| Mean period s | 7.054 | 6.535 | +0.519 | 0.0296 |
| Peak count mean | 2.225 | 2.335 | -0.110 | 0.385 |
| Multimodal candidate mean | 0.572 | 0.601 | -0.029 | 0.428 |

Interpretation:

- The high-delta windows are narrower, not broader.
- They are more concentrated, not more entropic.
- They do not have more peaks.
- They look more like organized coherent swell-like structure than confused or multi-modal sea state.

March 2026 shows weaker same-direction bandwidth effects. West Coast-only shows strong differences versus broad background, but those mostly vanish under month matching. The heldout group split is the strongest current attribution evidence.

## What The Experiment Establishes

Established:

- The agnostic detector can surface coherent cross-buoy structure from raw displacement data.
- The structure survives multiple controls aimed at tide, weak nulls, obvious dominant spectral modes, and group overfit.
- Post-hoc direct spectra suggest the surfaced structure is organized, longer-period, narrower-band sea state.
- The observation was label-blind: oceanographic quantities entered only after the detector had already surfaced candidate structure.

Open:

- Whether the structure corresponds cleanly to named swell events or broader basin-scale organization.
- Whether masking known swell-scale structure exposes additional residual structure.
- Whether any residual structure is related to rogue-wave precursor regimes.

The method result is the important one: a domain-agnostic detector recovered physically meaningful ocean structure without being told what oceans, swells, tides, or spectra are.

## Next Ocean-Specific Work

If this line earns more time:

1. Plot spectra for the top heldout clusters to visually confirm the narrowband/coherent read.
2. Add event-level clustering around the strongest March 18 and late-May windows.
3. Try an explicit known-swell conditioning or masking pass and rerun the detector on the residual.
4. Then, as a separate experiment, ask whether any residual structure is rogue-relevant.

For now, the result is good enough as a calibration bridge into other domains.
