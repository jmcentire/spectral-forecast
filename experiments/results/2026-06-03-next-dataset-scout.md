# Next Dataset Scout

Date: 2026-06-03

## Selection Criterion

CDIP worked because it had this shape:

- Many spatially or functionally distributed sensors.
- High-rate time series.
- A predictable layer that can be modeled, subtracted, or controlled.
- A coherent-residual question where structure across components is meaningful.
- Ground truth or external domain quantities for post-hoc checking.

Anything matching that shape is a proving ground. Anything matching it without good ground truth is a real application.

## Ranking

| Rank | Dataset family | Fit | Access | Ground truth | Recommendation |
| ---: | --- | --- | --- | --- | --- |
| 1 | CHB-MIT EEG seizure data | Very high | Public, immediate | Strong seizure labels | Implement next |
| 2 | PMU/synchrophasor grid data | Conceptually best bridge | Fragmented public access | Variable | Scout in parallel |
| 3 | EarthScope seismic waveforms | High scale, public | Public FDSN services | Strong event catalogs | Good second/third proving run |
| 4 | RIPE RIS / RouteViews BGP | Strong infrastructure analogy | Public, large MRT files | Partial incident ground truth | Good real-application proxy |
| 5 | GNSS deformation networks | Clean physics | Public/auth may vary | Slow-slip catalogs exist | Interesting but slower |
| 6 | SuperMAG/magnetometers | Good multi-station physics | Public access requires workflow check | Substorm/storm catalogs | Good later geophysical control |
| 7 | Financial markets | Structurally plausible | Public/commercial | Weak and narrative | Avoid as validation dataset |

## 1. CHB-MIT EEG

Why it fits:

- Multi-channel sensor array.
- 256 Hz sampling, close enough to CDIP's "raw high-frequency time series" shape.
- Predictable structure exists: patient/channel baselines, alpha/beta rhythm structure, line noise, artifact regimes.
- Coherent residual structure across channels is meaningful.
- Seizure onset and offset labels are explicit.

Access check:

- PhysioNet lists 664 EDF files and seizure annotation metadata.
- `RECORDS-WITH-SEIZURES` is directly downloadable.
- `chb01-summary.txt` is directly downloadable and includes seizure start/end offsets.
- The full uncompressed corpus is 42.6 GB, but a patient-level pilot can be much smaller.

Verified sample:

- `chb01-summary.txt` reports 256 Hz sampling.
- `chb01_03.edf` has one seizure from 2996 to 3036 seconds.
- `chb01_04.edf` has one seizure from 1467 to 1494 seconds.

Source:

- https://physionet.org/content/chbmit/1.0.0/

Bounded first experiment:

1. Download `chb01` only.
2. Parse EDF channels and summary seizure labels.
3. Build fixed windows around seizure and non-seizure intervals.
4. Run detector without labels.
5. Compare surfaced high-delta windows against seizure/pre-seizure/post-seizure intervals only after detection.
6. Hold the detector settings fixed before expanding to more patients.

Key methodological boundary:

- Do not train or tune on seizure labels.
- Use labels only as post-hoc attribution.
- Use patient-heldout splits before making any portability claim.

## 2. PMU / Synchrophasor

Why it fits:

- This is the actual infrastructure bridge.
- Many distributed sensors.
- High-rate synchronized measurements.
- Predictable layer: 50/60 Hz fundamental, known load/generation cycles, grid operating modes.
- Coherent residual: inter-area oscillations, instability, cascading disturbances.

Access reality:

- Public real PMU disturbance corpora are harder to pin down than EEG.
- `pmuBAGE` is public and useful, but synthetic.
- It contains synthetic frequency and voltage events with 100 PMU indices and 600 time samples per event.
- That makes it useful for adapter and method rehearsal, not for final validation.

Source:

- https://github.com/NanpengYu/pmuBAGE

Recommendation:

- Keep PMU as the bridge target.
- Scout for real multi-site disturbance records from BPA, EPRI, FNET/GridEye, DOE/NASPI, Texas Synchrophasor Network, or utility research releases.
- Use `pmuBAGE` only if we want a synthetic adapter smoke test.

## 3. EarthScope Seismic

Why it fits:

- Many distributed stations.
- High-rate waveforms.
- Predictable/known layers: teleseisms, instrument response, known event arrivals.
- Coherent residual questions are physically meaningful.
- Event catalogs provide strong ground truth.

Access check:

- EarthScope exposes FDSN station metadata and waveform services.
- A station query for `IU.ANMO.BHZ` returned channel metadata and 40 Hz sample rates.

Source:

- https://service.earthscope.org/

Recommendation:

- Use after EEG if we want a larger public geophysical proving run.
- Needs more careful event selection than EEG, so it is not the fastest next step.

## 4. BGP Routing Data

Why it fits:

- Distributed observers across the internet.
- Coherent anomalies can represent route leaks, outages, or propagation of routing instability.
- This is infrastructure-adjacent and public.

Access check:

- RIPE RIS exposes public collector archives in MRT format.
- Example `rrc00/2026.06` update files are about 3 to 4 MB per five-minute interval; route table dumps are about 400 MB.

Source:

- https://ris.ripe.net/docs/mrt/
- https://data.ris.ripe.net/rrc00/2026.06/

Recommendation:

- Good "real application" target once the method has a second labeled proving result.
- Needs a BGP-specific adapter and incident labeling strategy.

## 5. GNSS / GPS Deformation

Why it fits:

- Many spatially distributed sensors.
- Predictable layers include solid-earth tides and seasonal deformation.
- Coherent residuals can correspond to slow slip or crustal motion.

Access:

- EarthScope provides GNSS data streams and data products, but access terms and authentication need a closer pass.

Source:

- https://www.earthscope.org/data/gnss-realtime/

Recommendation:

- Strong physics, slower time scale.
- Better after seismic or EEG.

## 6. SuperMAG / Magnetometer Arrays

Why it fits:

- Distributed global sensors.
- Predictable layers include diurnal and solar-cycle structure.
- Coherent residuals can correspond to substorms or geomagnetic disturbances.

Recommendation:

- Worth a later pass, but access workflow and event-label strategy need verification before implementation.

## Proposed Next Move

Implement CHB-MIT first.

It is the fastest way to ask the next method question:

> Does the same agnostic spectral+stigmergy workflow surface labeled event-regime structure in a completely different multi-sensor domain?

If yes, CDIP plus EEG gives us two independent calibration domains before we point the detector at lower-ground-truth infrastructure data.

Parallel task:

Keep scouting real PMU event datasets. If a real multi-site PMU disturbance corpus becomes available, pivot to it as the infrastructure bridge.
