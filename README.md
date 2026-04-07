# spectral-forecast

**Entropy-Bounded Decomposition for Time Series Forecasting**

An analytical time series forecasting method that decomposes signals into interpretable components using classical signal processing, governed by an information-theoretic stopping criterion. No training data. No GPU. 1,200 lines of Python.

On datasets with periodic structure, it outperforms Google's TimesFM (200M parameters) by 42--68%.

**Paper:** [Entropy-Bounded Decomposition for Time Series Forecasting](paper/paper.pdf) | [DOI: 10.5281/zenodo.19451784](https://doi.org/10.5281/zenodo.19451784)

## Results

Normalized MSE on standard benchmarks (horizon = 96):

| Dataset | spectral-forecast | TimesFM (200M) | PatchTST | Verdict |
|---------|:-----------------:|:--------------:|:--------:|---------|
| ETTh1   | **0.131**         | 0.375          | 0.370    | +65%    |
| ETTm1   | **0.088**         | 0.320          | 0.293    | +68%    |
| Weather | 0.156             | ---            | **0.149**| -5%     |
| ETTh2   | 0.400             | **0.289**      | 0.274    | -38%    |
| ETTm2   | 0.251             | **0.175**      | 0.166    | -44%    |
| ECL     | 0.609             | ---            | **0.129**| -372%   |

The method dominates on periodic time series (temperature, meteorological data) and loses on non-periodic data (individual electricity consumption). The residual lag-1 autocorrelation after decomposition predicts which: AC(1) near zero means the Fourier basis captured everything; AC(1) > 0.1 means predictable structure remains that needs a different basis or a learned model.

All six benchmarks run in 80 seconds on a laptop. No GPU.

## How It Works

The signal is decomposed in layers. Each layer extracts a different kind of structure from the residual of the previous layer. Decomposition stops when the residual reaches maximum entropy --- no predictable structure remains.

**Layer 1: Fourier Extraction.** Iteratively identifies dominant periodic components using noise-aware spectral footprinting. Rather than treating each FFT peak as a separate signal, the algorithm models the expected spectral spread of a single sinusoid at the observed noise level and extracts the entire footprint as a unit. An extreme value threshold (scaled by log of the number of frequency bins) prevents extraction of noise peaks.

**Layer 2: Trend Fitting.** Fits the residual with candidate models (constant, linear, quadratic, exponential) using recency-weighted least squares. Selects by BIC. Nonlinear trends are damped beyond the training boundary --- extrapolation transitions to the tangent line to prevent divergence.

**Layer 3a: Shock Detection.** Scans for discrete events where the Layer 1+2 prediction diverges from reality. Fits three shapes (step, spike-decay, ramp) at each onset and selects by AIC.

**Layer 3b: Local Correction.** Fits a recency-weighted AR(p) model on the remaining residual to capture short-term momentum and mean-reversion. AIC selects the order. The correction decays exponentially with forecast horizon --- recent local dynamics inform near-term forecasts but fade at long horizons.

**Adaptive Wavelet Path.** If the residual after all layers still has autocorrelation (AC(1) >= 0.1), a discrete wavelet transform (Daubechies-4) captures localized time-frequency features the periodic basis missed.

The forecast is the superposition of all layers, clamped to the observed signal range with asymmetric error bounds from the residual distribution.

## The Stopping Criterion

Each layer is a lossy compressor that removes predictable structure. The stopping criterion is the information-theoretic limit: decomposition is complete when the residual is maximum entropy for its variance. For continuous data, this means Gaussian with zero autocorrelation at all lags --- the residual is incompressible. AC(1) near zero is the practical check.

This connects decomposition to rate-distortion theory: the layers define a codebook, the residual is the distortion, and the stopping criterion is the point where the distortion cannot be compressed further.

## Install

```bash
git clone https://github.com/jmcentire/spectral-forecast.git
cd spectral-forecast
pip install -e ".[dev]"
```

Requires Python >= 3.11, NumPy, SciPy, PyWavelets.

## Usage

```python
import numpy as np
from spectral_forecast import SpectralForecaster

# Fit and forecast
signal = np.sin(np.linspace(0, 20, 500)) + 0.1 * np.arange(500)
forecaster = SpectralForecaster()
result = forecaster.fit_forecast(signal, horizon=96)

# Inspect the decomposition
print(result.describe())

# Access components
print(result.point_forecast)    # forecast values
print(result.lower_bound)       # error bounds
print(result.upper_bound)
print(result.components)        # extracted frequencies
print(result.trend)             # trend model
print(result.shocks)            # detected shocks
print(result.local)             # AR correction
```

### CLI

```bash
# Forecast from CSV
spectral-forecast forecast data.csv --column OT --horizon 96 --describe

# Run benchmark
spectral-forecast benchmark data.csv --column OT --horizons 96 192 336 720

# Show decomposition
spectral-forecast decompose data.csv --column OT
```

## Tests

```bash
pytest tests/ -v
```

32 tests covering each layer independently and the full pipeline on synthetic data.

## Known Limitations

- **Univariate only.** No cross-variable dependencies.
- **Fourier stationarity.** Layer 1 assumes frequency content is stationary over the context window. Chirp signals or frequency-modulated data will blur.
- **Shock absorption.** Level shifts in the middle of the context leak spectral energy into Layer 1 before Layer 3a can detect them. An iterative architecture (extract, detect, re-extract) would fix this but is future work.
- **Non-periodic data.** On signals without periodic structure (ETTh2, ETTm2, ECL), transformer baselines win.

## Citation

```bibtex
@article{mcentire2026entropy,
  title={Entropy-Bounded Decomposition for Time Series Forecasting: Fourier Extraction, Adaptive Basis Selection, and the Maximum Entropy Stopping Criterion},
  author={McEntire, Jeremy},
  year={2026},
  doi={10.5281/zenodo.19451784}
}
```

## License

MIT
