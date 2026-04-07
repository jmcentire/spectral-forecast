"""Wavelet decomposition for non-periodic residual structure.

When the Fourier residual has high autocorrelation, it contains predictable
structure that isn't periodic — localized transients, frequency-modulated
oscillations, or slowly-varying envelopes. Wavelets capture these because
they have time-frequency resolution that Fourier lacks.

Uses discrete wavelet transform (DWT) via PyWavelets. The signal is
decomposed into approximation (low-frequency trend) and detail (localized
features at each scale) coefficients. Forecasting extends the coefficients
and reconstructs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    import pywt
except ImportError:
    pywt = None  # graceful degradation — wavelet path unavailable


@dataclass
class WaveletComponent:
    """A single wavelet scale's contribution."""

    level: int
    kind: str  # "approximation" or "detail"
    energy_fraction: float  # fraction of total signal energy at this scale
    coefficients: NDArray[np.floating]


@dataclass
class WaveletModel:
    """Fitted wavelet decomposition."""

    wavelet_name: str
    n_levels: int
    components: list[WaveletComponent]
    original_length: int
    residual_std: float

    def describe(self) -> str:
        lines = ["Wavelet: %s, %d levels" % (self.wavelet_name, self.n_levels)]
        for c in self.components:
            lines.append(
                "  L%d %s: %.1f%% energy, %d coeffs"
                % (c.level, c.kind, c.energy_fraction * 100, len(c.coefficients))
            )
        return "\n".join(lines)


@dataclass
class WaveletResult:
    """Full result of wavelet decomposition."""

    model: WaveletModel
    fitted_values: NDArray[np.floating]
    residual: NDArray[np.floating]


def _select_wavelet_and_levels(n: int) -> tuple[str, int]:
    """Select wavelet and decomposition depth for a given signal length.

    Uses Daubechies-4 (db4) — good balance of smoothness and compactness.
    Depth is limited so the coarsest scale has enough coefficients to be
    meaningful (at least 8).
    """
    wavelet = "db4"
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    # Limit depth so coarsest level has >= 8 coefficients
    level = min(max_level, int(np.log2(max(n, 16))) - 3)
    level = max(level, 1)
    return wavelet, level


def _extrapolate_coefficients(
    coeffs: NDArray[np.floating], n_extra: int
) -> NDArray[np.floating]:
    """Extrapolate wavelet coefficients forward.

    Uses linear extrapolation from the last few coefficients. Conservative:
    the extrapolation decays toward zero (mean of the coefficients) as it
    extends, reflecting decreasing confidence.
    """
    n = len(coeffs)
    if n < 2 or n_extra <= 0:
        return np.zeros(n_extra)

    # Fit a local trend on the last few coefficients
    tail_len = min(n, 8)
    tail = coeffs[-tail_len:]
    t = np.arange(tail_len, dtype=np.float64)
    # Linear regression
    X = np.column_stack([t, np.ones(tail_len)])
    fit, _, _, _ = np.linalg.lstsq(X, tail, rcond=None)
    slope, intercept = fit

    # Extrapolate with decay toward mean
    t_ext = np.arange(tail_len, tail_len + n_extra, dtype=np.float64)
    linear_ext = slope * t_ext + intercept
    coeff_mean = float(np.mean(coeffs))

    # Exponential decay toward mean with halflife = tail_len
    decay = np.exp(-np.log(2) * np.arange(n_extra, dtype=np.float64) / max(tail_len, 1))
    extrapolated = decay * linear_ext + (1 - decay) * coeff_mean

    return extrapolated


def fit_wavelet(
    residual: NDArray[np.floating],
    min_energy_fraction: float = 0.01,
) -> WaveletResult:
    """Decompose residual using discrete wavelet transform.

    Args:
        residual: 1D array — the signal after Fourier + trend extraction.
        min_energy_fraction: Minimum energy fraction for a scale to be
            included in the model. Scales below this are treated as noise.

    Returns:
        WaveletResult with decomposition and fitted values.
    """
    if pywt is None:
        raise ImportError(
            "PyWavelets (pywt) is required for wavelet decomposition. "
            "Install with: pip install PyWavelets"
        )

    residual = np.asarray(residual, dtype=np.float64)
    n = len(residual)

    wavelet_name, n_levels = _select_wavelet_and_levels(n)

    # Discrete wavelet decomposition
    coeffs_list = pywt.wavedec(residual, wavelet_name, level=n_levels)
    # coeffs_list[0] = approximation at coarsest level
    # coeffs_list[1:] = detail at each level (coarsest to finest)

    total_energy = float(np.sum(residual**2))
    if total_energy <= 0:
        total_energy = 1.0

    components = []

    # Approximation coefficients (coarsest scale — low-frequency envelope)
    approx_energy = float(np.sum(coeffs_list[0] ** 2))
    components.append(
        WaveletComponent(
            level=n_levels,
            kind="approximation",
            energy_fraction=approx_energy / total_energy,
            coefficients=coeffs_list[0].copy(),
        )
    )

    # Detail coefficients at each level
    for i, detail in enumerate(coeffs_list[1:]):
        level = n_levels - i
        detail_energy = float(np.sum(detail**2))
        energy_frac = detail_energy / total_energy
        components.append(
            WaveletComponent(
                level=level,
                kind="detail",
                energy_fraction=energy_frac,
                coefficients=detail.copy(),
            )
        )

    # Reconstruct fitted values (using all significant scales)
    # Zero out insignificant scales
    filtered_coeffs = []
    for i, comp in enumerate([components[0]] + components[1:]):
        if comp.energy_fraction >= min_energy_fraction:
            filtered_coeffs.append(coeffs_list[i].copy())
        else:
            filtered_coeffs.append(np.zeros_like(coeffs_list[i]))

    fitted = pywt.waverec(filtered_coeffs, wavelet_name)[:n]
    wavelet_residual = residual - fitted

    model = WaveletModel(
        wavelet_name=wavelet_name,
        n_levels=n_levels,
        components=components,
        original_length=n,
        residual_std=float(np.std(wavelet_residual)),
    )

    return WaveletResult(
        model=model,
        fitted_values=fitted,
        residual=wavelet_residual,
    )


def forecast_wavelet(
    model: WaveletModel,
    horizon: int,
) -> NDArray[np.floating]:
    """Forecast by extrapolating wavelet coefficients.

    Each scale's coefficients are extrapolated forward (with decay),
    then the inverse DWT reconstructs the forecast signal.

    Args:
        model: Fitted wavelet model.
        horizon: Number of steps to forecast.

    Returns:
        1D array of wavelet-based forecast corrections.
    """
    if pywt is None:
        return np.zeros(horizon)

    # We need to produce a signal of length (original + horizon) then
    # take the last `horizon` samples. Extrapolate each coefficient set.
    total_len = model.original_length + horizon

    # Figure out how many coefficients each level needs for total_len
    wavelet = pywt.Wavelet(model.wavelet_name)
    # Use wavedec on a dummy signal to get target coefficient lengths
    dummy = np.zeros(total_len)
    target_coeffs = pywt.wavedec(dummy, wavelet, level=model.n_levels)

    extended_coeffs = []
    for i, comp in enumerate(model.components):
        target_len = len(target_coeffs[i])
        current_len = len(comp.coefficients)
        if target_len > current_len:
            extra = _extrapolate_coefficients(
                comp.coefficients, target_len - current_len
            )
            extended = np.concatenate([comp.coefficients, extra])
        else:
            extended = comp.coefficients[:target_len]
        extended_coeffs.append(extended)

    # Reconstruct the extended signal
    reconstructed = pywt.waverec(extended_coeffs, model.wavelet_name)
    # Take only the forecast portion
    forecast = reconstructed[model.original_length : model.original_length + horizon]

    # Pad if reconstruction is short
    if len(forecast) < horizon:
        forecast = np.concatenate([forecast, np.zeros(horizon - len(forecast))])

    return forecast[:horizon]
