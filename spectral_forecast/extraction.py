"""Layer 1: Noise-aware iterative Fourier extraction.

Extracts dominant periodic components from a time series by iteratively
finding the strongest frequency, modeling its expected spectral footprint
(including noise spread), and subtracting the entire footprint as a unit.

A noisy sinusoid at frequency f spreads energy across nearby frequency bins.
Rather than treating each bin as a separate signal, we model the expected
spread from noise on one signal and subtract it whole. New signals are only
declared when residual structure exceeds what the noise model predicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar


@dataclass
class ExtractedComponent:
    """A single periodic component extracted from the signal."""

    frequency: float  # cycles per sample (multiply by sample_rate for Hz)
    amplitude: float
    phase: float  # radians
    snr: float  # signal-to-noise ratio at extraction time

    def evaluate(self, t: NDArray[np.floating]) -> NDArray[np.floating]:
        """Reconstruct this component's contribution at time indices t."""
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)


@dataclass
class ExtractionResult:
    """Full result of Fourier extraction."""

    components: list[ExtractedComponent]
    residual: NDArray[np.floating]
    noise_std: float
    n_iterations: int


def _estimate_noise_floor(power_spectrum: NDArray[np.floating]) -> float:
    """Estimate noise floor from the median of the power spectrum.

    The median is robust to spectral peaks — for a signal with a few strong
    frequencies plus noise, most bins are noise-dominated, so the median
    estimates the noise power level.
    """
    return float(np.median(power_spectrum))


def _spectral_footprint_width(noise_floor: float, peak_power: float, n: int) -> int:
    """Estimate how many bins a signal at peak_power spreads into due to noise.

    For a rectangular window, a sinusoid's main lobe is ~2 bins wide, but noise
    causes energy to spread further. The spread scales with sqrt(noise/signal).
    We use a conservative estimate: main lobe + noise-dependent skirt.
    """
    if peak_power <= 0:
        return 1
    snr = peak_power / max(noise_floor, 1e-30)
    # Main lobe (2 bins) + noise skirt. At high SNR, footprint is narrow.
    # At low SNR, it broadens as noise spreads energy.
    skirt = max(1, int(np.ceil(2.0 / np.sqrt(max(snr, 0.01)))))
    return min(2 + skirt, n // 4)  # never wider than quarter the spectrum


def _fit_sinusoid(
    signal: NDArray[np.floating], t: NDArray[np.floating], freq_guess: float
) -> tuple[float, float, float]:
    """Fit a sinusoid A*cos(2*pi*f*t + phi) to signal via least squares.

    Refines frequency around freq_guess, then solves for amplitude and phase
    analytically using the normal equations for the linear part.
    """
    n = len(signal)

    def _residual_energy(f: float) -> float:
        basis_cos = np.cos(2 * np.pi * f * t)
        basis_sin = np.sin(2 * np.pi * f * t)
        # Solve [a, b] for signal ≈ a*cos + b*sin
        X = np.column_stack([basis_cos, basis_sin])
        coeffs, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
        residual = signal - X @ coeffs
        return float(np.sum(residual**2))

    # Refine frequency in a narrow band around the FFT peak
    df = 1.0 / n  # frequency resolution
    result = minimize_scalar(
        _residual_energy,
        bounds=(max(freq_guess - 2 * df, df / 2), freq_guess + 2 * df),
        method="bounded",
    )
    freq = result.x

    # Extract amplitude and phase at refined frequency
    basis_cos = np.cos(2 * np.pi * freq * t)
    basis_sin = np.sin(2 * np.pi * freq * t)
    X = np.column_stack([basis_cos, basis_sin])
    coeffs, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
    a, b = coeffs[0], coeffs[1]

    amplitude = float(np.sqrt(a**2 + b**2))
    phase = float(np.arctan2(-b, a))

    return freq, amplitude, phase


def extract(
    signal: NDArray[np.floating],
    sample_rate: float = 1.0,
    max_components: int = 50,
    min_snr: float = 2.0,
    noise_threshold_ratio: float = 0.5,
    min_cycles: float = 2.0,
) -> ExtractionResult:
    """Extract periodic components using noise-aware iterative Fourier extraction.

    Args:
        signal: 1D time series values.
        sample_rate: Samples per unit time (used to report frequencies in Hz).
        max_components: Maximum number of components to extract.
        min_snr: Minimum SNR for a peak to be considered a real signal.
        noise_threshold_ratio: Stop when peak magnitude < this * cumulative
            extraction error. Default 0.5 means stop when the peak is less
            than half the accumulated noise from prior extractions.
        min_cycles: Minimum number of complete cycles in the data for a
            frequency to be extractable. Frequencies below min_cycles/n are
            left for Layer 2 (trend fitting). Default 2.0.

    Returns:
        ExtractionResult with extracted components, residual, and noise stats.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

    n = len(signal)
    if n < 8:
        raise ValueError(f"Signal too short for extraction (n={n}, need >= 8)")

    t = np.arange(n, dtype=np.float64)
    residual = signal.copy()
    components: list[ExtractedComponent] = []
    cumulative_extraction_error = 0.0

    for iteration in range(max_components):
        # Compute power spectrum of residual
        fft_vals = np.fft.rfft(residual)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n)

        # Zero out DC and sub-min_cycles frequencies — handled by Layer 2
        min_freq = min_cycles / n
        min_bin = max(1, int(np.ceil(min_freq * n)))
        power[:min_bin] = 0.0

        usable_power = power[min_bin:]
        if len(usable_power) == 0:
            break
        noise_floor = _estimate_noise_floor(usable_power)
        peak_idx = int(np.argmax(usable_power)) + min_bin
        peak_power = power[peak_idx]
        peak_freq = freqs[peak_idx]

        # SNR check against extreme value threshold.
        # The max of K i.i.d. exponential noise bins has expected value
        # ~ noise_floor * ln(K). We require the peak to exceed this by min_snr.
        n_bins = len(power) - min_bin  # excluding DC and sub-min_cycles
        extreme_value_threshold = noise_floor * (1 + np.log(max(n_bins, 2)))
        snr = peak_power / max(noise_floor, 1e-30)
        if peak_power < min_snr * extreme_value_threshold:
            break

        # Noise threshold check: stop if peak is below noise_threshold_ratio
        # times the cumulative extraction error
        if iteration > 0 and peak_power < noise_threshold_ratio * cumulative_extraction_error:
            break

        # Compute spectral footprint width
        footprint_w = _spectral_footprint_width(noise_floor, peak_power, len(power))

        # Fit sinusoid at refined frequency
        freq, amplitude, phase = _fit_sinusoid(residual, t, peak_freq)

        # Reconstruct and subtract
        component_signal = amplitude * np.cos(2 * np.pi * freq * t + phase)
        residual = residual - component_signal

        # Track extraction error: the energy we removed in the spectral footprint
        # beyond the main peak represents noise we inadvertently absorbed
        fft_after = np.fft.rfft(residual)
        power_after = np.abs(fft_after) ** 2

        # Extraction error is the energy change in the footprint zone minus the
        # energy of the pure sinusoid (which is the intended extraction)
        lo = max(1, peak_idx - footprint_w)
        hi = min(len(power), peak_idx + footprint_w + 1)
        energy_removed_footprint = float(np.sum(power[lo:hi] - power_after[lo:hi]))
        # FFT power |X[k]|^2 for amplitude A, length N: peak ≈ (A*N/2)^2
        pure_sinusoid_energy = (amplitude * n / 2) ** 2
        extraction_noise = max(0.0, energy_removed_footprint - pure_sinusoid_energy)
        cumulative_extraction_error += extraction_noise

        components.append(
            ExtractedComponent(
                frequency=freq,  # cycles per sample; multiply by sample_rate for Hz
                amplitude=amplitude,
                phase=phase,
                snr=snr,
            )
        )

    noise_std = float(np.std(residual))

    return ExtractionResult(
        components=components,
        residual=residual,
        noise_std=noise_std,
        n_iterations=len(components),
    )
