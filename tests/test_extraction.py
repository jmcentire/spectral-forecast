"""Tests for Layer 1: noise-aware iterative Fourier extraction."""

import numpy as np
import pytest

from spectral_forecast.extraction import extract


def _make_signal(freqs, amps, phases, n=1000, noise_std=0.0):
    """Helper: create synthetic signal from known components."""
    t = np.arange(n, dtype=np.float64)
    signal = np.zeros(n)
    for f, a, p in zip(freqs, amps, phases):
        signal += a * np.cos(2 * np.pi * f * t + p)
    if noise_std > 0:
        rng = np.random.default_rng(42)
        signal += rng.normal(0, noise_std, n)
    return signal


class TestSingleFrequency:
    """Extract a single sinusoid."""

    def test_clean_sinusoid(self):
        signal = _make_signal([0.05], [3.0], [0.7], n=1000)
        result = extract(signal)

        assert len(result.components) >= 1
        c = result.components[0]
        assert abs(c.frequency - 0.05) < 0.005, f"freq {c.frequency} != 0.05"
        assert abs(c.amplitude - 3.0) / 3.0 < 0.05, f"amp {c.amplitude} != 3.0"

    def test_noisy_sinusoid_extracts_one(self):
        """A noisy sinusoid should still be recognized as ONE component, not many."""
        signal = _make_signal([0.05], [5.0], [0.0], n=1000, noise_std=1.0)
        result = extract(signal, min_snr=3.0)

        # Should find 1 dominant component, not fragment into many
        dominant = [c for c in result.components if c.snr > 5.0]
        assert len(dominant) == 1, f"Expected 1 dominant, got {len(dominant)}"
        assert abs(dominant[0].frequency - 0.05) < 0.005

    def test_residual_is_small_for_clean_signal(self):
        signal = _make_signal([0.1], [2.0], [1.0], n=500)
        result = extract(signal)

        assert result.noise_std < 0.1, f"Residual too large: {result.noise_std}"


class TestMultipleFrequencies:
    """Extract multiple sinusoidal components."""

    def test_two_frequencies(self):
        signal = _make_signal([0.05, 0.12], [3.0, 2.0], [0.0, 0.5], n=1000)
        result = extract(signal)

        assert len(result.components) >= 2
        found_freqs = sorted([c.frequency for c in result.components[:2]])
        assert abs(found_freqs[0] - 0.05) < 0.005
        assert abs(found_freqs[1] - 0.12) < 0.005

    def test_three_frequencies_with_noise(self):
        signal = _make_signal(
            [0.03, 0.08, 0.15], [5.0, 3.0, 2.0], [0.0, 1.0, 2.0],
            n=2000, noise_std=0.5,
        )
        result = extract(signal, min_snr=2.0)

        found_freqs = sorted([c.frequency for c in result.components[:3]])
        assert len(found_freqs) >= 3
        assert abs(found_freqs[0] - 0.03) < 0.003
        assert abs(found_freqs[1] - 0.08) < 0.005
        assert abs(found_freqs[2] - 0.15) < 0.005

    def test_close_frequencies_not_merged(self):
        """Two frequencies that are close but distinct should both be found."""
        signal = _make_signal([0.10, 0.12], [3.0, 3.0], [0.0, 0.0], n=2000)
        result = extract(signal)

        found_freqs = sorted([c.frequency for c in result.components[:2]])
        assert len(found_freqs) >= 2
        assert abs(found_freqs[0] - 0.10) < 0.005
        assert abs(found_freqs[1] - 0.12) < 0.005


class TestEdgeCases:
    def test_pure_noise_extracts_few(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 500)
        result = extract(signal, min_snr=3.0)

        assert len(result.components) <= 2, "Should extract few from pure noise"

    def test_short_signal(self):
        signal = _make_signal([0.1], [1.0], [0.0], n=32)
        result = extract(signal)
        # Should not crash
        assert isinstance(result.components, list)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            extract(np.array([1.0, 2.0, 3.0]))

    def test_dc_component_ignored(self):
        """A constant offset should not be extracted as a periodic component."""
        signal = np.full(500, 10.0) + _make_signal([0.1], [2.0], [0.0], n=500)
        result = extract(signal)

        # All extracted frequencies should be > 0
        for c in result.components:
            assert c.frequency > 0.001, f"DC component extracted: freq={c.frequency}"
