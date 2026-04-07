"""Alternative forecasting models from speculative-forecast."""

from spectral_forecast.models.baseline import LinearARForecaster
from spectral_forecast.models.iterative import IterativeDecompositionForecaster
from spectral_forecast.models.stochastic_resonance import StochasticResonanceForecaster

__all__ = [
    "LinearARForecaster",
    "IterativeDecompositionForecaster",
    "StochasticResonanceForecaster",
]
