"""spectral-forecast: Three-layer analytical time series forecasting."""

from spectral_forecast.extraction import extract, ExtractedComponent, ExtractionResult
from spectral_forecast.trend import fit_trend, TrendModel, TrendResult, TrendType
from spectral_forecast.shock import detect_shocks, ShockComponent, ShockResult, ShockShape
from spectral_forecast.forecast import SpectralForecaster, ForecastResult

__all__ = [
    "extract",
    "ExtractedComponent",
    "ExtractionResult",
    "fit_trend",
    "TrendModel",
    "TrendResult",
    "TrendType",
    "detect_shocks",
    "ShockComponent",
    "ShockResult",
    "ShockShape",
    "SpectralForecaster",
    "ForecastResult",
]
