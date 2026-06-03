"""spectral-forecast: Three-layer analytical time series forecasting."""

from spectral_forecast.extraction import extract, ExtractedComponent, ExtractionResult
from spectral_forecast.trend import fit_trend, TrendModel, TrendResult, TrendType
from spectral_forecast.shock import detect_shocks, ShockComponent, ShockResult, ShockShape
from spectral_forecast.local import fit_local, forecast_local, LocalModel, LocalResult
from spectral_forecast.wavelet import fit_wavelet, forecast_wavelet, WaveletModel, WaveletResult
from spectral_forecast.forecast import SpectralForecaster, ForecastResult
from spectral_forecast.information import (
    InformationReadiness,
    ReadinessScan,
    information_readiness,
    scan_information_readiness,
)
from spectral_forecast.observation import (
    DecompositionState,
    ObservationPoint,
    ObservationResult,
    StigmergyPoint,
    StigmergyResult,
    build_stigmergy,
    decomposition_state,
    observe_series,
)
from spectral_forecast.autotune import (
    AutoTuneConfig,
    AutoTuneNullSummary,
    AutoTuneResult,
    AutoTuneScore,
    default_autotune_configs,
    score_autotune_config,
    tune_observation,
)

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
    "InformationReadiness",
    "ReadinessScan",
    "information_readiness",
    "scan_information_readiness",
    "DecompositionState",
    "ObservationPoint",
    "ObservationResult",
    "StigmergyPoint",
    "StigmergyResult",
    "build_stigmergy",
    "decomposition_state",
    "observe_series",
    "AutoTuneConfig",
    "AutoTuneNullSummary",
    "AutoTuneResult",
    "AutoTuneScore",
    "default_autotune_configs",
    "score_autotune_config",
    "tune_observation",
]
