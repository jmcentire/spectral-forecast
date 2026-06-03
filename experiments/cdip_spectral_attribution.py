"""Post-hoc CDIP spectral-shape attribution for surfaced anomaly windows.

This script fetches CDIP THREDDS wave spectra after the detector has already
surfaced windows. It is an attribution/control pass, not a detector input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, Sequence
from urllib.request import Request, urlopen

import numpy as np
from scipy import signal
from scipy.io import netcdf_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cdip_seastate_attribution import (  # noqa: E402
    compare_window_sets,
    select_window_sets,
)


CDIP_SPECTRUM_NOTEBOOK_URL = (
    "https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/spectrum_plot.html"
)
THREDDS_REALTIME_URL = (
    "https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/realtime/{station}p1_rt.nc"
)
THREDDS_HISTORIC_URL = (
    "https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/"
    "{station}p1/{station}p1_historic.nc"
)


@dataclass(frozen=True)
class SpectrumRow:
    station_id: str
    time: datetime
    frequency_hz: np.ndarray
    bandwidth_hz: np.ndarray
    energy_density: np.ndarray
    mean_direction_deg: np.ndarray | None = None
    directional_spread_deg: np.ndarray | None = None


@dataclass
class StationSpectra:
    station_id: str
    source: str
    time_epoch: np.ndarray
    frequency_hz: np.ndarray
    bandwidth_hz: np.ndarray
    energy_density: np.ndarray
    mean_direction_deg: np.ndarray | None
    directional_spread_deg: np.ndarray | None


class SpectraSource(Protocol):
    def rows(self, station_id: str, start: datetime, end: datetime) -> Sequence[SpectrumRow]:
        ...


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _station_id(platform_id: str) -> str:
    digits = []
    for char in str(platform_id):
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    if not digits:
        raise ValueError(f"Cannot derive CDIP station id from platform {platform_id!r}")
    return "".join(digits)


def _finite_positive(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values) & (values > 0.0), values, 0.0)


def _bandwidth_from_bounds_or_grid(nc: Any, frequency: np.ndarray) -> np.ndarray:
    if "waveBandwidth" in nc.variables:
        bandwidth = np.asarray(nc.variables["waveBandwidth"].data, dtype=np.float64)
        if bandwidth.shape == frequency.shape:
            return _finite_positive(bandwidth)
    if "waveFrequencyBounds" in nc.variables:
        bounds = np.asarray(nc.variables["waveFrequencyBounds"].data, dtype=np.float64)
        if bounds.ndim == 2 and bounds.shape[0] == len(frequency) and bounds.shape[1] >= 2:
            return _finite_positive(bounds[:, 1] - bounds[:, 0])
    if len(frequency) == 1:
        return np.ones_like(frequency, dtype=np.float64)
    edges = np.empty(len(frequency) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (frequency[:-1] + frequency[1:])
    edges[0] = max(0.0, frequency[0] - (edges[1] - frequency[0]))
    edges[-1] = frequency[-1] + (frequency[-1] - edges[-2])
    return _finite_positive(np.diff(edges))


def _numeric_var(nc: Any, name: str) -> np.ndarray | None:
    if name not in nc.variables:
        return None
    data = np.asarray(nc.variables[name].data, dtype=np.float64)
    return np.where(np.isfinite(data) & (data > -900.0), data, np.nan)


class NetcdfSpectraSource:
    """Read CDIP station spectra from cached THREDDS NetCDF files."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        dataset: str = "realtime",
        allow_historic_fallback: bool = False,
        timeout: float = 60.0,
    ) -> None:
        if dataset not in {"realtime", "historic"}:
            raise ValueError("dataset must be realtime or historic")
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.allow_historic_fallback = allow_historic_fallback
        self.timeout = timeout
        self._loaded: dict[tuple[str, str], StationSpectra | None] = {}

    def rows(self, station_id: str, start: datetime, end: datetime) -> Sequence[SpectrumRow]:
        station = self._load(station_id, self.dataset)
        rows = self._rows_from_station(station, start, end)
        if rows or not self.allow_historic_fallback or self.dataset == "historic":
            return rows
        historic = self._load(station_id, "historic")
        return self._rows_from_station(historic, start, end)

    def _load(self, station_id: str, dataset: str) -> StationSpectra | None:
        key = (dataset, station_id)
        if key in self._loaded:
            return self._loaded[key]
        path = self._path(station_id, dataset)
        if not path.exists():
            self._download(station_id, dataset, path)
        if not path.exists():
            self._loaded[key] = None
            return None
        try:
            spectra = self._read_station_file(station_id, dataset, path)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "spectra_station_read_failed",
                        "station_id": station_id,
                        "dataset": dataset,
                        "path": str(path),
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            spectra = None
        self._loaded[key] = spectra
        return spectra

    def _path(self, station_id: str, dataset: str) -> Path:
        suffix = "rt" if dataset == "realtime" else "historic"
        return self.cache_dir / f"{station_id}p1_{suffix}.nc"

    def _download(self, station_id: str, dataset: str, path: Path) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        url = (
            THREDDS_REALTIME_URL.format(station=station_id)
            if dataset == "realtime"
            else THREDDS_HISTORIC_URL.format(station=station_id)
        )
        print(
            json.dumps(
                {
                    "event": "spectra_download_start",
                    "station_id": station_id,
                    "dataset": dataset,
                    "url": url,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        part = path.with_suffix(path.suffix + ".part")
        try:
            request = Request(url, headers={"User-Agent": "spectral-forecast/0.4 cdip-spectra"})
            with urlopen(request, timeout=self.timeout) as response, part.open("wb") as fh:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            part.replace(path)
            print(
                json.dumps(
                    {
                        "event": "spectra_download_done",
                        "station_id": station_id,
                        "dataset": dataset,
                        "bytes": path.stat().st_size,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        except Exception as exc:
            if part.exists():
                part.unlink()
            print(
                json.dumps(
                    {
                        "event": "spectra_download_failed",
                        "station_id": station_id,
                        "dataset": dataset,
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )

    def _read_station_file(self, station_id: str, dataset: str, path: Path) -> StationSpectra:
        with netcdf_file(str(path), "r", mmap=False) as nc:
            time_epoch = np.asarray(nc.variables["waveTime"].data, dtype=np.float64).copy()
            frequency = np.asarray(nc.variables["waveFrequency"].data, dtype=np.float64).copy()
            bandwidth = _bandwidth_from_bounds_or_grid(nc, frequency).copy()
            energy = _numeric_var(nc, "waveEnergyDensity")
            if energy is None:
                raise ValueError("waveEnergyDensity missing")
            energy = energy.copy()
            mean_direction = _numeric_var(nc, "waveMeanDirection")
            if mean_direction is not None:
                mean_direction = mean_direction.copy()
            spread = _numeric_var(nc, "waveSpread")
            if spread is not None:
                spread = spread.copy()
        return StationSpectra(
            station_id=station_id,
            source=dataset,
            time_epoch=time_epoch,
            frequency_hz=frequency,
            bandwidth_hz=bandwidth,
            energy_density=energy,
            mean_direction_deg=mean_direction,
            directional_spread_deg=spread,
        )

    def _rows_from_station(
        self,
        spectra: StationSpectra | None,
        start: datetime,
        end: datetime,
    ) -> list[SpectrumRow]:
        if spectra is None:
            return []
        start_epoch = start.astimezone(UTC).timestamp()
        end_epoch = end.astimezone(UTC).timestamp()
        indices = np.where((spectra.time_epoch >= start_epoch) & (spectra.time_epoch <= end_epoch))[0]
        rows: list[SpectrumRow] = []
        for index in indices:
            rows.append(
                SpectrumRow(
                    station_id=spectra.station_id,
                    time=datetime.fromtimestamp(float(spectra.time_epoch[index]), tz=UTC),
                    frequency_hz=spectra.frequency_hz,
                    bandwidth_hz=spectra.bandwidth_hz,
                    energy_density=np.asarray(spectra.energy_density[index], dtype=np.float64),
                    mean_direction_deg=(
                        None
                        if spectra.mean_direction_deg is None
                        else np.asarray(spectra.mean_direction_deg[index], dtype=np.float64)
                    ),
                    directional_spread_deg=(
                        None
                        if spectra.directional_spread_deg is None
                        else np.asarray(spectra.directional_spread_deg[index], dtype=np.float64)
                    ),
                )
            )
        return rows


def _weighted_quantile_frequency(
    frequency: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float | None:
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    cumulative = np.cumsum(weights) / total
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    index = min(max(index, 0), len(frequency) - 1)
    return float(frequency[index])


def _peak_metrics(
    frequency: np.ndarray,
    density: np.ndarray,
    *,
    min_height_fraction: float,
    min_prominence_fraction: float,
) -> dict[str, float | int | None]:
    if len(density) < 3:
        return {
            "peak_count": 0,
            "primary_peak_frequency_hz": None,
            "primary_peak_period_s": None,
            "second_peak_ratio": None,
        }
    peak_height = float(np.nanmax(density))
    if not math.isfinite(peak_height) or peak_height <= 0.0:
        return {
            "peak_count": 0,
            "primary_peak_frequency_hz": None,
            "primary_peak_period_s": None,
            "second_peak_ratio": None,
        }
    peaks, properties = signal.find_peaks(
        density,
        height=peak_height * min_height_fraction,
        prominence=peak_height * min_prominence_fraction,
    )
    heights = np.asarray(properties.get("peak_heights", []), dtype=np.float64)
    if len(peaks) == 0:
        primary_index = int(np.nanargmax(density))
        return {
            "peak_count": 1,
            "primary_peak_frequency_hz": float(frequency[primary_index]),
            "primary_peak_period_s": float(1.0 / frequency[primary_index]),
            "second_peak_ratio": 0.0,
        }
    order = np.argsort(heights)[::-1]
    ordered_peaks = peaks[order]
    ordered_heights = heights[order]
    primary_index = int(ordered_peaks[0])
    second_ratio = (
        float(ordered_heights[1] / ordered_heights[0]) if len(ordered_heights) > 1 else 0.0
    )
    return {
        "peak_count": int(len(ordered_peaks)),
        "primary_peak_frequency_hz": float(frequency[primary_index]),
        "primary_peak_period_s": float(1.0 / frequency[primary_index]),
        "second_peak_ratio": second_ratio,
    }


def spectrum_metrics(
    row: SpectrumRow,
    *,
    peak_min_height_fraction: float = 0.15,
    peak_min_prominence_fraction: float = 0.10,
    multimodal_second_peak_ratio: float = 0.35,
) -> dict[str, float | int | None]:
    frequency = np.asarray(row.frequency_hz, dtype=np.float64)
    bandwidth = np.asarray(row.bandwidth_hz, dtype=np.float64)
    density = np.asarray(row.energy_density, dtype=np.float64)
    valid = (
        np.isfinite(frequency)
        & np.isfinite(bandwidth)
        & np.isfinite(density)
        & (frequency > 0.0)
        & (bandwidth > 0.0)
        & (density >= 0.0)
    )
    frequency = frequency[valid]
    bandwidth = bandwidth[valid]
    density = density[valid]
    if len(frequency) == 0:
        return {"records": 1, "usable_bins": 0}
    weights = density * bandwidth
    total_energy = float(np.sum(weights))
    if total_energy <= 0.0:
        return {"records": 1, "usable_bins": int(len(frequency)), "energy_total": 0.0}
    probability = weights / total_energy
    mean_frequency = float(np.sum(frequency * weights) / total_energy)
    frequency_variance = float(np.sum(((frequency - mean_frequency) ** 2) * weights) / total_energy)
    bandwidth_hz = math.sqrt(max(0.0, frequency_variance))
    q05 = _weighted_quantile_frequency(frequency, weights, 0.05)
    q95 = _weighted_quantile_frequency(frequency, weights, 0.95)
    entropy = float(-np.sum(probability * np.log(probability + 1e-300)) / math.log(len(probability)))
    concentration = float(np.sum(probability**2))
    peak = _peak_metrics(
        frequency,
        density,
        min_height_fraction=peak_min_height_fraction,
        min_prominence_fraction=peak_min_prominence_fraction,
    )
    second_ratio = float(peak.get("second_peak_ratio") or 0.0)
    peak_count = int(peak.get("peak_count") or 0)
    directional_spread = None
    if row.directional_spread_deg is not None:
        spread = np.asarray(row.directional_spread_deg, dtype=np.float64)[valid]
        finite = np.isfinite(spread)
        if np.any(finite):
            directional_spread = float(np.sum(spread[finite] * weights[finite]) / np.sum(weights[finite]))
    return {
        "records": 1,
        "usable_bins": int(len(frequency)),
        "energy_total": total_energy,
        "hm0_estimate": float(4.0 * math.sqrt(total_energy)),
        "mean_frequency_hz": mean_frequency,
        "mean_period_s": float(1.0 / mean_frequency) if mean_frequency > 0.0 else None,
        "bandwidth_hz": bandwidth_hz,
        "normalized_bandwidth": float(bandwidth_hz / mean_frequency) if mean_frequency > 0.0 else None,
        "energy_width_90_hz": float(q95 - q05) if q05 is not None and q95 is not None else None,
        "spectral_entropy": entropy,
        "spectral_concentration": concentration,
        "effective_bin_count": float(1.0 / concentration) if concentration > 0.0 else None,
        "primary_peak_frequency_hz": peak["primary_peak_frequency_hz"],
        "primary_peak_period_s": peak["primary_peak_period_s"],
        "peak_count": peak_count,
        "second_peak_ratio": second_ratio,
        "multimodal_candidate": float(peak_count >= 2 and second_ratio >= multimodal_second_peak_ratio),
        "directional_spread_energy_weighted_deg": directional_spread,
    }


def _finite(values: Sequence[float | int | None]) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            out.append(number)
    return out


def _mean(values: Sequence[float | int | None]) -> float | None:
    finite = _finite(values)
    if not finite:
        return None
    return float(np.mean(finite))


def _max(values: Sequence[float | int | None]) -> float | None:
    finite = _finite(values)
    if not finite:
        return None
    return float(np.max(finite))


def _station_summary(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "energy_total",
        "hm0_estimate",
        "mean_period_s",
        "bandwidth_hz",
        "normalized_bandwidth",
        "energy_width_90_hz",
        "spectral_entropy",
        "spectral_concentration",
        "effective_bin_count",
        "primary_peak_period_s",
        "peak_count",
        "second_peak_ratio",
        "multimodal_candidate",
        "directional_spread_energy_weighted_deg",
    ]
    out: dict[str, Any] = {"records": len(metrics)}
    for field in fields:
        values = [row.get(field) for row in metrics]
        out[f"{field}_mean"] = _mean(values)
        if field in {"peak_count", "second_peak_ratio", "multimodal_candidate"}:
            out[f"{field}_max"] = _max(values)
    return out


def _window_features(station_summaries: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    present = [row for row in station_summaries if int(row.get("records", 0)) > 0]
    fields = [
        "energy_total_mean",
        "hm0_estimate_mean",
        "mean_period_s_mean",
        "bandwidth_hz_mean",
        "normalized_bandwidth_mean",
        "energy_width_90_hz_mean",
        "spectral_entropy_mean",
        "spectral_concentration_mean",
        "effective_bin_count_mean",
        "primary_peak_period_s_mean",
        "peak_count_mean",
        "peak_count_max",
        "second_peak_ratio_mean",
        "second_peak_ratio_max",
        "multimodal_candidate_mean",
        "multimodal_candidate_max",
        "directional_spread_energy_weighted_deg_mean",
    ]
    out: dict[str, float | None] = {
        "stations_with_spectra": float(len(present)),
        "spectra_records": float(sum(int(row.get("records", 0)) for row in present)),
    }
    for field in fields:
        values = [row.get(field) for row in present]
        out[field] = _mean(values)
        if field.endswith("_max"):
            out[field] = _max(values)
    return out


def enrich_window(
    window: dict[str, Any],
    *,
    source: SpectraSource,
    pad_minutes: float,
    peak_min_height_fraction: float,
    peak_min_prominence_fraction: float,
    multimodal_second_peak_ratio: float,
) -> dict[str, Any]:
    start = _parse_time(str(window["start"]))
    end = _parse_time(str(window["end"]))
    query_start = start - timedelta(minutes=pad_minutes)
    query_end = end + timedelta(minutes=pad_minutes)
    station_summaries = []
    for platform in window.get("group", []):
        station = _station_id(str(platform))
        rows = source.rows(station, query_start, query_end)
        metrics = [
            spectrum_metrics(
                row,
                peak_min_height_fraction=peak_min_height_fraction,
                peak_min_prominence_fraction=peak_min_prominence_fraction,
                multimodal_second_peak_ratio=multimodal_second_peak_ratio,
            )
            for row in rows
        ]
        summary = _station_summary(metrics)
        summary["platform_id"] = platform
        summary["station_id"] = station
        station_summaries.append(summary)
    return {
        "global_window_index": window.get("global_window_index"),
        "selection_rank": window.get("selection_rank"),
        "start": window.get("start"),
        "end": window.get("end"),
        "group": window.get("group", []),
        "observed_minus_null": window.get("observed_minus_null"),
        "station_summaries": station_summaries,
        "features": _window_features(station_summaries),
    }


def _rank_comparisons(comparisons: dict[str, Any], *, limit: int = 12) -> list[dict[str, Any]]:
    rows = []
    for feature, comparison in comparisons.items():
        effect = comparison.get("standardized_mean_difference")
        if effect is None:
            continue
        rows.append(
            {
                "feature": feature,
                "standardized_mean_difference": effect,
                "mean_difference_high_minus_baseline": comparison.get(
                    "mean_difference_high_minus_baseline"
                ),
                "mann_whitney_u_p_two_sided": comparison.get("mann_whitney_u_p_two_sided"),
                "cliffs_delta_high_vs_baseline": comparison.get("cliffs_delta_high_vs_baseline"),
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["standardized_mean_difference"])), reverse=True)[
        :limit
    ]


def build_spectral_attribution(
    report: dict[str, Any],
    *,
    source: SpectraSource,
    top_windows: int,
    baseline_windows: int,
    pad_minutes: float,
    peak_min_height_fraction: float,
    peak_min_prominence_fraction: float,
    multimodal_second_peak_ratio: float,
    progress_every: int = 25,
) -> dict[str, Any]:
    selected = select_window_sets(
        report,
        top_windows=top_windows,
        baseline_windows=baseline_windows,
    )
    enriched: dict[str, list[dict[str, Any]]] = {}
    total = sum(len(rows) for rows in selected.values())
    seen = 0
    for name, rows in selected.items():
        enriched_rows = []
        for row in rows:
            seen += 1
            if progress_every > 0 and (seen == 1 or seen % progress_every == 0 or seen == total):
                print(
                    json.dumps(
                        {
                            "event": "spectral_attribution_progress",
                            "window_set": name,
                            "done": seen,
                            "total": total,
                            "start": row.get("start"),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
            enriched_rows.append(
                enrich_window(
                    row,
                    source=source,
                    pad_minutes=pad_minutes,
                    peak_min_height_fraction=peak_min_height_fraction,
                    peak_min_prominence_fraction=peak_min_prominence_fraction,
                    multimodal_second_peak_ratio=multimodal_second_peak_ratio,
                )
            )
        enriched[name] = enriched_rows

    comparisons = {
        "high_delta_vs_background_baseline": compare_window_sets(
            enriched["high_delta"],
            enriched["background_baseline"],
        ),
        "high_delta_vs_month_matched_baseline": compare_window_sets(
            enriched["high_delta"],
            enriched["month_matched_baseline"],
        ),
    }
    return {
        "source": {
            "role": "post-hoc attribution only; CDIP spectral arrays are not detector inputs",
            "cdip_spectrum_notebook_url": CDIP_SPECTRUM_NOTEBOOK_URL,
            "thredds_realtime_template": THREDDS_REALTIME_URL,
            "report_summary": {
                "windows_run": report.get("summary", {}).get("windows_run"),
                "observed_minus_null_total": report.get("summary", {}).get(
                    "observed_minus_null_total"
                ),
                "observed_total_z_effect_size": report.get("summary", {}).get("observed_total_z"),
                "empirical_p_floor": report.get("summary", {}).get(
                    "null_total_unique_empirical_p_floor"
                ),
            },
        },
        "selection": {
            "top_windows": len(enriched["high_delta"]),
            "background_baseline_windows": len(enriched["background_baseline"]),
            "month_matched_baseline_windows": len(enriched["month_matched_baseline"]),
            "pad_minutes": pad_minutes,
            "peak_min_height_fraction": peak_min_height_fraction,
            "peak_min_prominence_fraction": peak_min_prominence_fraction,
            "multimodal_second_peak_ratio": multimodal_second_peak_ratio,
        },
        "method_notes": [
            "Uses CDIP waveEnergyDensity, waveFrequency, and waveBandwidth from THREDDS NetCDF files",
            "Spectral bandwidth is the energy-weighted standard deviation in frequency",
            "Energy width 90 is the frequency span between the 5th and 95th cumulative energy percentiles",
            "Peak count uses scipy.signal.find_peaks on spectral density with relative height and prominence thresholds",
            "A multimodal candidate requires at least two peaks and a second/first peak-height ratio above the configured threshold",
            "Realtime station NetCDF coverage is deployment-bound; windows outside cached coverage are reported through missing spectra counts",
        ],
        "comparisons": comparisons,
        "ranked_feature_shifts": {
            name: _rank_comparisons(comparison)
            for name, comparison in comparisons.items()
        },
        "windows": enriched,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Merged CDIP batch report JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-windows", type=int, default=50)
    parser.add_argument("--baseline-windows", type=int, default=100)
    parser.add_argument("--pad-minutes", type=float, default=45.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/cdip_spectral_cache"))
    parser.add_argument("--dataset", choices=["realtime", "historic"], default="realtime")
    parser.add_argument("--allow-historic-fallback", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--peak-min-height-fraction", type=float, default=0.15)
    parser.add_argument("--peak-min-prominence-fraction", type=float, default=0.10)
    parser.add_argument("--multimodal-second-peak-ratio", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = json.loads(args.report.read_text())
    source = NetcdfSpectraSource(
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        allow_historic_fallback=args.allow_historic_fallback,
        timeout=args.timeout,
    )
    attribution = build_spectral_attribution(
        report,
        source=source,
        top_windows=args.top_windows,
        baseline_windows=args.baseline_windows,
        pad_minutes=args.pad_minutes,
        peak_min_height_fraction=args.peak_min_height_fraction,
        peak_min_prominence_fraction=args.peak_min_prominence_fraction,
        multimodal_second_peak_ratio=args.multimodal_second_peak_ratio,
        progress_every=args.progress_every,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(attribution, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "top_windows": attribution["selection"]["top_windows"],
                "background_baseline_windows": attribution["selection"][
                    "background_baseline_windows"
                ],
                "month_matched_baseline_windows": attribution["selection"][
                    "month_matched_baseline_windows"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
