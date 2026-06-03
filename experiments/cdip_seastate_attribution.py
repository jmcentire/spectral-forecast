"""Post-hoc CDIP sea-state attribution for surfaced anomaly windows.

This script intentionally runs after the agnostic detector. It asks whether
high-delta windows share CDIP-published sea-state conditions without feeding
those conditions back into the detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from scipy import stats


WAVE_AGG_BASE_URL = "https://erddap.cdip.ucsd.edu/erddap/tabledap/wave_agg.csv"
WAVE_AGG_INFO_URL = "https://erddap.cdip.ucsd.edu/erddap/tabledap/wave_agg.html"
WAVE_AGG_COLUMNS = (
    "station_id",
    "time",
    "waveHs",
    "waveTp",
    "waveTa",
    "waveTz",
    "waveDp",
    "wavePeakPSD",
    "waveFlagPrimary",
)
@dataclass(frozen=True)
class WaveAggRow:
    station_id: str
    time: datetime
    waveHs: float | None
    waveTp: float | None
    waveTa: float | None
    waveTz: float | None
    waveDp: float | None
    wavePeakPSD: float | None
    waveFlagPrimary: float | None


WaveAggFetcher = Callable[[str, datetime, datetime], Sequence[WaveAggRow | Mapping[str, Any]]]


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_erddap_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _station_id(platform_id: str) -> str:
    match = re.match(r"^(\d+)", str(platform_id))
    if not match:
        raise ValueError(f"Cannot derive CDIP station id from platform {platform_id!r}")
    return match.group(1)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= -999.0:
        return None
    return out


def _row_from_mapping(row: WaveAggRow | Mapping[str, Any]) -> WaveAggRow:
    if isinstance(row, WaveAggRow):
        return row
    raw_time = row.get("time")
    if isinstance(raw_time, datetime):
        parsed_time = raw_time.astimezone(UTC)
    else:
        parsed_time = _parse_time(str(raw_time))
    return WaveAggRow(
        station_id=str(row.get("station_id", "")),
        time=parsed_time,
        waveHs=_float_or_none(row.get("waveHs")),
        waveTp=_float_or_none(row.get("waveTp")),
        waveTa=_float_or_none(row.get("waveTa")),
        waveTz=_float_or_none(row.get("waveTz")),
        waveDp=_float_or_none(row.get("waveDp")),
        wavePeakPSD=_float_or_none(row.get("wavePeakPSD")),
        waveFlagPrimary=_float_or_none(row.get("waveFlagPrimary")),
    )


def _wave_agg_url(station_id: str, start: datetime, end: datetime) -> str:
    query = (
        ",".join(WAVE_AGG_COLUMNS)
        + f'&station_id="{station_id}"'
        + f"&time>={_format_erddap_time(start)}"
        + f"&time<={_format_erddap_time(end)}"
        + "&waveFlagPrimary=1"
    )
    return WAVE_AGG_BASE_URL + "?" + quote(query, safe=',=&"<>:-TZ')


def _cache_path(cache_dir: Path, station_id: str, start: datetime, end: datetime) -> Path:
    start_key = start.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    end_key = end.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return cache_dir / f"wave_agg_{station_id}_{start_key}_{end_key}.csv"


def _parse_wave_agg_csv(text: str) -> list[WaveAggRow]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return []
    # ERDDAP CSV line 2 contains units, not data.
    reader = csv.DictReader([lines[0], *lines[2:]])
    rows: list[WaveAggRow] = []
    for row in reader:
        try:
            rows.append(_row_from_mapping(row))
        except Exception:
            continue
    return rows


def fetch_wave_agg_rows(
    station_id: str,
    start: datetime,
    end: datetime,
    *,
    cache_dir: Path,
    timeout: float = 30.0,
    retries: int = 2,
) -> list[WaveAggRow]:
    """Fetch CDIP-published wave_agg rows for one station/time interval."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, station_id, start, end)
    if path.exists():
        return _parse_wave_agg_csv(path.read_text())

    url = _wave_agg_url(station_id, start, end)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "spectral-forecast/0.4 cdip-attribution"})
            with urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
            path.write_text(text)
            return _parse_wave_agg_csv(text)
        except HTTPError as exc:
            if exc.code in {400, 404}:
                return []
            last_error = exc
        except URLError as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(1.0 + attempt)
    if last_error is not None:
        raise last_error
    return []


def _finite(values: Iterable[float | None]) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            out.append(number)
    return out


def circular_resultant_length(degrees: Sequence[float]) -> float | None:
    values = _finite(degrees)
    if not values:
        return None
    radians = np.deg2rad(values)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    return float(min(1.0, math.hypot(sin_mean, cos_mean)))


def circular_mean_degrees(degrees: Sequence[float]) -> float | None:
    values = _finite(degrees)
    if not values:
        return None
    radians = np.deg2rad(values)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    return float((math.degrees(math.atan2(sin_mean, cos_mean)) + 360.0) % 360.0)


def circular_spread_degrees(degrees: Sequence[float]) -> float | None:
    resultant = circular_resultant_length(degrees)
    if resultant is None:
        return None
    if resultant <= 1e-12:
        return 180.0
    return float(math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant)))))


def _mean(values: Iterable[float | None]) -> float | None:
    finite = _finite(values)
    if not finite:
        return None
    return float(np.mean(finite))


def _max(values: Iterable[float | None]) -> float | None:
    finite = _finite(values)
    if not finite:
        return None
    return float(np.max(finite))


def _std(values: Iterable[float | None]) -> float | None:
    finite = _finite(values)
    if len(finite) < 2:
        return None
    return float(np.std(finite, ddof=1))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _station_summary(rows: Sequence[WaveAggRow]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "waveHs_mean": _mean(row.waveHs for row in rows),
        "waveTp_mean": _mean(row.waveTp for row in rows),
        "waveTa_mean": _mean(row.waveTa for row in rows),
        "waveTz_mean": _mean(row.waveTz for row in rows),
        "wavePeakPSD_mean": _mean(row.wavePeakPSD for row in rows),
        "waveDp_mean_deg": circular_mean_degrees(
            [row.waveDp for row in rows if row.waveDp is not None]
        ),
        "waveDp_resultant_length": circular_resultant_length(
            [row.waveDp for row in rows if row.waveDp is not None]
        ),
        "waveDp_spread_deg": circular_spread_degrees(
            [row.waveDp for row in rows if row.waveDp is not None]
        ),
    }


def _window_features(station_summaries: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    present = [row for row in station_summaries if int(row.get("records", 0)) > 0]
    hs = [row.get("waveHs_mean") for row in present]
    tp = [row.get("waveTp_mean") for row in present]
    ta = [row.get("waveTa_mean") for row in present]
    tz = [row.get("waveTz_mean") for row in present]
    peak = [row.get("wavePeakPSD_mean") for row in present]
    station_mean_directions = [row.get("waveDp_mean_deg") for row in present]

    tp_ta_gap = [
        float(period) - float(mean_period)
        for period, mean_period in zip(tp, ta)
        if period is not None and mean_period is not None
    ]
    tp_tz_gap = [
        float(period) - float(zero_period)
        for period, zero_period in zip(tp, tz)
        if period is not None and zero_period is not None
    ]
    peak_per_hs2 = [
        _ratio(float(power), float(height) ** 2)
        for power, height in zip(peak, hs)
        if power is not None and height is not None
    ]
    station_direction_spreads = [row.get("waveDp_spread_deg") for row in present]
    station_direction_resultants = [row.get("waveDp_resultant_length") for row in present]

    return {
        "stations_with_wave_agg": float(len(present)),
        "wave_agg_records": float(sum(int(row.get("records", 0)) for row in present)),
        "waveHs_mean": _mean(hs),
        "waveHs_max": _max(hs),
        "waveHs_std_across_platforms": _std(hs),
        "waveTp_mean": _mean(tp),
        "waveTp_std_across_platforms": _std(tp),
        "waveTa_mean": _mean(ta),
        "waveTz_mean": _mean(tz),
        "tp_ta_gap_mean": _mean(tp_ta_gap),
        "tp_tz_gap_mean": _mean(tp_tz_gap),
        "tp_ta_ratio_mean": _mean(
            [
                _ratio(float(period), float(mean_period))
                for period, mean_period in zip(tp, ta)
                if period is not None and mean_period is not None
            ]
        ),
        "tp_tz_ratio_mean": _mean(
            [
                _ratio(float(period), float(zero_period))
                for period, zero_period in zip(tp, tz)
                if period is not None and zero_period is not None
            ]
        ),
        "wavePeakPSD_mean": _mean(peak),
        "wavePeakPSD_per_hs2_mean": _mean(peak_per_hs2),
        "station_waveDp_spread_deg_mean": _mean(station_direction_spreads),
        "station_waveDp_resultant_length_mean": _mean(station_direction_resultants),
        "waveDp_spread_deg_across_platforms": circular_spread_degrees(
            [float(value) for value in station_mean_directions if value is not None]
        ),
        "waveDp_resultant_length_across_platforms": circular_resultant_length(
            [float(value) for value in station_mean_directions if value is not None]
        ),
    }


def _select_evenly(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return [dict(row) for row in rows]
    indices = np.linspace(0, len(rows) - 1, num=count, dtype=int)
    seen: set[int] = set()
    out = []
    for index in indices:
        if int(index) in seen:
            continue
        seen.add(int(index))
        out.append(dict(rows[int(index)]))
    return out


def _window_id(row: dict[str, Any], fallback: int) -> str:
    value = row.get("global_window_index")
    if value is not None:
        return f"global:{value}"
    group = ",".join(str(item) for item in row.get("group", []))
    start = row.get("start")
    end = row.get("end")
    delta = row.get("observed_minus_null")
    if start is not None and end is not None:
        return f"window:{start}:{end}:{group}:{delta}"
    return f"fallback:{fallback}"


def select_window_sets(
    report: dict[str, Any],
    *,
    top_windows: int,
    baseline_windows: int,
) -> dict[str, list[dict[str, Any]]]:
    windows = [dict(row) for row in report.get("windows", [])]
    ranked = sorted(
        enumerate(windows),
        key=lambda item: float(item[1].get("observed_minus_null", 0.0)),
        reverse=True,
    )
    high_pairs = ranked[:top_windows]
    high_ids = {_window_id(row, index) for index, row in high_pairs}
    high = [dict(row) | {"selection_rank": rank} for rank, (_, row) in enumerate(high_pairs, 1)]

    background_pool = [
        row
        for index, row in enumerate(sorted(windows, key=lambda item: str(item.get("start", ""))))
        if _window_id(row, index) not in high_ids
    ]
    background = _select_evenly(background_pool, baseline_windows)

    month_pool: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(windows):
        if _window_id(row, index) in high_ids:
            continue
        start = str(row.get("start", ""))
        if len(start) >= 7:
            month_pool.setdefault(start[:7], []).append(dict(row))
    for rows in month_pool.values():
        rows.sort(key=lambda item: str(item.get("start", "")))

    used: set[str] = set()
    month_matched: list[dict[str, Any]] = []
    for _, high_row in high_pairs:
        high_start = _parse_time(str(high_row["start"]))
        candidates = month_pool.get(str(high_row.get("start", ""))[:7], [])
        best: tuple[float, int, dict[str, Any]] | None = None
        for candidate_index, candidate in enumerate(candidates):
            identity = _window_id(candidate, candidate_index)
            if identity in used:
                continue
            delta = abs((_parse_time(str(candidate["start"])) - high_start).total_seconds())
            if best is None or delta < best[0]:
                best = (delta, candidate_index, candidate)
        if best is None:
            continue
        used.add(_window_id(best[2], best[1]))
        month_matched.append(dict(best[2]))

    return {
        "high_delta": high,
        "background_baseline": background,
        "month_matched_baseline": month_matched,
    }


def enrich_window(
    window: dict[str, Any],
    *,
    fetcher: WaveAggFetcher,
    pad_minutes: float,
) -> dict[str, Any]:
    start = _parse_time(str(window["start"]))
    end = _parse_time(str(window["end"]))
    query_start = start - timedelta(minutes=pad_minutes)
    query_end = end + timedelta(minutes=pad_minutes)
    station_summaries = []
    for platform in window.get("group", []):
        station = _station_id(str(platform))
        rows = [_row_from_mapping(row) for row in fetcher(station, query_start, query_end)]
        rows = [row for row in rows if query_start <= row.time <= query_end]
        summary = _station_summary(rows)
        summary["platform_id"] = platform
        summary["station_id"] = station
        station_summaries.append(summary)
    features = _window_features(station_summaries)
    return {
        "global_window_index": window.get("global_window_index"),
        "selection_rank": window.get("selection_rank"),
        "start": window.get("start"),
        "end": window.get("end"),
        "group": window.get("group", []),
        "observed_minus_null": window.get("observed_minus_null"),
        "station_summaries": station_summaries,
        "features": features,
    }


def _feature_values(windows: Sequence[dict[str, Any]], feature: str) -> list[float]:
    values = []
    for window in windows:
        value = window.get("features", {}).get(feature)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(values, q))


def _cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right:
        return None
    greater = 0
    less = 0
    for left_value in left:
        for right_value in right:
            if left_value > right_value:
                greater += 1
            elif left_value < right_value:
                less += 1
    return float((greater - less) / (len(left) * len(right)))


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "q25": _quantile(values, 0.25),
        "q75": _quantile(values, 0.75),
    }


def compare_window_sets(
    high_windows: Sequence[dict[str, Any]],
    baseline_windows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    feature_names = sorted(
        {
            feature
            for window in [*high_windows, *baseline_windows]
            for feature in window.get("features", {})
        }
    )
    comparisons: dict[str, Any] = {}
    for feature in feature_names:
        high_values = _feature_values(high_windows, feature)
        baseline_values = _feature_values(baseline_windows, feature)
        pooled_std = (
            float(np.std([*high_values, *baseline_values], ddof=1))
            if len(high_values) + len(baseline_values) > 1
            else None
        )
        high_mean = float(np.mean(high_values)) if high_values else None
        baseline_mean = float(np.mean(baseline_values)) if baseline_values else None
        diff = (
            float(high_mean - baseline_mean)
            if high_mean is not None and baseline_mean is not None
            else None
        )
        try:
            p_value = (
                float(stats.mannwhitneyu(high_values, baseline_values, alternative="two-sided").pvalue)
                if high_values and baseline_values
                else None
            )
        except ValueError:
            p_value = None
        comparisons[feature] = {
            "high_delta": _summary(high_values),
            "baseline": _summary(baseline_values),
            "mean_difference_high_minus_baseline": diff,
            "standardized_mean_difference": (
                float(diff / pooled_std) if diff is not None and pooled_std not in {None, 0.0} else None
            ),
            "mann_whitney_u_p_two_sided": p_value,
            "cliffs_delta_high_vs_baseline": _cliffs_delta(high_values, baseline_values),
        }
    return comparisons


def _rank_comparisons(comparisons: Mapping[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
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


def build_attribution(
    report: dict[str, Any],
    *,
    top_windows: int,
    baseline_windows: int,
    pad_minutes: float,
    fetcher: WaveAggFetcher,
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
                            "event": "seastate_progress",
                            "window_set": name,
                            "done": seen,
                            "total": total,
                            "start": row.get("start"),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
            enriched_rows.append(enrich_window(row, fetcher=fetcher, pad_minutes=pad_minutes))
        enriched[name] = enriched_rows

    comparisons: dict[str, Any] = {}
    for baseline_name in ("background_baseline", "month_matched_baseline"):
        comparisons[f"high_delta_vs_{baseline_name}"] = compare_window_sets(
            enriched["high_delta"],
            enriched.get(baseline_name, []),
        )

    return {
        "source": {
            "cdip_wave_agg_url": WAVE_AGG_INFO_URL,
            "role": "post-hoc attribution only; CDIP wave_agg fields are not detector inputs",
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
        },
        "method_notes": [
            "wave_agg exposes significant height, periods, peak direction, peak PSD, and QC flags",
            "literal spectral bandwidth and directional spreading are not wave_agg columns",
            "period gaps/ratios and peak PSD normalized by Hs^2 are treated as published-parameter proxies",
            "cross-platform direction spread is circular spread across station peak directions inside the selected group",
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
    parser.add_argument("--top-windows", type=int, default=100)
    parser.add_argument("--baseline-windows", type=int, default=250)
    parser.add_argument("--pad-minutes", type=float, default=45.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/cdip_wave_agg_cache"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = json.loads(args.report.read_text())

    def fetcher(station_id: str, start: datetime, end: datetime) -> Sequence[WaveAggRow]:
        return fetch_wave_agg_rows(
            station_id,
            start,
            end,
            cache_dir=args.cache_dir,
            timeout=args.timeout,
        )

    attribution = build_attribution(
        report,
        top_windows=args.top_windows,
        baseline_windows=args.baseline_windows,
        pad_minutes=args.pad_minutes,
        fetcher=fetcher,
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
