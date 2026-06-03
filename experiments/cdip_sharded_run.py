"""Run CDIP batch shards with bounded local concurrency.

This wrapper exists because each ``cdip_batch.py`` shard loads the selected
CDIP record set independently. A high shard count is fine for deterministic
partitioning, but launching every shard at once can spike local memory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


FORBIDDEN_BATCH_ARGS = {
    "--checkpoint",
    "--format",
    "--include-merge-state",
    "--merge-reports",
    "--shard-count",
    "--shard-index",
}


@dataclass(frozen=True)
class ShardPaths:
    report: Path
    log: Path
    checkpoint: Path


def _total_memory_gb() -> float | None:
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return int(output.strip()) / (1024**3)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024**3)
    except Exception:
        return None


def _batch_args_after_delimiter(args: Sequence[str]) -> list[str]:
    out = list(args)
    if out and out[0] == "--":
        out = out[1:]
    if not out:
        raise ValueError("Pass cdip_batch.py arguments after --")
    forbidden = sorted(set(out) & FORBIDDEN_BATCH_ARGS)
    if forbidden:
        raise ValueError(
            "These cdip_batch.py arguments are managed by cdip_sharded_run.py: "
            + ", ".join(forbidden)
        )
    return out


def validate_concurrency(
    *,
    shard_count: int,
    max_workers: int,
    estimated_shard_rss_gb: float,
    max_memory_fraction: float,
    total_memory_gb: float | None,
    allow_high_memory: bool,
) -> None:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if max_workers > shard_count:
        raise ValueError("--max-workers cannot exceed --shard-count")
    if estimated_shard_rss_gb <= 0:
        raise ValueError("--estimated-shard-rss-gb must be positive")
    if not 0 < max_memory_fraction <= 1:
        raise ValueError("--max-memory-fraction must be in (0, 1]")

    estimated_peak = max_workers * estimated_shard_rss_gb
    if total_memory_gb is None:
        if max_workers > 4 and not allow_high_memory:
            raise ValueError(
                "Refusing >4 local workers without --allow-high-memory because "
                "system memory could not be detected"
            )
        return

    allowed_peak = total_memory_gb * max_memory_fraction
    if estimated_peak > allowed_peak and not allow_high_memory:
        raise ValueError(
            "Refusing local shard concurrency: estimated peak %.1f GB exceeds "
            "%.0f%% of system memory %.1f GB. Lower --max-workers or pass "
            "--allow-high-memory explicitly."
            % (estimated_peak, max_memory_fraction * 100.0, total_memory_gb)
        )


def shard_paths(run_dir: Path, shard_index: int) -> ShardPaths:
    return ShardPaths(
        report=run_dir / f"shard_{shard_index}.json",
        log=run_dir / f"shard_{shard_index}.log",
        checkpoint=run_dir / f"shard_{shard_index}.checkpoint.json",
    )


def build_shard_command(
    *,
    python: str,
    batch_script: Path,
    batch_args: Sequence[str],
    shard_count: int,
    shard_index: int,
    paths: ShardPaths,
) -> list[str]:
    return [
        python,
        str(batch_script),
        *batch_args,
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
        "--include-merge-state",
        "--checkpoint",
        str(paths.checkpoint),
        "--format",
        "json",
    ]


def _checkpoint_progress(run_dir: Path) -> tuple[int, int, int]:
    processed = 0
    total = 0
    complete = 0
    for path in run_dir.glob("shard_*.checkpoint.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        processed += int(payload.get("next_window_index", 0))
        total += int(payload.get("total_windows", 0))
        complete += int(bool(payload.get("complete")))
    return processed, total, complete


def _print_monitor(run_dir: Path, *, started_at: float, shard_count: int, active: int) -> None:
    processed, total, complete = _checkpoint_progress(run_dir)
    elapsed = max(time.time() - started_at, 1e-9)
    rate = processed / elapsed if processed else 0.0
    eta = (total - processed) / rate if total and rate else 0.0
    print(
        "cdip_sharded_run progress processed=%d/%d complete_shards=%d/%d "
        "active=%d elapsed=%.1fs rate=%.3f_windows_s eta=%.1fs"
        % (processed, total, complete, shard_count, active, elapsed, rate, eta),
        file=sys.stderr,
        flush=True,
    )


def run_shards(args: argparse.Namespace, batch_args: Sequence[str]) -> Path:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    total_memory = _total_memory_gb()
    validate_concurrency(
        shard_count=args.shard_count,
        max_workers=args.max_workers,
        estimated_shard_rss_gb=args.estimated_shard_rss_gb,
        max_memory_fraction=args.max_memory_fraction,
        total_memory_gb=total_memory,
        allow_high_memory=args.allow_high_memory,
    )

    run_meta = {
        "batch_args": list(batch_args),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimated_shard_rss_gb": args.estimated_shard_rss_gb,
        "max_memory_fraction": args.max_memory_fraction,
        "max_workers": args.max_workers,
        "shard_count": args.shard_count,
        "total_memory_gb": total_memory,
    }
    (args.run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n")

    next_shard = 0
    active: dict[int, subprocess.Popen[bytes]] = {}
    open_files: dict[int, tuple[object, object]] = {}
    failures: list[int] = []
    started_at = time.time()
    last_monitor = 0.0

    while next_shard < args.shard_count or active:
        while next_shard < args.shard_count and len(active) < args.max_workers:
            paths = shard_paths(args.run_dir, next_shard)
            log_fh = paths.log.open("wb")
            report_fh = paths.report.open("wb")
            command = build_shard_command(
                python=args.python,
                batch_script=args.batch_script,
                batch_args=batch_args,
                shard_count=args.shard_count,
                shard_index=next_shard,
                paths=paths,
            )
            proc = subprocess.Popen(command, stdout=report_fh, stderr=log_fh)
            active[next_shard] = proc
            open_files[next_shard] = (report_fh, log_fh)
            next_shard += 1

        now = time.time()
        if args.monitor_every > 0 and now - last_monitor >= args.monitor_every:
            _print_monitor(
                args.run_dir,
                started_at=started_at,
                shard_count=args.shard_count,
                active=len(active),
            )
            last_monitor = now

        for shard_index, proc in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            report_fh, log_fh = open_files.pop(shard_index)
            report_fh.close()
            log_fh.close()
            del active[shard_index]
            if code != 0:
                failures.append(shard_index)

        if active:
            time.sleep(args.poll_seconds)

    _print_monitor(
        args.run_dir,
        started_at=started_at,
        shard_count=args.shard_count,
        active=0,
    )

    if failures:
        raise RuntimeError("Shard failures: " + ", ".join(str(item) for item in failures))

    if args.no_merge:
        return args.run_dir

    reports = [str(shard_paths(args.run_dir, index).report) for index in range(args.shard_count)]
    merged = args.run_dir / "merged.json"
    command = [
        args.python,
        str(args.batch_script),
        "--merge-reports",
        *reports,
        "--top-windows",
        str(args.top_windows),
        "--format",
        "json",
    ]
    with merged.open("wb") as fh:
        subprocess.check_call(command, stdout=fh)
    return merged


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory for shard outputs")
    parser.add_argument("--shard-count", type=int, required=True, help="Total deterministic shards")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum local shard processes")
    parser.add_argument("--estimated-shard-rss-gb", type=float, default=17.0)
    parser.add_argument("--max-memory-fraction", type=float, default=0.60)
    parser.add_argument("--allow-high-memory", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--batch-script",
        type=Path,
        default=Path(__file__).with_name("cdip_batch.py"),
    )
    parser.add_argument("--top-windows", type=int, default=100000)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--monitor-every", type=float, default=30.0)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("batch_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        batch_args = _batch_args_after_delimiter(args.batch_args)
        validate_concurrency(
            shard_count=args.shard_count,
            max_workers=args.max_workers,
            estimated_shard_rss_gb=args.estimated_shard_rss_gb,
            max_memory_fraction=args.max_memory_fraction,
            total_memory_gb=_total_memory_gb(),
            allow_high_memory=args.allow_high_memory,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args, batch_args


def main(argv: Sequence[str] | None = None) -> None:
    args, batch_args = parse_args(argv)
    output = run_shards(args, batch_args)
    print(output)


if __name__ == "__main__":
    main()
