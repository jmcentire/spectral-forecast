"""Tests for bounded CDIP shard launcher helpers."""

from pathlib import Path

import pytest

from experiments.cdip_sharded_run import (
    _batch_args_after_delimiter,
    build_shard_command,
    shard_paths,
    validate_concurrency,
)


def test_batch_args_rejects_wrapper_managed_flags():
    with pytest.raises(ValueError, match="managed"):
        _batch_args_after_delimiter(
            [
                "--",
                "data/cdip/001p1_xy.nc",
                "--preset",
                "scale",
                "--shard-index",
                "0",
            ]
        )


def test_validate_concurrency_rejects_estimated_memory_spike():
    with pytest.raises(ValueError, match="estimated peak"):
        validate_concurrency(
            shard_count=16,
            max_workers=16,
            estimated_shard_rss_gb=17.0,
            max_memory_fraction=0.60,
            total_memory_gb=128.0,
            allow_high_memory=False,
        )


def test_validate_concurrency_accepts_bounded_local_workers():
    validate_concurrency(
        shard_count=16,
        max_workers=2,
        estimated_shard_rss_gb=17.0,
        max_memory_fraction=0.60,
        total_memory_gb=128.0,
        allow_high_memory=False,
    )


def test_build_shard_command_adds_checkpoint_and_shard_flags(tmp_path):
    paths = shard_paths(tmp_path, 3)

    command = build_shard_command(
        python="python3",
        batch_script=Path("experiments/cdip_batch.py"),
        batch_args=["data/cdip/a_xy.nc", "--preset", "scale"],
        shard_count=16,
        shard_index=3,
        paths=paths,
    )

    assert command[:3] == ["python3", "experiments/cdip_batch.py", "data/cdip/a_xy.nc"]
    assert command[command.index("--shard-count") + 1] == "16"
    assert command[command.index("--shard-index") + 1] == "3"
    assert command[command.index("--checkpoint") + 1] == str(paths.checkpoint)
    assert command[-2:] == ["--format", "json"]
