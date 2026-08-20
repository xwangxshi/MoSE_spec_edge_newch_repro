#!/usr/bin/env python3
"""Validate MoSE Table 1 runs without trusting a zero process exit code."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def load_stats(path: Path, last_epoch: int, metric: str) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"missing stats file: {path}")

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epochs = [int(row["epoch"]) for row in rows]
    expected = list(range(last_epoch + 1))
    if epochs != expected:
        raise RuntimeError(
            f"{path}: expected exactly epochs 0..{last_epoch}; "
            f"found {len(rows)} rows with first={epochs[:3]} "
            f"and last={epochs[-3:]}"
        )

    for row in rows:
        value = row.get(metric)
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(
                f"{path}: missing/non-finite {metric} at epoch {row['epoch']}"
            )
    return rows


def verify_seed(
    run_dir: Path,
    last_epoch: int,
    metric: str,
    direction: str,
) -> dict:
    records = {
        split: load_stats(run_dir / split / "stats.json", last_epoch, metric)
        for split in ("train", "val", "test")
    }

    checkpoint = run_dir / "ckpt" / f"{last_epoch}.ckpt"
    if not checkpoint.is_file():
        raise RuntimeError(f"missing final checkpoint: {checkpoint}")

    values = [float(row[metric]) for row in records["val"]]
    best_value = min(values) if direction == "argmin" else max(values)
    best_epoch = values.index(best_value)  # NumPy argmin/argmax also keeps first tie.
    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        f"val_{metric}": float(records["val"][best_epoch][metric]),
        f"test_{metric}": float(records["test"][best_epoch][metric]),
    }


def verify_aggregate(
    run_parent: Path,
    summaries: list[dict],
    metric: str,
) -> dict:
    path = run_parent / "agg" / "test" / "best.json"
    if not path.is_file():
        raise RuntimeError(f"missing aggregate file: {path}")
    aggregate = json.loads(path.read_text(encoding="utf-8"))

    key = f"test_{metric}"
    values = [summary[key] for summary in summaries]
    expected_mean = round(mean(values), 5)
    expected_std = round(pstdev(values), 5)
    actual_mean = float(aggregate[metric])
    actual_std = float(aggregate[f"{metric}_std"])
    if not math.isclose(actual_mean, expected_mean, abs_tol=5e-6):
        raise RuntimeError(
            f"{path}: {metric}={actual_mean}, expected {expected_mean}"
        )
    if not math.isclose(actual_std, expected_std, abs_tol=5e-6):
        raise RuntimeError(
            f"{path}: {metric}_std={actual_std}, expected {expected_std}"
        )
    return {
        "aggregate_path": str(path),
        metric: actual_mean,
        f"{metric}_std": actual_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_parent", type=Path)
    parser.add_argument("last_epoch", type=int)
    parser.add_argument("metric")
    parser.add_argument("direction", choices=("argmin", "argmax"))
    parser.add_argument("seeds", nargs="+", type=int)
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="also require agg/test/best.json to match all listed seeds",
    )
    args = parser.parse_args()

    summaries = [
        verify_seed(
            args.run_parent / str(seed),
            args.last_epoch,
            args.metric,
            args.direction,
        )
        for seed in args.seeds
    ]
    result: dict = {"seeds": summaries}
    if args.aggregate:
        result["aggregate"] = verify_aggregate(
            args.run_parent, summaries, args.metric
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
