"""Paired moving-block bootstrap for six MCC comparisons.
Manifest: JSON list of name, full and baseline CSV paths, relative to the manifest directory.
Resample whole trading dates; pool observations per seed, then average MCC differences across seeds."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Counts = Tuple[int, int, int, int]
RowKey = Tuple[int, str, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def add_counts(left: Counts, right: Counts) -> Counts:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def mcc(counts: Counts) -> float:
    tp, tn, fp, fn = counts
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def resolve_prediction_paths(specification: object, base: Path | None = None) -> List[Path]:
    patterns = specification if isinstance(specification, list) else [specification]
    paths: List[Path] = []
    for pattern in patterns:
        if base is not None and not Path(str(pattern)).is_absolute():
            pattern = base / str(pattern)
        matches = [Path(value) for value in glob.glob(str(pattern))]
        if not matches:
            raise FileNotFoundError(f"No prediction files match {pattern!r}")
        paths.extend(matches)
    return sorted(set(paths))


def load_predictions(
    specification: object, base: Path | None = None
) -> Tuple[Dict[Tuple[int, str], Counts], Dict[RowKey, Tuple[int, str, str]]]:
    grouped: Dict[Tuple[int, str], Counts] = defaultdict(lambda: (0, 0, 0, 0))
    records: Dict[RowKey, Tuple[int, str, str]] = {}
    for path in resolve_prediction_paths(specification, base):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row_key = (
                    int(row["seed"]),
                    row["date"],
                    int(row["horizon"]),
                    int(row["stock_index"]),
                )
                if row_key in records:
                    raise ValueError(f"Duplicate prediction row {row_key} in {path}")
                seed, date = row_key[:2]
                truth = int(row["y_true"])
                prediction = int(row["y_pred"])
                if truth not in (0, 1) or prediction not in (0, 1):
                    raise ValueError(f"Labels must be binary at {row_key} in {path}")
                records[row_key] = (truth, row.get("stock_symbol", ""), row.get("split", ""))
                increment: Counts = (
                    int(truth == 1 and prediction == 1),
                    int(truth == 0 and prediction == 0),
                    int(truth == 0 and prediction == 1),
                    int(truth == 1 and prediction == 0),
                )
                grouped[(seed, date)] = add_counts(grouped[(seed, date)], increment)
    if not grouped:
        raise ValueError("No prediction rows found")
    return dict(grouped), records


def load_daily_counts(specification: object) -> Dict[Tuple[int, str], Counts]:
    return load_predictions(specification)[0]


def moving_block_indices(length: int, block_length: int, rng: random.Random) -> List[int]:
    if block_length <= 0 or block_length > length:
        raise ValueError("block_length must be between 1 and the number of test dates")
    starts = list(range(length - block_length + 1))
    sampled: List[int] = []
    while len(sampled) < length:
        start = rng.choice(starts)
        sampled.extend(range(start, start + block_length))
    return sampled[:length]


def paired_difference(
    full: Mapping[Tuple[int, str], Counts],
    baseline: Mapping[Tuple[int, str], Counts],
    dates: Sequence[str],
    seeds: Sequence[int],
    sampled_indices: Iterable[int],
) -> float:
    selected_dates = [dates[index] for index in sampled_indices]
    seed_differences = []
    for seed in seeds:
        full_counts: Counts = (0, 0, 0, 0)
        baseline_counts: Counts = (0, 0, 0, 0)
        for date in selected_dates:
            full_counts = add_counts(full_counts, full[(seed, date)])
            baseline_counts = add_counts(baseline_counts, baseline[(seed, date)])
        seed_differences.append(mcc(full_counts) - mcc(baseline_counts))
    return sum(seed_differences) / len(seed_differences)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_comparison(
    full_path: object,
    baseline_path: object,
    block_length: int,
    repetitions: int,
    rng: random.Random,
    base: Path | None = None,
) -> Dict[str, float]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    full, full_records = load_predictions(full_path, base)
    baseline, baseline_records = load_predictions(baseline_path, base)
    if full_records != baseline_records:
        raise ValueError(
            "Paired files must have identical seed, date, horizon, stock, label and split records"
        )
    if set(full) != set(baseline):
        raise ValueError("Paired files must contain identical seed-date keys")
    seeds = sorted({seed for seed, _ in full})
    dates = sorted({date for _, date in full})
    reference = {
        key[1:]: value for key, value in full_records.items() if key[0] == seeds[0]
    }
    for seed in seeds[1:]:
        observations = {
            key[1:]: value for key, value in full_records.items() if key[0] == seed
        }
        if observations != reference:
            raise ValueError("Every seed must evaluate the same observations and labels")
    expected = {(seed, date) for seed in seeds for date in dates}
    if set(full) != expected:
        raise ValueError("Each random seed must contain the same ordered test dates")

    observed = paired_difference(
        full, baseline, dates, seeds, range(len(dates))
    )
    replicates = [
        paired_difference(
            full,
            baseline,
            dates,
            seeds,
            moving_block_indices(len(dates), block_length, rng),
        )
        for _ in range(repetitions)
    ]
    non_positive = sum(value <= 0.0 for value in replicates)
    non_negative = sum(value >= 0.0 for value in replicates)
    p_value = min(1.0, 2.0 * (min(non_positive, non_negative) + 1) / (repetitions + 1))
    return {
        "delta_mcc": observed,
        "ci_low": percentile(replicates, 0.025),
        "ci_high": percentile(replicates, 0.975),
        "p_raw": p_value,
        "dates": float(len(dates)),
        "seeds": float(len(seeds)),
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(manifest) != 6:
        raise ValueError("The primary-test manifest must contain exactly six comparisons")
    rng = random.Random(args.seed)
    rows = []
    for item in manifest:
        result = bootstrap_comparison(
            item["full"],
            item["baseline"],
            args.block_length,
            args.repetitions,
            rng,
            base=args.manifest.resolve().parent,
        )
        rows.append({"comparison": item["name"], **result})
    adjusted = holm_adjust([float(row["p_raw"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
        row["block_length"] = args.block_length
        row["repetitions"] = args.repetitions

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
