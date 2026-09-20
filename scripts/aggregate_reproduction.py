from __future__ import annotations

import csv
from pathlib import Path
import statistics as st

RESULTS = Path("SOURCE_METHOD_REPRODUCTION/results")


def load_rows(system: str) -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob(f"{system}_seed*_evaluation.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append(row)
    return rows


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return st.mean(values), st.stdev(values)


def per_seed_metric(rows: list[dict], split: str, scenario_type: str | None, field: str) -> list[float]:
    seeds: dict[str, list[float]] = {}
    for row in rows:
        if row["split"] != split:
            continue
        if scenario_type is not None and row["scenario_type"] != scenario_type:
            continue
        seeds.setdefault(row["seed"], []).append(float(row[field]))
    return [st.mean(v) for v in seeds.values() if v]


def report_system(system: str) -> dict:
    rows = load_rows(system)
    n_seeds = len({r["seed"] for r in rows})
    out = {"system": system, "n_seeds": n_seeds}
    for label, split, stype in [
        ("test_single_fault", "test", "single_fault"),
        ("test_multi_fault", "test", "multiple_fault"),
        ("test_der_variation", "test", "der_variation"),
        ("validation_single_fault", "validation", "single_fault"),
    ]:
        for field in ["der_utilization", "total_load_fraction", "cumulative_reward", "violation_rate", "latency_ms"]:
            vals = per_seed_metric(rows, split, stype, field)
            mean, sd = summarize(vals)
            out[f"{label}__{field}_mean"] = mean
            out[f"{label}__{field}_sd"] = sd
    training_seconds = sorted({float(r["training_seconds"]) for r in rows})
    out["training_seconds_mean"] = st.mean(training_seconds) if training_seconds else float("nan")
    out["training_seconds_sd"] = st.stdev(training_seconds) if len(training_seconds) > 1 else 0.0
    return out


def main() -> None:
    for system in ["ieee123", "ieee8500"]:
        summary = report_system(system)
        print(f"=== {system} (n_seeds={summary['n_seeds']}) ===")
        for key in sorted(summary):
            if key in {"system", "n_seeds"}:
                continue
            print(f"{key}: {summary[key]:.4f}")
        print()


if __name__ == "__main__":
    main()
