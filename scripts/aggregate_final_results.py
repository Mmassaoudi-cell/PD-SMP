"""Produces LaTeX-ready rows for every revised manuscript table directly from
the per-seed CSVs, to avoid manual transcription arithmetic errors (the
review round caught several of exactly this kind). Run after all reruns
(FINAL_MODEL_data, BENCHMARKS_data, SOURCE_METHOD_REPRODUCTION/results,
ABLATION_data/*_matched, ROBUSTNESS_data) complete with 10 seeds (101-110)
for main/ablation/efficiency and 5 seeds (101-105) for robustness.
"""

from __future__ import annotations

import csv
from pathlib import Path
import statistics as st
from collections import defaultdict

MAIN_SEEDS = list(range(101, 111))
ROBUST_SEEDS = list(range(101, 106))

LABELS = [
    ("random", "Random"),
    ("dqn_centralized", "DQN"),
    ("ppo_centralized", "PPO"),
    ("madqn", "MADQN"),
    ("meanfield_q", "Mean-Field Q"),
    ("qmix", "QMIX"),
    ("graph_qmix", "Graph-QMIX"),
    ("__source__", "Source reconstruction"),
    ("masked_iql", "Masked IQL"),
    ("greedy_planner_only", "Greedy planner (no RL)"),
]


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        if p.exists():
            rows.extend(csv.DictReader(p.open()))
    return rows


def bench_paths(system: str, mode: str) -> list[Path]:
    return sorted(Path("BENCHMARKS_data").glob(f"{system}_{mode}_seed*_evaluation.csv"))


def source_paths(system: str) -> list[Path]:
    return sorted(Path("SOURCE_METHOD_REPRODUCTION/results").glob(f"{system}_seed*_evaluation.csv"))


def final_paths(system: str) -> list[Path]:
    return [Path(f"FINAL_MODEL_data/{system}_c4b_seed{s}_evaluation.csv") for s in MAIN_SEEDS]


def mean_by_seed(rows: list[dict], field: str, split: str = "test") -> dict[int, float]:
    by_seed = defaultdict(list)
    for r in rows:
        if r["split"] != split:
            continue
        by_seed[int(r["seed"])].append(float(r[field]))
    return {s: st.mean(v) for s, v in by_seed.items() if v}


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def table_main() -> None:
    print("=== Table: main comparison (DER util / load fraction / steps), test split, 10 seeds ===")
    for system in ["ieee123", "ieee8500"]:
        print(f"-- {system} --")
        for key, label in LABELS:
            paths = source_paths(system) if key == "__source__" else bench_paths(system, key)
            rows = load_rows(paths)
            der = mean_by_seed(rows, "der_utilization")
            load = mean_by_seed(rows, "total_load_fraction")
            steps = mean_by_seed(rows, "steps")
            if not der:
                print(f"{label}: NO DATA ({len(rows)} rows)")
                continue
            print(f"{label}: DER={fmt(st.mean(der.values()))} Load={fmt(st.mean(load.values()))} Steps={fmt(st.mean(steps.values()),1)} n_seeds={len(der)}")
        rows = load_rows(final_paths(system))
        der = mean_by_seed(rows, "der_utilization")
        load = mean_by_seed(rows, "total_load_fraction")
        steps = mean_by_seed(rows, "steps")
        if der:
            print(f"PD-SMP: DER={fmt(st.mean(der.values()))} Load={fmt(st.mean(load.values()))} Steps={fmt(st.mean(steps.values()),1)} n_seeds={len(der)}")
        else:
            print(f"PD-SMP: NO DATA ({len(rows)} rows)")


def table_violation_rate() -> None:
    print("\n=== Violation rate, test split, 10 seeds ===")
    for system in ["ieee123", "ieee8500"]:
        print(f"-- {system} --")
        for key, label in LABELS + [("__final__", "PD-SMP")]:
            paths = (
                final_paths(system) if key == "__final__" else source_paths(system) if key == "__source__" else bench_paths(system, key)
            )
            rows = load_rows(paths)
            viol = mean_by_seed(rows, "violation_rate")
            if viol:
                print(f"{label}: viol={fmt(st.mean(viol.values()))}")


def table_causes() -> None:
    print("\n=== Violation cause breakdown (mean count per episode), test split, 10 seeds ===")
    for system in ["ieee123", "ieee8500"]:
        print(f"-- {system} --")
        for key, label in [("dqn_centralized", "DQN"), ("ppo_centralized", "PPO"), ("madqn", "MADQN"), ("meanfield_q", "Mean-Field Q"), ("qmix", "QMIX"), ("graph_qmix", "Graph-QMIX"), ("__source__", "Source reconstruction")]:
            paths = source_paths(system) if key == "__source__" else bench_paths(system, key)
            rows = load_rows(paths)
            if not rows or "cycle_violations" not in rows[0]:
                print(f"{label}: NO CAUSE DATA")
                continue
            cyc = mean_by_seed(rows, "cycle_violations")
            ill = mean_by_seed(rows, "invalid_semantics_violations")
            dup = mean_by_seed(rows, "duplicate_target_violations")
            print(f"{label}: cyc={fmt(st.mean(cyc.values()),3)} illeg={fmt(st.mean(ill.values()),3)} dup={fmt(st.mean(dup.values()),3)}")


def table_ablation() -> None:
    print("\n=== Ablation, IEEE-123 test split, 10 seeds, matched budget ===")
    configs = [
        ("Full model (PD-SMP)", final_paths("ieee123")),
        ("- behavior cloning", [Path(f"ABLATION_data/no_bc_matched/ieee123_c4b_seed{s}_evaluation.csv") for s in MAIN_SEEDS]),
        ("- RL fine-tuning", [Path(f"ABLATION_data/bc_only_matched/ieee123_c4b_seed{s}_evaluation.csv") for s in MAIN_SEEDS]),
        ("- joint-radiality mask", [Path(f"ABLATION_data/no_joint_mask_matched/ieee123_c4_seed{s}_evaluation.csv") for s in MAIN_SEEDS]),
    ]
    for label, paths in configs:
        rows = load_rows(paths)
        der = mean_by_seed(rows, "der_utilization")
        viol = mean_by_seed(rows, "violation_rate")
        if der:
            print(f"{label}: DER={fmt(st.mean(der.values()))} viol={fmt(st.mean(viol.values()))} n_seeds={len(der)}")
        else:
            print(f"{label}: NO DATA ({len(rows)} rows found)")


def table_robustness() -> None:
    print("\n=== Robustness, test split, 5 seeds ===")
    for system in ["ieee123", "ieee8500"]:
        print(f"-- {system} --")
        by_cond = defaultdict(list)
        for path in Path("ROBUSTNESS_data").glob(f"{system}_c4b_seed*_robustness.csv"):
            for row in csv.DictReader(path.open()):
                by_cond[row["condition"]].append(row)
        order = ["clean", "fault_noise_p0.1", "fault_noise_p0.3", "load_noise_sigma0.1", "load_noise_sigma0.3"]
        for c in order:
            rows = by_cond.get(c, [])
            if not rows:
                print(f"{c}: NO DATA")
                continue
            der = st.mean(float(r["der_utilization"]) for r in rows)
            viol = st.mean(float(r["violation_rate"]) for r in rows)
            print(f"{c}: der={fmt(der)} viol={fmt(viol)} n={len(rows)}")


def table_efficiency() -> None:
    print("\n=== Efficiency: params and latency, both systems, 10 seeds ===")
    for system in ["ieee123", "ieee8500"]:
        print(f"-- {system} --")
        for key, label in LABELS + [("__final__", "PD-SMP")]:
            paths = (
                final_paths(system) if key == "__final__" else source_paths(system) if key == "__source__" else bench_paths(system, key)
            )
            rows = load_rows(paths)
            if not rows:
                print(f"{label}: NO DATA")
                continue
            test_rows = [r for r in rows if r["split"] == "test"]
            lat = st.mean(float(r["latency_ms"]) for r in test_rows)
            params = int(test_rows[0].get("n_params", -1))
            print(f"{label}: params={params} latency_ms={fmt(lat,2)}")


def wtl_summary() -> None:
    path = Path("BENCHMARK_WTL.csv")
    if not path.exists():
        print("\n=== WTL: not yet generated (run scripts/statistics_report.py) ===")
        return
    rows = list(csv.DictReader(path.open()))
    wins = sum(1 for r in rows if r.get("verdict", "").endswith("superior"))
    ties = sum(1 for r in rows if r.get("verdict") == "statistically_indistinguishable")
    losses = sum(1 for r in rows if r.get("verdict") == "inferior")
    print(f"\n=== WTL: {wins} wins / {ties} ties / {losses} losses / {len(rows)} total comparisons ===")


if __name__ == "__main__":
    table_main()
    table_violation_rate()
    table_causes()
    table_ablation()
    table_robustness()
    table_efficiency()
    wtl_summary()
