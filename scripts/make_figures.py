"""Sec. 31: publication figures generated directly from CSV result files.
Run after FINAL_MODEL_data, BENCHMARKS_data, ABLATION_data, and
ROBUSTNESS_data are complete. Writes PNG (for quick inspection) and PDF (for
LaTeX inclusion) into manuscript/figures/.
"""

from __future__ import annotations

import csv
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("manuscript/figures")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [101, 102, 103, 104, 105]

BENCH_LABELS = {
    "random": "Random",
    "greedy_planner_only": "Greedy planner (no learning)",
    "dqn_centralized": "DQN",
    "ppo_centralized": "PPO",
    "madqn": "MADQN",
    "meanfield_q": "Mean-Field Q",
    "qmix": "QMIX",
    "masked_iql": "Masked IQL",
    "graph_qmix": "Graph-QMIX",
}


def per_seed_test(paths, metric="der_utilization"):
    out = {}
    for path in paths:
        rows = list(csv.DictReader(path.open()))
        test_rows = [r for r in rows if r["split"] == "test"]
        if not test_rows:
            continue
        seed = int(test_rows[0]["seed"])
        out[seed] = st.mean(float(r[metric]) for r in test_rows)
    return out


def load_final(system):
    return per_seed_test([Path(f"FINAL_MODEL_data/{system}_c4b_seed{s}_evaluation.csv") for s in SEEDS if Path(f"FINAL_MODEL_data/{system}_c4b_seed{s}_evaluation.csv").exists()])


def load_bench(system, mode):
    return per_seed_test(list(Path("BENCHMARKS_data").glob(f"{system}_{mode}_seed*_evaluation.csv")))


def load_source(system):
    return per_seed_test(list(Path("SOURCE_METHOD_REPRODUCTION/results").glob(f"{system}_seed*_evaluation.csv")))


def fig_main_comparison(system: str):
    final = load_final(system)
    if not final:
        print(f"[skip fig3-{system}] no final-model data yet")
        return
    series = {"PD-SMP (proposed)": final, "Source reconstruction": load_source(system)}
    for mode, label in BENCH_LABELS.items():
        vals = load_bench(system, mode)
        if vals:
            series[label] = vals
    names = list(series.keys())
    means = [st.mean(series[n].values()) for n in names]
    stds = [st.pstdev(series[n].values()) if len(series[n]) > 1 else 0.0 for n in names]
    order = sorted(range(len(names)), key=lambda i: -means[i])
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    colors = ["#2a9d8f" if n == "PD-SMP (proposed)" else ("#e76f51" if n == "Source reconstruction" else "#8d99ae") for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, means, xerr=stds, color=colors)
    ax.set_xlabel("Test-split mean DER-utilization")
    ax.set_title(f"Main benchmark comparison ({system})")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / f"fig3_main_comparison_{system}.pdf")
    fig.savefig(OUT / f"fig3_main_comparison_{system}.png", dpi=150)
    plt.close(fig)
    print(f"wrote fig3_main_comparison_{system}")


def fig_ablation():
    import json

    def summarize(pattern, metric="der_utilization", split="test"):
        vals = []
        for path in sorted(Path(".").glob(pattern)):
            rows = list(csv.DictReader(path.open()))
            val = [r for r in rows if r["split"] == split]
            if val:
                vals.append(st.mean(float(r[metric]) for r in val))
        return vals

    configs = [
        ("Full model", "FINAL_MODEL_data/ieee123_c4b_seed1*_evaluation.csv"),
        ("- behavior cloning", "ABLATION_data/no_bc_matched/ieee123_c4b_seed1*_evaluation.csv"),
        ("- RL fine-tune", "ABLATION_data/bc_only_matched/ieee123_c4b_seed1*_evaluation.csv"),
        ("- joint mask", "ABLATION_data/no_joint_mask_matched/ieee123_c4_seed1*_evaluation.csv"),
    ]
    labels, der_means, der_stds, viol_means = [], [], [], []
    for label, pattern in configs:
        vals = summarize(pattern, "der_utilization")
        viol = summarize(pattern, "violation_rate")
        if not vals:
            continue
        labels.append(label)
        der_means.append(st.mean(vals))
        der_stds.append(st.pstdev(vals) if len(vals) > 1 else 0.0)
        viol_means.append(st.mean(viol) if viol else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.bar(labels, der_means, yerr=der_stds, color="#2a9d8f")
    ax1.set_ylabel("Test DER-utilization")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(labels, viol_means, color="#e76f51")
    ax2.set_ylabel("Test violation rate")
    ax2.tick_params(axis="x", rotation=30)
    fig.suptitle("Ablation: IEEE-123, 10 seeds, matched budget")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_ablation.pdf")
    fig.savefig(OUT / "fig5_ablation.png", dpi=150)
    plt.close(fig)
    print("wrote fig5_ablation")


def fig_robustness(system: str):
    paths = list(Path("ROBUSTNESS_data").glob(f"{system}_c4b_seed*_robustness.csv"))
    if not paths:
        print(f"[skip fig-robustness-{system}] no data yet")
        return
    from collections import defaultdict

    by_cond = defaultdict(list)
    for path in paths:
        for row in csv.DictReader(path.open()):
            by_cond[row["condition"]].append(row)
    order = ["clean", "fault_noise_p0.1", "fault_noise_p0.3", "load_noise_sigma0.1", "load_noise_sigma0.3"]
    labels = [c for c in order if c in by_cond]
    der_means = [st.mean(float(r["der_utilization"]) for r in by_cond[c]) for c in labels]
    viol_means = [st.mean(float(r["violation_rate"]) for r in by_cond[c]) for c in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.bar(labels, der_means, color="#264653")
    ax1.set_ylabel("DER-utilization")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(labels, viol_means, color="#e76f51")
    ax2.set_ylabel("Violation rate")
    ax2.set_ylim(0, max(0.05, max(viol_means) * 1.2 if viol_means else 0.05))
    ax2.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Robustness under fault-telemetry and load-scale perturbation ({system})")
    fig.tight_layout()
    fig.savefig(OUT / f"fig4_robustness_{system}.pdf")
    fig.savefig(OUT / f"fig4_robustness_{system}.png", dpi=150)
    plt.close(fig)
    print(f"wrote fig4_robustness_{system}")


def fig_pareto(system: str):
    final = load_final(system)
    if not final:
        print(f"[skip fig6-pareto-{system}] no final-model data yet")
        return

    def latency_and_params(paths):
        lat, params = [], None
        for path in paths:
            rows = list(csv.DictReader(path.open()))
            test_rows = [r for r in rows if r["split"] == "test"]
            if test_rows:
                lat.append(st.mean(float(r["latency_ms"]) for r in test_rows))
                params = int(test_rows[0]["n_params"])
        return (st.mean(lat) if lat else float("nan")), params

    points = {}
    final_paths = [Path(f"FINAL_MODEL_data/{system}_c4b_seed{s}_evaluation.csv") for s in SEEDS if Path(f"FINAL_MODEL_data/{system}_c4b_seed{s}_evaluation.csv").exists()]
    lat, params = latency_and_params(final_paths)
    points["PD-SMP (proposed)"] = (lat, st.mean(final.values()))
    for mode, label in BENCH_LABELS.items():
        paths = list(Path("BENCHMARKS_data").glob(f"{system}_{mode}_seed*_evaluation.csv"))
        vals = per_seed_test(paths)
        if not vals:
            continue
        lat, params = latency_and_params(paths)
        points[label] = (lat, st.mean(vals.values()))

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, (lat, perf) in points.items():
        marker = "*" if label.startswith("PD-SMP") else "o"
        size = 200 if label.startswith("PD-SMP") else 80
        ax.scatter(lat, perf, s=size, marker=marker, label=label)
    ax.set_xlabel("Inference latency (ms/step)")
    ax.set_ylabel("Test DER-utilization")
    ax.set_title(f"Performance-latency Pareto ({system})")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"fig6_pareto_{system}.pdf")
    fig.savefig(OUT / f"fig6_pareto_{system}.png", dpi=150)
    plt.close(fig)
    print(f"wrote fig6_pareto_{system}")


if __name__ == "__main__":
    for system in ["ieee123", "ieee8500"]:
        fig_main_comparison(system)
        fig_robustness(system)
        fig_pareto(system)
    fig_ablation()
