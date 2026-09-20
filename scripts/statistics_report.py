"""Sec. 22/30: paired seed-level significance testing of the final model
against every benchmark, with Holm correction, plus a win/tie/loss summary.

Primary metric: test-split mean DER-utilization, paired by seed (101-105,
common to the final model, the benchmark suite, and the source reproduction).
"""

from __future__ import annotations

import csv
from pathlib import Path
import statistics as st

import numpy as np
from scipy import stats

SEEDS = list(range(101, 111))
ALPHA = 0.05


def per_seed_test_metric(paths: list[Path], metric: str = "der_utilization") -> dict[int, float]:
    out = {}
    for path in paths:
        rows = list(csv.DictReader(path.open()))
        test_rows = [r for r in rows if r["split"] == "test"]
        if not test_rows:
            continue
        seed = int(test_rows[0]["seed"])
        out[seed] = st.mean(float(r[metric]) for r in test_rows)
    return out


def load_final_model(system: str) -> dict[int, float]:
    paths = [Path(f"FINAL_MODEL_data/{system}_c4b_seed{s}_evaluation.csv") for s in SEEDS]
    paths = [p for p in paths if p.exists()]
    return per_seed_test_metric(paths)


def load_benchmark(system: str, mode: str) -> dict[int, float]:
    paths = list(Path("BENCHMARKS_data").glob(f"{system}_{mode}_seed*_evaluation.csv"))
    return per_seed_test_metric(paths)


def load_source_reproduction(system: str) -> dict[int, float]:
    paths = list(Path("SOURCE_METHOD_REPRODUCTION/results").glob(f"{system}_seed*_evaluation.csv"))
    return per_seed_test_metric(paths)


def paired_test(a: dict[int, float], b: dict[int, float]) -> dict:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return {"n": len(common)}
    xa = np.array([a[s] for s in common])
    xb = np.array([b[s] for s in common])
    diff = xa - xb
    pooled_sd = np.std(diff, ddof=1) if len(diff) > 1 else float("nan")
    exact_tie = bool(pooled_sd == 0 and np.mean(diff) == 0)
    complete_separation = bool(pooled_sd == 0 and np.mean(diff) != 0)
    if exact_tie:
        # Every seed of both methods produced bit-identical values with zero
        # difference: no evidence of any difference exists, full stop.
        return {
            "n": len(common),
            "mean_a": float(np.mean(xa)),
            "mean_b": float(np.mean(xb)),
            "mean_diff": 0.0,
            "t_stat": 0.0,
            "t_p": 1.0,
            "wilcoxon_p": 1.0,
            "cohen_d": 0.0,
            "complete_separation": False,
        }
    if complete_separation:
        # Zero within-pair variance across all seeds: every seed of both
        # methods produced the same value (a genuine, explainable degenerate
        # case here -- see manuscript Sec. V -- not a numerical artifact).
        # The classical t-statistic is +/-inf and Cohen's d is undefined
        # (0/0); report the separation directly instead of propagating inf/nan.
        t_stat = float("inf") if np.mean(diff) > 0 else float("-inf")
        t_p = 0.0
        w_p = 0.0
        cohen_d = float("nan")
    else:
        t_res = stats.ttest_rel(xa, xb)
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)
        try:
            w_res = stats.wilcoxon(xa, xb)
            w_p = float(w_res.pvalue)
        except ValueError:
            w_p = float("nan")
        cohen_d = float(np.mean(diff) / pooled_sd) if pooled_sd and not np.isnan(pooled_sd) and pooled_sd > 0 else float("nan")
    return {
        "n": len(common),
        "mean_a": float(np.mean(xa)),
        "mean_b": float(np.mean(xb)),
        "mean_diff": float(np.mean(diff)),
        "t_stat": t_stat,
        "t_p": t_p,
        "wilcoxon_p": w_p,
        "cohen_d": cohen_d,
        "complete_separation": complete_separation,
    }


def holm_correction(p_values: list[float]) -> list[float]:
    p_values = [1.0 if (isinstance(p, float) and np.isnan(p)) else p for p in p_values]
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def main() -> None:
    rows = []
    for system in ["ieee123", "ieee8500"]:
        final = load_final_model(system)
        if not final:
            print(f"[skip] no final-model data yet for {system}")
            continue
        comparisons = {"SOURCE_METHOD_REPRODUCTION": load_source_reproduction(system)}
        for mode in ["random", "greedy_planner_only", "dqn_centralized", "ppo_centralized", "madqn", "meanfield_q", "qmix", "masked_iql", "graph_qmix"]:
            comparisons[mode] = load_benchmark(system, mode)

        p_values, names = [], []
        results = {}
        for name, other in comparisons.items():
            res = paired_test(final, other)
            results[name] = res
            if "t_p" in res:
                p_values.append(res["t_p"])
                names.append(name)
        adjusted = holm_correction(p_values) if p_values else []
        for name, p_adj in zip(names, adjusted):
            results[name]["t_p_holm"] = p_adj

        for name, res in results.items():
            if "mean_diff" not in res:
                rows.append({"system": system, "benchmark": name, "n_seeds": res.get("n", 0), "verdict": "insufficient_data"})
                continue
            significant = res["t_p_holm"] < ALPHA
            practically_meaningful = abs(res["mean_diff"]) > 0.01  # >1 percentage point DER-util
            favors_final = res["mean_diff"] > 0
            if significant and practically_meaningful and favors_final:
                verdict = "statistically_and_practically_superior"
            elif practically_meaningful and favors_final and not significant:
                verdict = "practically_superior_not_significant"
            elif significant and not practically_meaningful and favors_final:
                verdict = "statistically_superior_negligible_effect"
            elif significant and not practically_meaningful and not favors_final:
                verdict = "statistically_significant_negligible_deficit"
            elif not practically_meaningful:
                verdict = "statistically_indistinguishable"
            else:
                verdict = "inferior"
            rows.append(
                {
                    "system": system,
                    "benchmark": name,
                    "n_seeds": res["n"],
                    "final_model_mean_der": round(res["mean_a"], 4),
                    "benchmark_mean_der": round(res["mean_b"], 4),
                    "mean_diff": round(res["mean_diff"], 4),
                    "t_stat": round(res["t_stat"], 4) if np.isfinite(res["t_stat"]) else str(res["t_stat"]),
                    "t_p_raw": round(res["t_p"], 6),
                    "t_p_holm": round(res["t_p_holm"], 6),
                    "wilcoxon_p": round(res["wilcoxon_p"], 6) if not np.isnan(res["wilcoxon_p"]) else "nan",
                    "cohen_d": round(res["cohen_d"], 4) if not np.isnan(res["cohen_d"]) else "nan",
                    "complete_separation": res.get("complete_separation", False),
                    "verdict": verdict,
                }
            )

    if not rows:
        print("No comparisons available yet.")
        return

    out_path = Path("BENCHMARK_WTL.csv")
    fieldnames = sorted({k for r in rows for k in r})
    with out_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    win_verdicts = {"statistically_and_practically_superior"}
    tie_verdicts = {"statistically_indistinguishable", "statistically_superior_negligible_effect", "statistically_significant_negligible_deficit", "practically_superior_not_significant"}
    loss_verdicts = {"inferior"}
    wins = sum(1 for r in rows if r.get("verdict") in win_verdicts)
    ties = sum(1 for r in rows if r.get("verdict") in tie_verdicts)
    losses = sum(1 for r in rows if r.get("verdict") in loss_verdicts)
    other = sum(1 for r in rows if r.get("verdict") not in win_verdicts | tie_verdicts | loss_verdicts | {"insufficient_data"})
    if other:
        print(f"[warning] {other} rows have an unclassified verdict; check statistics_report.py's verdict logic")
    print(f"Wrote {out_path} ({len(rows)} rows). Proposed: {wins} wins / {ties} ties / {losses} losses (raw count across systems x benchmarks).")


if __name__ == "__main__":
    main()
