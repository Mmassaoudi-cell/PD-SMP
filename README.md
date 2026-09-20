# PD-SMP: Planner-Distilled Sequential Masking for Feasible Multiagent Power Distribution Restoration

Source code for PD-SMP, a constraint-aware multiagent reinforcement learning
method for power distribution system restoration. Agents act sequentially
within each control step under an incremental joint-feasibility mask that
guarantees radiality and switching-legality by construction, with per-agent
policies trained by behavior cloning against a closed-form beam-search
planner and briefly fine-tuned with a single-critic advantage actor-critic
objective.

This repository contains the method implementation, the environment, every
benchmark baseline used for comparison, and every script needed to retrain
and re-evaluate the model end to end. Result artifacts (figures, tables,
trained checkpoints, and the manuscript) are intentionally not included here;
running the scripts below regenerates them.

## Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

See `requirements.txt` for exact package versions. Set `PYTHONPATH` to `src`
before running any script:

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src').Path
```

## Repository layout

| Path | Contents |
|---|---|
| `src/restoration/environment.py` | Restoration environment: feasibility oracle (radiality + switching legality), reward, metrics |
| `src/restoration/source_model.py` | Reconstruction of the comparison baseline's discriminator-based soft actor-critic architecture |
| `src/restoration/candidates.py` | PD-SMP and the other candidate architectures explored during development |
| `src/restoration/benchmarks.py` | Benchmark baselines: DQN, PPO, MADQN, Mean-Field Q, QMIX, Graph-QMIX, Masked IQL, random, greedy planner |
| `scripts/prepare_feeders.py` | Builds the IEEE-123 / IEEE-8500 feeder specifications and the train/val/test scenario split |
| `scripts/train_source_reproduction.py` | Trains/evaluates the reconstructed baseline |
| `scripts/train_candidate.py` | Trains/evaluates PD-SMP and the other candidates |
| `scripts/train_benchmark.py` | Trains/evaluates every benchmark baseline |
| `scripts/robustness_eval.py` | Fault-telemetry and load-scale perturbation evaluation |
| `scripts/statistics_report.py` | Paired seed-level significance testing (Holm-corrected) |
| `scripts/aggregate_final_results.py`, `scripts/aggregate_reproduction.py` | Result aggregation |
| `scripts/make_figures.py` | Figure generation from result CSVs |
| `data/processed/` | Reconstructed IEEE-123 / IEEE-8500 feeder topologies (JSON) consumed by the environment |

## Reproducing the pipeline

```powershell
# 1. Feeder specs + scenario split
python .\scripts\prepare_feeders.py

# 2. Reconstructed baseline
foreach ($system in "ieee123","ieee8500") {
  foreach ($seed in 101..110) {
    python .\scripts\train_source_reproduction.py --system $system --seed $seed --episodes 500
  }
}
python .\scripts\aggregate_reproduction.py

# 3. PD-SMP (final configuration, frozen in FINAL_MODEL_CONFIG.yaml)
foreach ($system in "ieee123","ieee8500") {
  foreach ($seed in 101..110) {
    python .\scripts\train_candidate.py --system $system --candidate c4b --seed $seed --episodes 400 --pretrain-iterations 500 --output FINAL_MODEL_data
  }
}

# 4. Benchmark suite
foreach ($system in "ieee123","ieee8500") {
  foreach ($seed in 101..110) {
    foreach ($mode in "dqn_centralized","ppo_centralized","madqn","meanfield_q","qmix","masked_iql","graph_qmix","random","greedy_planner_only") {
      python .\scripts\train_benchmark.py --system $system --mode $mode --seed $seed --episodes 400 --output BENCHMARKS_data
    }
  }
}

# 5. Robustness evaluation
foreach ($system in "ieee123","ieee8500") {
  foreach ($seed in 101..110) {
    python .\scripts\robustness_eval.py --system $system --seed $seed --output ROBUSTNESS_data
  }
}

# 6. Statistics and figures
python .\scripts\statistics_report.py
python .\scripts\make_figures.py
```

`FINAL_MODEL_CONFIG.yaml` holds the frozen PD-SMP architecture and
hyperparameters used for every reported result.

## Method summary

PD-SMP replaces a discriminator-regularized soft actor-critic backbone with:

1. **Sequential joint-feasibility masking** — agents act in a fixed order
   within each step; each agent's action mask is computed against the true
   partial joint status after all previously-acting agents in that step,
   guaranteeing the assembled joint action is radial and individually legal
   by construction.
2. **Planner-distilled policy** — a compact, context-conditioned per-agent
   policy trained by behavior cloning against a closed-form beam-search
   planner defined over the same incremental mask, then briefly fine-tuned
   with an on-policy, single-critic advantage actor-critic objective.

## License

MIT — see `LICENSE`.
