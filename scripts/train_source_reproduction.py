from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from restoration import FeederSpec, RestorationEnv, Scenario
from restoration.source_model import CGenMARL, ReplayBuffer, SourceConfig, seed_everything


def load_scenarios(path: Path, system: str, split: str) -> list[Scenario]:
    scenarios = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["system"] != system or row["split"] != split:
                continue
            faults = tuple(int(x) for x in row["faulted_switches"].split("|") if x != "")
            scenarios.append(
                Scenario(
                    scenario_id=row["scenario_id"],
                    split=row["split"],
                    faulted_switches=faults,
                    der_availability=float(row["der_availability"]),
                    load_scale=float(row["load_scale"]),
                    scenario_type=row["scenario_type"],
                )
            )
    return scenarios


def evaluate(model: CGenMARL, env: RestorationEnv, scenarios: list[Scenario], seed: int) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed + 100_000)
    for scenario in scenarios:
        env.reset(scenario)
        cumulative_reward = 0.0
        violations = 0
        cycle_violations = 0
        invalid_semantics_count = 0
        duplicate_target_count = 0
        start = time.perf_counter()
        for _ in range(env.horizon):
            state = env.observation(source_view=True)
            actions = model.select_actions(state, epsilon=0.0, rng=rng, deterministic=True)
            _, reward, terminated, truncated, info = env.step(actions)
            cumulative_reward += reward
            violations += int(not info["feasible"])
            cycle_violations += int(info.get("cycle_violation", False))
            invalid_semantics_count += int(info.get("invalid_semantics", False))
            duplicate_target_count += int(info.get("duplicate_target", False))
            if terminated or truncated:
                break
        elapsed_ms = (time.perf_counter() - start) * 1000 / max(env.t, 1)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "split": scenario.split,
                "scenario_type": scenario.scenario_type,
                "seed": seed,
                "cumulative_reward": cumulative_reward,
                "restored_kw": env.restored_kw,
                "weighted_restored_kw": env.weighted_restored_kw,
                "der_utilization": env.last_info["der_utilization"],
                "total_load_fraction": env.last_info["total_load_fraction"],
                "violation_rate": violations / max(env.t, 1),
                "violations": violations,
                "cycle_violations": cycle_violations,
                "invalid_semantics_violations": invalid_semantics_count,
                "duplicate_target_violations": duplicate_target_count,
                "steps": env.t,
                "latency_ms": elapsed_ms,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["ieee123", "ieee8500"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("SOURCE_METHOD_REPRODUCTION/results"))
    args = parser.parse_args()
    rng = seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = FeederSpec.load(Path("data/processed") / f"{args.system}_feeder.json")
    env = RestorationEnv(spec, horizon=16, penalty_scale=10.0)
    train_scenarios = load_scenarios(Path("DATA_SPLIT_MANIFEST.csv"), args.system, "train")
    validation_scenarios = load_scenarios(Path("DATA_SPLIT_MANIFEST.csv"), args.system, "validation")
    test_scenarios = load_scenarios(Path("DATA_SPLIT_MANIFEST.csv"), args.system, "test")
    if args.system == "ieee123":
        config = SourceConfig(hidden_dim=128, actor_lr=4.5e-4, critic_lr=5.5e-4, tau=0.01)
    else:
        config = SourceConfig(hidden_dim=256, actor_lr=5e-4, critic_lr=6.5e-4, tau=0.02, attention_dim=64)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    model = CGenMARL(
        n_switches=env.n_switches,
        local_switch_indices=[env.local_switch_indices(k) for k in range(spec.n_agents)],
        local_action_sizes=env.local_action_sizes,
        config=config,
        device=device,
    )
    replay = ReplayBuffer(config.buffer_size, env.n_switches, spec.n_agents)
    episode_rows = []
    best_validation = -np.inf
    best_state = None
    training_start = time.perf_counter()
    for episode in range(args.episodes):
        scenario = train_scenarios[int(rng.integers(0, len(train_scenarios)))]
        env.reset(scenario)
        cumulative_reward = 0.0
        violations = 0
        last_losses = None
        epsilon = max(0.05, 1.0 - episode / max(int(args.episodes * 0.8), 1))
        for _ in range(env.horizon):
            state = env.observation(source_view=True)
            actions = model.select_actions(state, epsilon=epsilon, rng=rng, deterministic=False)
            _, reward, terminated, truncated, info = env.step(actions)
            next_state = env.observation(source_view=True)
            replay.add(state, actions, reward, next_state, terminated or truncated, info["feasible"])
            last_losses = model.update(replay, rng)
            cumulative_reward += reward
            violations += int(not info["feasible"])
            if terminated or truncated:
                break
        row = {
            "episode": episode,
            "scenario_id": scenario.scenario_id,
            "epsilon": epsilon,
            "cumulative_reward": cumulative_reward,
            "restored_kw": env.restored_kw,
            "der_utilization": env.last_info["der_utilization"],
            "violations": violations,
        }
        if last_losses:
            row.update(last_losses)
        episode_rows.append(row)
        if (episode + 1) % max(10, args.episodes // 20) == 0:
            validation_rows = evaluate(model, env, validation_scenarios, args.seed)
            score = float(np.mean([x["der_utilization"] - x["violation_rate"] for x in validation_rows]))
            if score > best_validation:
                best_validation = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(json.dumps({"episode": episode + 1, "validation_score": score, "best": best_validation}), flush=True)
    training_seconds = time.perf_counter() - training_start
    if best_state is not None:
        model.load_state_dict(best_state)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"{args.system}_seed{args.seed}.pt"
    torch.save({"state_dict": model.state_dict(), "config": config.__dict__, "best_validation": best_validation}, checkpoint)
    with (args.output / f"{args.system}_seed{args.seed}_training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in episode_rows for k in row}))
        writer.writeheader()
        writer.writerows(episode_rows)
    final_rows = evaluate(model, env, validation_scenarios + test_scenarios, args.seed)
    for row in final_rows:
        row.update({"method": "SOURCE_METHOD_REPRODUCTION", "system": args.system, "training_seconds": training_seconds})
    with (args.output / f"{args.system}_seed{args.seed}_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)
    print(json.dumps({"checkpoint": str(checkpoint), "training_seconds": training_seconds, "best_validation": best_validation}))


if __name__ == "__main__":
    main()
