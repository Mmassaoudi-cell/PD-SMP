from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

torch.set_num_threads(2)  # tiny networks; avoid oversubscribing cores when many runs execute concurrently
import torch.nn.functional as F

from restoration import FeederSpec, RestorationEnv, Scenario
from restoration.candidates import CANDIDATES, Candidate4Student, greedy_plan, seed_everything

GAMMA = 0.95


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


def build_env(system: str) -> RestorationEnv:
    spec = FeederSpec.load(Path("data/processed") / f"{system}_feeder.json")
    return RestorationEnv(spec, horizon=16, penalty_scale=10.0)


def rollout_episode(model, env: RestorationEnv, scenario: Scenario, rng: np.random.Generator, greedy: bool):
    env.reset(scenario)
    if hasattr(model, "reset_episode"):
        model.reset_episode()
    step_logps, rewards, values = [], [], []
    violations = 0
    cycle_violations = 0
    invalid_semantics_count = 0
    duplicate_target_count = 0
    for _ in range(env.horizon):
        actions, logps, value = model.act(env, rng, greedy=greedy)
        _, reward, terminated, truncated, info = env.step(actions)
        step_logps.append(logps)
        rewards.append(float(reward))
        values.append(value)
        violations += int(not info["feasible"])
        cycle_violations += int(info.get("cycle_violation", False))
        invalid_semantics_count += int(info.get("invalid_semantics", False))
        duplicate_target_count += int(info.get("duplicate_target", False))
        if terminated or truncated:
            break
    bootstrap = 0.0
    if not terminated and truncated:
        with torch.no_grad():
            bootstrap = float(
                model.value(torch.as_tensor(env.observation(source_view=False)).float().unsqueeze(0)).item()
            )
    return {
        "step_logps": step_logps,
        "rewards": rewards,
        "values": values,
        "bootstrap": bootstrap,
        "violations": violations,
        "cycle_violations": cycle_violations,
        "invalid_semantics_violations": invalid_semantics_count,
        "duplicate_target_violations": duplicate_target_count,
        "steps": env.t,
        "restored_kw": env.restored_kw,
        "weighted_restored_kw": env.weighted_restored_kw,
        "der_utilization": env.last_info["der_utilization"],
        "total_load_fraction": env.last_info["total_load_fraction"],
    }


def actor_critic_update(model, optimizer, episodes: list[dict]) -> dict[str, float]:
    actor_terms = []
    critic_terms = []
    for episode in episodes:
        rewards = episode["rewards"]
        returns = [0.0] * len(rewards)
        running = episode["bootstrap"]
        for t in reversed(range(len(rewards))):
            running = rewards[t] + GAMMA * running
            returns[t] = running
        for t, (logps, value, ret) in enumerate(zip(episode["step_logps"], episode["values"], returns)):
            ret_t = torch.tensor(ret, dtype=torch.float32)
            advantage = (ret_t - value).detach()
            step_logp = torch.stack(logps).sum()
            actor_terms.append(-step_logp * advantage)
            critic_terms.append(F.mse_loss(value, ret_t))
    actor_loss = torch.stack(actor_terms).mean()
    critic_loss = torch.stack(critic_terms).mean()
    loss = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach())}


def evaluate(model, env: RestorationEnv, scenarios: list[Scenario], seed: int) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed + 100_000)
    for scenario in scenarios:
        start = time.perf_counter()
        result = rollout_episode(model, env, scenario, rng, greedy=True)
        elapsed_ms = (time.perf_counter() - start) * 1000 / max(result["steps"], 1)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "split": scenario.split,
                "scenario_type": scenario.scenario_type,
                "seed": seed,
                "cumulative_reward": sum(result["rewards"]),
                "restored_kw": result["restored_kw"],
                "weighted_restored_kw": result["weighted_restored_kw"],
                "der_utilization": result["der_utilization"],
                "total_load_fraction": result["total_load_fraction"],
                "violation_rate": result["violations"] / max(result["steps"], 1),
                "violations": result["violations"],
                "cycle_violations": result["cycle_violations"],
                "invalid_semantics_violations": result["invalid_semantics_violations"],
                "duplicate_target_violations": result["duplicate_target_violations"],
                "steps": result["steps"],
                "latency_ms": elapsed_ms,
            }
        )
    return rows


def pretrain_candidate4(model: Candidate4Student, env: RestorationEnv, scenarios: list[Scenario], rng: np.random.Generator, iterations: int, optimizer) -> None:
    for i in range(iterations):
        scenario = scenarios[int(rng.integers(0, len(scenarios)))]
        env.reset(scenario)
        for _ in range(env.horizon):
            plan = greedy_plan(env, beam_width=4)[0]
            loss = model.imitation_loss(env, plan)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _, _, terminated, truncated, _ = env.step(plan)
            if terminated or truncated:
                break


def build_and_train(
    system: str,
    candidate: str,
    seed: int,
    episodes: int,
    batch_episodes: int = 8,
    lr: float = 3e-3,
    pretrain_iterations: int = 0,
    skip_finetune: bool = False,
):
    """Core training routine, factored out so robustness/ablation scripts can
    reuse a freshly-trained model object without re-implementing training."""
    rng = seed_everything(seed)
    env = build_env(system)
    model_cls = CANDIDATES[candidate]
    model = model_cls(env)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_scenarios = load_scenarios(Path("DATA_SPLIT_MANIFEST.csv"), system, "train")
    validation_scenarios = load_scenarios(Path("DATA_SPLIT_MANIFEST.csv"), system, "validation")
    test_scenarios = load_scenarios(Path("DATA_SPLIT_MANIFEST.csv"), system, "test")

    if candidate in ("c4", "c4b") and pretrain_iterations > 0:
        pretrain_candidate4(model, env, train_scenarios, rng, pretrain_iterations, optimizer)

    training_start = time.perf_counter()
    training_rows = []
    n_updates = max(episodes // batch_episodes, 1) if not skip_finetune else 0
    for update in range(n_updates):
        batch = []
        for _ in range(batch_episodes):
            scenario = train_scenarios[int(rng.integers(0, len(train_scenarios)))]
            batch.append(rollout_episode(model, env, scenario, rng, greedy=False))
        losses = actor_critic_update(model, optimizer, batch)
        mean_reward = float(np.mean([sum(e["rewards"]) for e in batch]))
        mean_violation = float(np.mean([e["violations"] / max(e["steps"], 1) for e in batch]))
        training_rows.append({"update": update, "mean_reward": mean_reward, "mean_violation_rate": mean_violation, **losses})
    training_seconds = time.perf_counter() - training_start
    return model, env, train_scenarios, validation_scenarios, test_scenarios, training_rows, training_seconds


def train(
    system: str,
    candidate: str,
    seed: int,
    episodes: int,
    batch_episodes: int = 8,
    lr: float = 3e-3,
    pretrain_iterations: int = 0,
    skip_finetune: bool = False,
    output: Path = Path("MODEL_SELECTION_REPORT_data"),
) -> dict:
    model, env, train_scenarios, validation_scenarios, test_scenarios, training_rows, training_seconds = build_and_train(
        system, candidate, seed, episodes, batch_episodes, lr, pretrain_iterations, skip_finetune
    )

    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{system}_{candidate}_seed{seed}"
    if candidate in ("c4b", "c4"):
        torch.save({"state_dict": model.state_dict(), "system": system, "candidate": candidate, "seed": seed}, output / f"{prefix}.pt")
    with (output / f"{prefix}_training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in training_rows for k in r}))
        writer.writeheader()
        writer.writerows(training_rows)

    eval_rows = evaluate(model, env, validation_scenarios + test_scenarios, seed)
    n_params = sum(p.numel() for p in model.parameters())
    for row in eval_rows:
        row.update(
            {
                "method": type(model).name,
                "candidate": candidate,
                "system": system,
                "training_seconds": training_seconds,
                "n_params": n_params,
            }
        )
    with (output / f"{prefix}_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=eval_rows[0].keys())
        writer.writeheader()
        writer.writerows(eval_rows)

    summary = {
        "system": system,
        "candidate": candidate,
        "seed": seed,
        "n_params": n_params,
        "training_seconds": training_seconds,
        "final_train_mean_reward": training_rows[-1]["mean_reward"] if training_rows else float("nan"),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["ieee123", "ieee8500"], required=True)
    parser.add_argument("--candidate", choices=list(CANDIDATES.keys()), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--pretrain-iterations", type=int, default=0)
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("MODEL_SELECTION_REPORT_data"))
    args = parser.parse_args()
    summary = train(
        args.system,
        args.candidate,
        args.seed,
        args.episodes,
        pretrain_iterations=args.pretrain_iterations,
        skip_finetune=args.skip_finetune,
        output=args.output,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
