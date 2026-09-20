from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

torch.set_num_threads(2)  # tiny networks; avoid oversubscribing cores when many runs execute concurrently

from restoration import FeederSpec, RestorationEnv, Scenario
from restoration.benchmarks import (
    BenchmarkConfig,
    GreedyPlannerOnly,
    PPOAgentSet,
    QLearningAgentSet,
    RandomPolicy,
    ReplayBuffer,
    seed_everything,
)
from restoration.candidates import global_input_dim

VALUE_MODES = {"dqn_centralized", "madqn", "meanfield_q", "qmix", "masked_iql", "graph_qmix"}
NO_TRAIN_MODES = {"random", "greedy_planner_only"}


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


def evaluate(model, env: RestorationEnv, scenarios: list[Scenario], seed: int, mode: str) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed + 100_000)
    for scenario in scenarios:
        env.reset(scenario)
        if hasattr(model, "reset_episode"):
            model.reset_episode()
        cumulative_reward = 0.0
        violations = 0
        cycle_violations = 0
        invalid_semantics_count = 0
        duplicate_target_count = 0
        start = time.perf_counter()
        for _ in range(env.horizon):
            if mode == "ppo_centralized":
                actions, _, _ = model.act(env, rng, greedy=True)
            else:
                actions = model.act(env, rng, greedy=True) if mode in NO_TRAIN_MODES else model.act(env, epsilon=0.0, rng=rng, greedy=True)
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


def train_value_based(env, mode, train_scenarios, episodes, seed, rng, config: BenchmarkConfig, update_every: int = 4):
    """``update_every`` bounds gradient-update cost: masked/planner-guided
    modes reliably run the full episode horizon (Sec. above), so they
    otherwise accumulate far more updates per episode than the unmasked
    baselines, which usually terminate after 1-2 steps. Updating once every
    ``update_every`` environment steps keeps wall-clock comparable across
    modes without changing the algorithm."""
    device = torch.device("cpu")
    model = QLearningAgentSet(env, mode, config, device)
    buffer = ReplayBuffer(config.buffer_size, global_input_dim(env), env.spec.n_agents)
    global_step = 0
    for episode in range(episodes):
        scenario = train_scenarios[int(rng.integers(0, len(train_scenarios)))]
        env.reset(scenario)
        model.reset_episode()
        epsilon = max(config.epsilon_end, config.epsilon_start - episode / max(episodes * 0.8, 1))
        for _ in range(env.horizon):
            g = env.observation(source_view=False)
            actions = model.act(env, epsilon, rng, greedy=False)
            _, reward, terminated, truncated, info = env.step(actions)
            ng = env.observation(source_view=False)
            buffer.add(g, actions, reward, ng, float(terminated or truncated))
            global_step += 1
            if global_step % update_every == 0:
                model.update(buffer, rng)
            if terminated or truncated:
                break
    return model


def train_ppo(env, train_scenarios, episodes, rng, config: BenchmarkConfig, batch_episodes: int = 8):
    model = PPOAgentSet(env, config)
    gamma = config.gamma
    n_updates = max(episodes // batch_episodes, 1)
    for _ in range(n_updates):
        batch_globals, batch_actions, batch_returns, batch_old_logps = [], [], [], []
        for _ in range(batch_episodes):
            scenario = train_scenarios[int(rng.integers(0, len(train_scenarios)))]
            env.reset(scenario)
            globals_, actions_, rewards_, old_logps_ = [], [], [], []
            for _ in range(env.horizon):
                g = env.observation(source_view=False)
                actions, logps, _ = model.act(env, rng, greedy=False)
                _, reward, terminated, truncated, _ = env.step(actions)
                globals_.append(g)
                actions_.append(actions)
                rewards_.append(reward)
                old_logps_.append(sum(logps).item())
                if terminated or truncated:
                    break
            returns_, running = [], 0.0
            for r in reversed(rewards_):
                running = r + gamma * running
                returns_.insert(0, running)
            batch_globals.extend(globals_)
            batch_actions.extend(actions_)
            batch_returns.extend(returns_)
            batch_old_logps.extend(old_logps_)
        g_t = torch.as_tensor(np.array(batch_globals)).float()
        a_t = torch.as_tensor(np.array(batch_actions)).long()
        ret_t = torch.as_tensor(batch_returns).float()
        old_logp_t = torch.as_tensor(batch_old_logps).float()
        for _ in range(4):
            new_logp, entropy, value = model.evaluate_actions(g_t, a_t)
            advantage = (ret_t - value).detach()
            ratio = torch.exp(new_logp - old_logp_t)
            surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 1 - model.clip, 1 + model.clip) * advantage)
            actor_loss = -surrogate.mean() - 0.01 * entropy.mean()
            critic_loss = torch.nn.functional.mse_loss(value, ret_t)
            loss = actor_loss + 0.5 * critic_loss
            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            model.optimizer.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["ieee123", "ieee8500"], required=True)
    parser.add_argument(
        "--mode",
        choices=["random", "greedy_planner_only", "dqn_centralized", "ppo_centralized", "madqn", "meanfield_q", "qmix", "masked_iql", "graph_qmix"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("BENCHMARKS_data"))
    args = parser.parse_args()

    rng = seed_everything(args.seed)
    env = build_env(args.system)
    train_scenarios = load_scenarios(Path("DATA_SPLIT_MANIFEST.csv"), args.system, "train")
    validation_scenarios = load_scenarios(Path("DATA_SPLIT_MANIFEST.csv"), args.system, "validation")
    test_scenarios = load_scenarios(Path("DATA_SPLIT_MANIFEST.csv"), args.system, "test")

    config = BenchmarkConfig()
    training_start = time.perf_counter()
    if args.mode == "random":
        model = RandomPolicy(env)
    elif args.mode == "greedy_planner_only":
        model = GreedyPlannerOnly(env)
    elif args.mode == "ppo_centralized":
        model = train_ppo(env, train_scenarios, args.episodes, rng, config)
    else:
        model = train_value_based(env, args.mode, train_scenarios, args.episodes, args.seed, rng, config)
    training_seconds = time.perf_counter() - training_start

    n_params = sum(p.numel() for p in model.parameters()) if isinstance(model, torch.nn.Module) else 0
    eval_rows = evaluate(model, env, validation_scenarios + test_scenarios, args.seed, args.mode)
    for row in eval_rows:
        row.update({"method": args.mode, "system": args.system, "training_seconds": training_seconds, "n_params": n_params})
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.system}_{args.mode}_seed{args.seed}"
    with (args.output / f"{prefix}_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=eval_rows[0].keys())
        writer.writeheader()
        writer.writerows(eval_rows)
    print(json.dumps({"mode": args.mode, "n_params": n_params, "training_seconds": training_seconds}))


if __name__ == "__main__":
    main()
