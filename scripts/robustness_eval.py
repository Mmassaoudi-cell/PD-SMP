"""Robustness experiments (Sec. 20), scoped to what is domain-relevant and
implementable from the existing environment: (a) fault-telemetry observation
noise, fed to the policy network but never to the hard mask (the mask reads
`env.status`/`env.scenario.faulted_switches` directly, mirroring how a real
feeder's switch/fault status is normally obtained from protection relays,
which are far more reliable than derived fault-classification features); and
(b) load-forecast/measurement uncertainty via a scaled perturbation to
`scenario.load_scale`. Both are applied only at evaluation time; the model is
never retrained or re-selected against them (Sec. 21: target-side data stays
untouched during model development).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from restoration import RestorationEnv, Scenario
from restoration.candidates import sequential_hard_mask
from train_candidate import build_and_train, load_scenarios


class NoisyFaultObservationEnv:
    """Proxy over RestorationEnv: local_observation()/observation() fault
    channel is bit-flipped with probability ``flip_prob`` (independent per
    switch, resampled each call). Everything else (status, masks, step
    dynamics, cycle checks) is untouched -- the flip never reaches the
    feasibility mask, only the policy's input features."""

    def __init__(self, env: RestorationEnv, flip_prob: float, rng: np.random.Generator):
        self._env = env
        self.flip_prob = flip_prob
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _flip(self, fault: np.ndarray) -> np.ndarray:
        if self.flip_prob <= 0:
            return fault
        flips = self._rng.random(fault.shape) < self.flip_prob
        return np.logical_xor(fault.astype(bool), flips).astype(np.float32)

    def observation(self, source_view: bool) -> np.ndarray:
        obs = self._env.observation(source_view)
        if source_view:
            return obs
        n = self._env.n_switches
        obs = obs.copy()
        obs[n : 2 * n] = self._flip(obs[n : 2 * n])
        return obs

    def local_observation(self, agent: int, source_view: bool) -> np.ndarray:
        obs = self._env.local_observation(agent, source_view)
        if source_view:
            return obs
        idx = self._env.local_switch_indices(agent)
        n = len(idx)
        obs = obs.copy()
        obs[n : 2 * n] = self._flip(obs[n : 2 * n])
        return obs


def rollout_eval(model, env, scenario: Scenario, rng: np.random.Generator) -> dict:
    env.reset(scenario)
    if hasattr(model, "reset_episode"):
        model.reset_episode()
    cumulative_reward, violations = 0.0, 0
    for _ in range(env.horizon):
        actions, _, _ = model.act(env, rng, greedy=True)
        _, reward, terminated, truncated, info = env.step(actions)
        cumulative_reward += reward
        violations += int(not info["feasible"])
        if terminated or truncated:
            break
    return {
        "cumulative_reward": cumulative_reward,
        "der_utilization": env.last_info["der_utilization"],
        "total_load_fraction": env.last_info["total_load_fraction"],
        "violation_rate": violations / max(env.t, 1),
        "steps": env.t,
    }


def perturb_load(scenario: Scenario, sigma: float, rng: np.random.Generator) -> Scenario:
    factor = float(np.clip(1.0 + rng.normal(0, sigma), 0.5, 1.5))
    return Scenario(
        scenario_id=scenario.scenario_id + f"_loadnoise{sigma}",
        split=scenario.split,
        faulted_switches=scenario.faulted_switches,
        der_availability=scenario.der_availability,
        load_scale=scenario.load_scale * factor,
        scenario_type=scenario.scenario_type,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["ieee123", "ieee8500"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("ROBUSTNESS_data"))
    args = parser.parse_args()

    model, env, _, _, test_scenarios, _, _ = build_and_train(
        args.system, "c4b", args.seed, episodes=400, pretrain_iterations=500
    )
    model.eval()

    conditions = [
        ("clean", 0.0, 0.0),
        ("fault_noise_p0.1", 0.1, 0.0),
        ("fault_noise_p0.3", 0.3, 0.0),
        ("load_noise_sigma0.1", 0.0, 0.1),
        ("load_noise_sigma0.3", 0.0, 0.3),
    ]
    rows = []
    for name, flip_prob, load_sigma in conditions:
        rng = np.random.default_rng(args.seed + 500_000)
        noisy_env = NoisyFaultObservationEnv(env, flip_prob, rng) if flip_prob > 0 else env
        for scenario in test_scenarios:
            s = perturb_load(scenario, load_sigma, rng) if load_sigma > 0 else scenario
            result = rollout_eval(model, noisy_env, s, rng)
            rows.append({"condition": name, "scenario_id": scenario.scenario_id, "seed": args.seed, **result})

    args.output.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.system}_c4b_seed{args.seed}"
    with (args.output / f"{prefix}_robustness.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"system": args.system, "seed": args.seed, "n_rows": len(rows)}))


if __name__ == "__main__":
    main()
