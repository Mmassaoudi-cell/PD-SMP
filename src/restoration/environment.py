from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Switch:
    name: str
    u: str
    v: str
    agent: int
    normally_closed: bool = False


@dataclass(frozen=True)
class FeederSpec:
    name: str
    nodes: tuple[str, ...]
    fixed_edges: tuple[tuple[str, str], ...]
    switches: tuple[Switch, ...]
    load_kw: dict[str, float]
    priority: dict[str, float]
    der_kw: dict[str, float]
    n_agents: int
    source_commit: str
    construction_notes: tuple[str, ...]

    @property
    def total_load_kw(self) -> float:
        return float(sum(self.load_kw.values()))

    @property
    def total_der_kw(self) -> float:
        return float(sum(self.der_kw.values()))

    def save(self, path: str | Path) -> None:
        payload = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeederSpec":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["nodes"] = tuple(raw["nodes"])
        raw["fixed_edges"] = tuple(tuple(x) for x in raw["fixed_edges"])
        raw["switches"] = tuple(Switch(**x) for x in raw["switches"])
        raw["construction_notes"] = tuple(raw["construction_notes"])
        return cls(**raw)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    faulted_switches: tuple[int, ...]
    der_availability: float = 1.0
    load_scale: float = 1.0
    scenario_type: str = "single_fault"


class RestorationEnv:
    """Fast, deterministic topology/capacity restoration environment.

    The environment is a transparent reconstruction, not the undisclosed author
    simulator. Fixed feeder sections remain connected. Controllable switches join
    sections. A state is feasible when the closed-switch graph is radial and every
    energized island's load fits within the available DER capacity in that island.
    """

    def __init__(self, spec: FeederSpec, horizon: int = 16, penalty_scale: float = 10.0):
        self.spec = spec
        self.horizon = int(horizon)
        self.penalty_scale = float(penalty_scale)
        self._base = nx.Graph()
        self._base.add_nodes_from(spec.nodes)
        self._base.add_edges_from(spec.fixed_edges)
        self._zone_of: dict[str, int] = {}
        components = list(nx.connected_components(self._base))
        for zone, component in enumerate(components):
            for node in component:
                self._zone_of[node] = zone
        self._zone_load = np.array(
            [sum(spec.load_kw.get(n, 0.0) for n in component) for component in components], dtype=np.float64
        )
        self._zone_weighted_load = np.array(
            [
                sum(spec.load_kw.get(n, 0.0) * spec.priority.get(n, 1.0) for n in component)
                for component in components
            ],
            dtype=np.float64,
        )
        self._zone_der = np.array(
            [sum(spec.der_kw.get(n, 0.0) for n in component) for component in components], dtype=np.float64
        )
        self._switch_zones = np.array(
            [(self._zone_of[s.u], self._zone_of[s.v]) for s in spec.switches], dtype=np.int64
        )
        self._switch_by_agent = [
            np.array([i for i, s in enumerate(spec.switches) if s.agent == k], dtype=np.int64)
            for k in range(spec.n_agents)
        ]
        self.scenario: Scenario | None = None
        self.status = np.zeros(len(spec.switches), dtype=np.int8)
        self.t = 0
        self.restored_kw = 0.0
        self.weighted_restored_kw = 0.0
        self.last_info: dict[str, float | bool | str] = {}

    @property
    def n_switches(self) -> int:
        return len(self.spec.switches)

    @property
    def local_action_sizes(self) -> tuple[int, ...]:
        return tuple(2 * len(x) + 1 for x in self._switch_by_agent)

    def local_switch_indices(self, agent: int) -> np.ndarray:
        return self._switch_by_agent[agent].copy()

    def reset(self, scenario: Scenario) -> tuple[np.ndarray, dict]:
        self.scenario = scenario
        self.status[:] = np.array([int(s.normally_closed) for s in self.spec.switches], dtype=np.int8)
        if scenario.faulted_switches:
            self.status[list(scenario.faulted_switches)] = 0
        self.t = 0
        feasible, metrics = self._evaluate(self.status)
        if not feasible:
            raise RuntimeError(f"Initial state is infeasible for {scenario.scenario_id}: {metrics}")
        self.restored_kw = metrics["restored_kw"]
        self.weighted_restored_kw = metrics["weighted_restored_kw"]
        self.last_info = metrics
        return self.observation(source_view=False), metrics.copy()

    def observation(self, source_view: bool) -> np.ndarray:
        if self.scenario is None:
            raise RuntimeError("reset() must be called first")
        if source_view:
            return self.status.astype(np.float32).copy()
        faults = np.zeros(self.n_switches, dtype=np.float32)
        faults[list(self.scenario.faulted_switches)] = 1.0
        context = np.array(
            [self.scenario.der_availability, self.scenario.load_scale, self.t / self.horizon],
            dtype=np.float32,
        )
        return np.concatenate([self.status.astype(np.float32), faults, context])

    def local_observation(self, agent: int, source_view: bool) -> np.ndarray:
        idx = self._switch_by_agent[agent]
        if source_view:
            return self.status[idx].astype(np.float32)
        faults = np.zeros(len(idx), dtype=np.float32)
        fault_set = set(self.scenario.faulted_switches if self.scenario else ())
        for j, switch_index in enumerate(idx):
            faults[j] = float(int(switch_index) in fault_set)
        context = np.array(
            [self.scenario.der_availability, self.scenario.load_scale, self.t / self.horizon],
            dtype=np.float32,
        )
        return np.concatenate([self.status[idx].astype(np.float32), faults, context])

    def decode_local_action(self, agent: int, action: int) -> tuple[str, int | None]:
        idx = self._switch_by_agent[agent]
        n = len(idx)
        if action < 0 or action > 2 * n:
            raise ValueError(f"Invalid action {action} for agent {agent}")
        if action < n:
            return "on", int(idx[action])
        if action < 2 * n:
            return "off", int(idx[action - n])
        return "noop", None

    def valid_local_action_mask(self, agent: int, status: np.ndarray | None = None, claimed: set[int] | None = None) -> np.ndarray:
        """Hard per-agent mask (W1/W2/Candidate-1,2 fix).

        ``status`` lets callers evaluate the mask against a partially-decided
        joint status (sequential generation) instead of ``self.status``.
        ``claimed`` additionally excludes switches already targeted by an
        earlier agent in the same step, which is what prevents the
        duplicate-target infeasibility class identified in W3.
        """
        base_status = self.status if status is None else status
        idx = self._switch_by_agent[agent]
        n = len(idx)
        mask = np.ones(2 * n + 1, dtype=bool)
        faulted = set(self.scenario.faulted_switches if self.scenario else ())
        claimed = claimed or set()
        for j, switch_index in enumerate(idx):
            si = int(switch_index)
            mask[j] = base_status[si] == 0 and si not in faulted and si not in claimed
            mask[n + j] = base_status[si] == 1 and si not in claimed
        return mask

    def would_create_cycle(self, status: np.ndarray, switch_index: int) -> bool:
        """True if closing ``switch_index`` on top of ``status`` breaks radiality."""
        proposed = status.copy()
        proposed[switch_index] = 1
        feasible, _ = self._evaluate(proposed)
        return not feasible

    def evaluate_status(self, status: np.ndarray) -> tuple[bool, dict]:
        """Public wrapper around the internal radiality/capacity oracle."""
        return self._evaluate(status)

    def step(self, joint_actions: Sequence[int]) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.scenario is None:
            raise RuntimeError("reset() must be called first")
        if len(joint_actions) != self.spec.n_agents:
            raise ValueError("One local action is required per agent")
        proposed = self.status.copy()
        duplicate_targets: set[int] = set()
        seen: set[int] = set()
        faulted = set(self.scenario.faulted_switches)
        invalid_semantics = False
        for agent, action in enumerate(joint_actions):
            op, switch_index = self.decode_local_action(agent, int(action))
            if switch_index is None:
                continue
            if switch_index in seen:
                duplicate_targets.add(switch_index)
            seen.add(switch_index)
            if op == "on":
                if switch_index in faulted or proposed[switch_index] == 1:
                    invalid_semantics = True
                proposed[switch_index] = 1
            elif op == "off":
                if proposed[switch_index] == 0:
                    invalid_semantics = True
                proposed[switch_index] = 0
        proposed_feasible, proposed_metrics = self._evaluate(proposed)
        cycle_count = int(proposed_metrics["cycle_count"])
        feasible = proposed_feasible and not invalid_semantics and not duplicate_targets
        previous = self.weighted_restored_kw
        self.t += 1
        if feasible:
            self.status = proposed
            self.restored_kw = proposed_metrics["restored_kw"]
            self.weighted_restored_kw = proposed_metrics["weighted_restored_kw"]
            reward = (self.weighted_restored_kw - previous) / max(self.spec.total_der_kw, 1.0)
            reported_metrics = dict(proposed_metrics)
        else:
            overload = max(float(proposed_metrics.get("max_overload_kw", 0.0)), 1.0)
            violation_ratio = max(overload / max(self.spec.total_der_kw, 1.0), 0.1)
            reward = -self.penalty_scale * violation_ratio**2
            # Reporting metrics: the joint action was rejected, so `self.status`
            # (and therefore restored_kw/weighted_restored_kw, updated only in
            # the `if feasible` branch above) is unchanged. der_utilization and
            # total_load_fraction must be recomputed from the *committed*
            # status, not from the rejected `proposed` state, or they would
            # silently report the restoration level of a network state that
            # was never actually realized (an inconsistency that previously
            # affected every terminated episode's headline metrics).
            _, reported_metrics = self._evaluate(self.status)
        terminated = not feasible
        truncated = self.t >= self.horizon
        reported_metrics = dict(reported_metrics)
        reported_metrics.update(
            {
                "feasible": bool(feasible),
                "invalid_semantics": bool(invalid_semantics),
                "duplicate_target": bool(duplicate_targets),
                "cycle_violation": bool(cycle_count > 0),
                "reward": float(reward),
                "step": self.t,
            }
        )
        self.last_info = reported_metrics
        return self.observation(source_view=False), float(reward), terminated, truncated, reported_metrics

    def simulate(self, joint_actions: Sequence[int]) -> dict:
        status = self.status.copy()
        faulted = set(self.scenario.faulted_switches if self.scenario else ())
        invalid = False
        for agent, action in enumerate(joint_actions):
            op, switch_index = self.decode_local_action(agent, int(action))
            if switch_index is None:
                continue
            if op == "on":
                invalid |= switch_index in faulted or status[switch_index] == 1
                status[switch_index] = 1
            else:
                invalid |= status[switch_index] == 0
                status[switch_index] = 0
        feasible, info = self._evaluate(status)
        info = dict(info)
        info["feasible"] = bool(feasible and not invalid)
        return info

    def _evaluate(self, status: np.ndarray) -> tuple[bool, dict]:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self._zone_load)))
        for closed, zones in zip(status, self._switch_zones):
            if closed:
                graph.add_edge(int(zones[0]), int(zones[1]))
        cycle_count = len(nx.cycle_basis(graph))
        load_scale = self.scenario.load_scale if self.scenario else 1.0
        der_scale = self.scenario.der_availability if self.scenario else 1.0
        restored = 0.0
        weighted = 0.0
        max_overload = 0.0
        energized_components = 0
        for component in nx.connected_components(graph):
            indices = np.fromiter(component, dtype=np.int64)
            load = float(self._zone_load[indices].sum()) * load_scale
            capacity = float(self._zone_der[indices].sum()) * der_scale
            if capacity > 0:
                energized_components += 1
                served = min(load, capacity)
                restored += served
                weighted_total = float(self._zone_weighted_load[indices].sum()) * load_scale
                weighted += served * (weighted_total / max(load, 1e-12))
                max_overload = max(max_overload, load - capacity)
        feasible = cycle_count == 0
        return feasible, {
            "restored_kw": float(restored),
            "weighted_restored_kw": float(weighted),
            "der_utilization": float(restored / max(self.spec.total_der_kw * der_scale, 1e-12)),
            "total_load_fraction": float(restored / max(self.spec.total_load_kw * load_scale, 1e-12)),
            "cycle_count": int(cycle_count),
            "max_overload_kw": float(max_overload),
            "energized_components": int(energized_components),
        }


def stable_rank(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def deterministic_split(ids: Iterable[str], train: float = 0.6, validation: float = 0.2) -> dict[str, str]:
    ordered = sorted(ids, key=stable_rank)
    n = len(ordered)
    n_train = int(round(n * train))
    n_val = int(round(n * validation))
    out: dict[str, str] = {}
    for i, item in enumerate(ordered):
        out[item] = "train" if i < n_train else "validation" if i < n_train + n_val else "test"
    return out
