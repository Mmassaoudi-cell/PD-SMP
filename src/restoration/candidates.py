"""Candidate hybrid models (MODEL_CANDIDATES.md).

All candidates share one on-policy actor-critic trainer (`ActorCriticTrainer`)
instead of the source paper's replay-buffer SAC, which removes the
entropy-temperature/target-network/discriminator machinery identified as
unreliable or costly in `SOURCE_WEAKNESS_ANALYSIS.md` (W2, W3, W7). Candidates
differ only in actor architecture, action generation order, and masking
policy - the axis the weakness analysis says actually matters.

Every candidate is context-conditioned (fixes W1): each agent observes its
local switch status, local fault mask, DER availability, load scale, and
episode progress (`RestorationEnv.local_observation(..., source_view=False)`),
never the source paper's switch-status-only view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .environment import RestorationEnv


def local_input_dims(env: RestorationEnv) -> list[int]:
    return [2 * len(env.local_switch_indices(k)) + 3 for k in range(env.spec.n_agents)]


def global_input_dim(env: RestorationEnv) -> int:
    return 2 * env.n_switches + 3


def sequential_hard_mask(env: RestorationEnv, agent: int, partial_status: np.ndarray, use_cycle_mask: bool = True) -> np.ndarray:
    """Per-agent hard mask evaluated against a partial, within-step joint
    status, additionally forbidding 'on' actions that would create a cycle
    against switches already decided earlier in the same step (the joint
    radiality risk independent simultaneous actors cannot see; W3)."""
    idx = env.local_switch_indices(agent)
    mask = env.valid_local_action_mask(agent, status=partial_status)
    if use_cycle_mask:
        for j, switch_index in enumerate(idx):
            if mask[j] and env.would_create_cycle(partial_status, int(switch_index)):
                mask[j] = False
    return mask


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    if bool((~mask).all()):
        masked_logits = logits  # degenerate: no-op is always index -1 safe fallback handled by caller
    return torch.distributions.Categorical(logits=masked_logits)


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Candidate 1: Context-Conditioned Hard-Masked Actor-Critic (CC-HM-AC)
# ---------------------------------------------------------------------------


class IndependentActor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


class Candidate1(nn.Module):
    """Independent per-agent actors, context-conditioned, hard-masked. No discriminator."""

    name = "c1_context_hardmask"

    def __init__(self, env: RestorationEnv, hidden_dim: int = 64):
        super().__init__()
        self.env = env
        dims = local_input_dims(env)
        sizes = env.local_action_sizes
        self.actors = nn.ModuleList([IndependentActor(d, hidden_dim, a) for d, a in zip(dims, sizes)])
        self.critic = Critic(global_input_dim(env), hidden_dim)

    def act(self, env: RestorationEnv, rng: np.random.Generator, greedy: bool) -> tuple[list[int], list[torch.Tensor], torch.Tensor]:
        actions, logps = [], []
        for k, actor in enumerate(self.actors):
            obs = torch.as_tensor(env.local_observation(k, source_view=False)).float().unsqueeze(0)
            mask = torch.as_tensor(env.valid_local_action_mask(k)).unsqueeze(0)
            logits = actor(obs)
            dist = masked_categorical(logits, mask)
            action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
            logps.append(dist.log_prob(action).squeeze(0))
            actions.append(int(action.item()))
        value = self.critic(torch.as_tensor(env.observation(source_view=False)).float().unsqueeze(0)).squeeze(0)
        return actions, logps, value

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(global_obs)


# ---------------------------------------------------------------------------
# Candidate 2: Sequential Autoregressive Switch Generator (SASG)
# ---------------------------------------------------------------------------


class Candidate2(nn.Module):
    """Agents act sequentially; each agent's mask accounts for switches already
    closed/opened earlier in the same step, preventing joint radiality
    violations that independent simultaneous actors cannot see (W3)."""

    name = "c2_sequential_generator"

    def __init__(self, env: RestorationEnv, hidden_dim: int = 64, use_cycle_mask: bool = True):
        super().__init__()
        self.env = env
        self.hidden_dim = hidden_dim
        self.use_cycle_mask = use_cycle_mask
        dims = local_input_dims(env)
        sizes = env.local_action_sizes
        self.embed = nn.ModuleList([nn.Linear(d, hidden_dim) for d in dims])
        self.context_embed = nn.Linear(3, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, a) for a in sizes])
        self.critic = Critic(global_input_dim(env), hidden_dim)

    def act(self, env: RestorationEnv, rng: np.random.Generator, greedy: bool) -> tuple[list[int], list[torch.Tensor], torch.Tensor]:
        context = torch.as_tensor(
            [env.scenario.der_availability, env.scenario.load_scale, env.t / env.horizon], dtype=torch.float32
        ).unsqueeze(0)
        h = torch.tanh(self.context_embed(context))
        partial_status = env.status.copy()
        actions, logps = [], []
        for k in range(env.spec.n_agents):
            obs = torch.as_tensor(env.local_observation(k, source_view=False)).float().unsqueeze(0)
            x = F.silu(self.embed[k](obs))
            h = self.gru(x, h)
            logits = self.heads[k](h)
            mask = torch.as_tensor(sequential_hard_mask(env, k, partial_status, self.use_cycle_mask)).unsqueeze(0)
            dist = masked_categorical(logits, mask)
            action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
            logps.append(dist.log_prob(action).squeeze(0))
            actions.append(int(action.item()))
            op, switch_index = env.decode_local_action(k, int(action.item()))
            if switch_index is not None:
                partial_status[switch_index] = 1 if op == "on" else 0
        value = self.critic(torch.as_tensor(env.observation(source_view=False)).float().unsqueeze(0)).squeeze(0)
        return actions, logps, value

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(global_obs)


# ---------------------------------------------------------------------------
# Candidate 3: Graph-structural encoder + hard-masked independent actors
# ---------------------------------------------------------------------------


class SwitchGraphEncoder(nn.Module):
    """One-hop mean-aggregation graph layer over the switch-adjacency graph
    (switches sharing a zone are neighbors). Deliberately lightweight (Sec. 8):
    a full multi-layer GNN library is not warranted at <=100 nodes."""

    def __init__(self, n_switches: int, feature_dim: int, hidden_dim: int, adjacency: np.ndarray):
        super().__init__()
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        degree = adjacency.sum(axis=1, keepdims=True).clip(min=1.0)
        self.register_buffer("inv_degree", torch.as_tensor(1.0 / degree, dtype=torch.float32))
        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.proj(node_features))
        neighbor_sum = torch.matmul(self.adjacency, h)
        neighbor_mean = neighbor_sum * self.inv_degree
        return F.silu(self.combine(torch.cat([h, neighbor_mean], dim=-1)))


def switch_adjacency(env: RestorationEnv) -> np.ndarray:
    n = env.n_switches
    adjacency = np.zeros((n, n), dtype=np.float32)
    for k in range(env.spec.n_agents):
        idx = env.local_switch_indices(k)
        for a in idx:
            for b in idx:
                if a != b:
                    adjacency[a, b] = 1.0
    return adjacency


class Candidate3(nn.Module):
    """Graph-encoded features feed independent hard-masked actors (isolates
    explicit structure vs. A2Reg's attention-dropout proxy for structure)."""

    name = "c3_graph_hardmask"

    def __init__(self, env: RestorationEnv, hidden_dim: int = 64):
        super().__init__()
        self.env = env
        adjacency = switch_adjacency(env)
        self.encoder = SwitchGraphEncoder(env.n_switches, 2, hidden_dim, adjacency)
        self.context_embed = nn.Linear(3, hidden_dim)
        sizes = env.local_action_sizes
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim * len(env.local_switch_indices(k)) + hidden_dim, a) for k, a in enumerate(sizes)]
        )
        self.critic = Critic(global_input_dim(env), hidden_dim)

    def _node_features(self, env: RestorationEnv) -> torch.Tensor:
        faults = np.zeros(env.n_switches, dtype=np.float32)
        faults[list(env.scenario.faulted_switches)] = 1.0
        features = np.stack([env.status.astype(np.float32), faults], axis=-1)
        return torch.as_tensor(features).float()

    def act(self, env: RestorationEnv, rng: np.random.Generator, greedy: bool) -> tuple[list[int], list[torch.Tensor], torch.Tensor]:
        embeddings = self.encoder(self._node_features(env))
        context = torch.as_tensor(
            [env.scenario.der_availability, env.scenario.load_scale, env.t / env.horizon], dtype=torch.float32
        )
        context_h = F.silu(self.context_embed(context))
        actions, logps = [], []
        for k, head in enumerate(self.heads):
            idx = env.local_switch_indices(k)
            local_embed = embeddings[idx].reshape(-1)
            logits = head(torch.cat([local_embed, context_h], dim=-1)).unsqueeze(0)
            mask = torch.as_tensor(env.valid_local_action_mask(k)).unsqueeze(0)
            dist = masked_categorical(logits, mask)
            action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
            logps.append(dist.log_prob(action).squeeze(0))
            actions.append(int(action.item()))
        value = self.critic(torch.as_tensor(env.observation(source_view=False)).float().unsqueeze(0)).squeeze(0)
        return actions, logps, value

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(global_obs)


# ---------------------------------------------------------------------------
# Candidate 4: Planner-distilled compact policy
# ---------------------------------------------------------------------------


def greedy_plan(env: RestorationEnv, beam_width: int = 4) -> list[list[int]]:
    """Within-step beam search over per-agent local actions for the CURRENT
    environment step only, maximizing the immediate next-step weighted
    restored load subject to the same hard masks used elsewhere (one-step
    lookahead, not a multi-step plan). Returns a single-element list holding
    the best joint action for this step; does not mutate ``env``. Called
    again, fresh, at every step by the caller (see ``pretrain_candidate4``)."""
    n_agents = env.spec.n_agents
    beams: list[tuple[np.ndarray, list[int], float]] = [(env.status.copy(), [], 0.0)]
    for k in range(n_agents):
        candidates: list[tuple[np.ndarray, list[int], float]] = []
        for status, partial_actions, score in beams:
            mask = env.valid_local_action_mask(k, status=status)
            options = np.flatnonzero(mask)
            for action in options:
                op, switch_index = env.decode_local_action(k, int(action))
                new_status = status.copy()
                delta = 0.0
                if switch_index is not None:
                    if op == "on" and env.would_create_cycle(status, int(switch_index)):
                        continue
                    new_status[switch_index] = 1 if op == "on" else 0
                    feasible, metrics = env.evaluate_status(new_status)
                    if not feasible:
                        continue
                    delta = metrics["weighted_restored_kw"]
                else:
                    _, metrics = env.evaluate_status(new_status)
                    delta = metrics["weighted_restored_kw"]
                candidates.append((new_status, partial_actions + [int(action)], delta))
        if not candidates:
            candidates = [(status, partial_actions + [len(env.local_action_sizes) - 1], score) for status, partial_actions, score in beams]
        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_width]
    best_status, best_actions, _ = max(beams, key=lambda x: x[2])
    return [best_actions]


class Candidate4Student(nn.Module):
    """Small MLP student distilled from the beam-search planner, then
    fine-tuned with the shared actor-critic trainer.

    ``sequential_joint_mask=True`` (Candidate 4b in MODEL_SELECTION_REPORT.md)
    decides agents in a fixed order using the same incremental joint-radiality
    mask as Candidate 2 (`sequential_hard_mask`), instead of each agent's
    independent per-agent-only mask. This was added during validation-only
    screening after the independent-mask version showed a nonzero violation
    rate despite per-agent masking (Stage 2 evidence), isolating whether the
    residual risk was joint (cross-agent) or per-agent."""

    name = "c4_planner_distilled"

    def __init__(self, env: RestorationEnv, hidden_dim: int = 32, sequential_joint_mask: bool = False):
        super().__init__()
        self.env = env
        self.sequential_joint_mask = sequential_joint_mask
        dims = local_input_dims(env)
        sizes = env.local_action_sizes
        self.actors = nn.ModuleList([IndependentActor(d, hidden_dim, a) for d, a in zip(dims, sizes)])
        self.critic = Critic(global_input_dim(env), hidden_dim)

    def act(self, env: RestorationEnv, rng: np.random.Generator, greedy: bool) -> tuple[list[int], list[torch.Tensor], torch.Tensor]:
        actions, logps = [], []
        partial_status = env.status.copy() if self.sequential_joint_mask else None
        for k, actor in enumerate(self.actors):
            obs = torch.as_tensor(env.local_observation(k, source_view=False)).float().unsqueeze(0)
            if self.sequential_joint_mask:
                mask = torch.as_tensor(sequential_hard_mask(env, k, partial_status, use_cycle_mask=True)).unsqueeze(0)
            else:
                mask = torch.as_tensor(env.valid_local_action_mask(k)).unsqueeze(0)
            logits = actor(obs)
            dist = masked_categorical(logits, mask)
            action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
            logps.append(dist.log_prob(action).squeeze(0))
            actions.append(int(action.item()))
            if self.sequential_joint_mask:
                op, switch_index = env.decode_local_action(k, int(action.item()))
                if switch_index is not None:
                    partial_status[switch_index] = 1 if op == "on" else 0
        value = self.critic(torch.as_tensor(env.observation(source_view=False)).float().unsqueeze(0)).squeeze(0)
        return actions, logps, value

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(global_obs)

    def imitation_loss(self, env: RestorationEnv, target_actions: Sequence[int]) -> torch.Tensor:
        loss = torch.zeros(())
        for k, (actor, target) in enumerate(zip(self.actors, target_actions)):
            obs = torch.as_tensor(env.local_observation(k, source_view=False)).float().unsqueeze(0)
            mask = torch.as_tensor(env.valid_local_action_mask(k)).unsqueeze(0)
            logits = actor(obs)
            masked_logits = torch.where(mask, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))
            loss = loss + F.cross_entropy(masked_logits, torch.tensor([target]))
        return loss / len(self.actors)


# ---------------------------------------------------------------------------
# Candidate 5: Deterministic-repair mean-field actor
# ---------------------------------------------------------------------------


def repair_joint_action(env: RestorationEnv, actions: list[int]) -> list[int]:
    """Greedily drops the minimal set of 'on' operations needed to restore
    radiality, in agent order, leaving 'off'/no-op operations untouched
    (opening a switch can never create a cycle)."""
    repaired = list(actions)
    status = env.status.copy()
    on_ops: list[tuple[int, int]] = []
    for k, action in enumerate(repaired):
        op, switch_index = env.decode_local_action(k, action)
        if switch_index is None:
            continue
        if op == "off":
            status[switch_index] = 0
        else:
            on_ops.append((k, switch_index))
    for k, switch_index in on_ops:
        if env.would_create_cycle(status, switch_index):
            n = len(env.local_switch_indices(k))
            repaired[k] = 2 * n  # no-op index for agent k
        else:
            status[switch_index] = 1
    return repaired


class Candidate5(nn.Module):
    """Mean-field-coordinated independent actors (each conditions on the mean
    embedding of the other agents' previous action) with post-hoc deterministic
    repair instead of hard masking during generation."""

    name = "c5_meanfield_repair"

    def __init__(self, env: RestorationEnv, hidden_dim: int = 64):
        super().__init__()
        self.env = env
        dims = local_input_dims(env)
        sizes = env.local_action_sizes
        self.mean_field_dim = 8
        self.action_embed = nn.ModuleList([nn.Linear(a, self.mean_field_dim) for a in sizes])
        self.actors = nn.ModuleList([IndependentActor(d + self.mean_field_dim, hidden_dim, a) for d, a in zip(dims, sizes)])
        self.critic = Critic(global_input_dim(env), hidden_dim)
        self._prev_action_probs: list[torch.Tensor] | None = None

    def _mean_field_context(self, exclude: int) -> torch.Tensor:
        if self._prev_action_probs is None:
            return torch.zeros(self.mean_field_dim)
        embeds = [
            self.action_embed[k](probs) for k, probs in enumerate(self._prev_action_probs) if k != exclude
        ]
        return torch.stack(embeds, dim=0).mean(dim=0) if embeds else torch.zeros(self.mean_field_dim)

    def act(self, env: RestorationEnv, rng: np.random.Generator, greedy: bool) -> tuple[list[int], list[torch.Tensor], torch.Tensor]:
        raw_actions, logps, probs_list = [], [], []
        for k, actor in enumerate(self.actors):
            obs = torch.as_tensor(env.local_observation(k, source_view=False)).float()
            mean_field = self._mean_field_context(k)
            logits = actor(torch.cat([obs, mean_field], dim=-1).unsqueeze(0))
            mask = torch.as_tensor(env.valid_local_action_mask(k)).unsqueeze(0)
            dist = masked_categorical(logits, mask)
            action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
            logps.append(dist.log_prob(action).squeeze(0))
            raw_actions.append(int(action.item()))
            probs_list.append(dist.probs.squeeze(0).detach())
        self._prev_action_probs = probs_list
        repaired = repair_joint_action(env, raw_actions)
        value = self.critic(torch.as_tensor(env.observation(source_view=False)).float().unsqueeze(0)).squeeze(0)
        return repaired, logps, value

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(global_obs)

    def reset_episode(self) -> None:
        self._prev_action_probs = None


class Candidate4bStudent(Candidate4Student):
    """Candidate 4 + sequential joint-radiality masking (see Candidate4Student docstring)."""

    name = "c4b_planner_distilled_sequential"

    def __init__(self, env: RestorationEnv, hidden_dim: int = 32):
        super().__init__(env, hidden_dim=hidden_dim, sequential_joint_mask=True)


CANDIDATES = {
    "c1": Candidate1,
    "c2": Candidate2,
    "c3": Candidate3,
    "c4": Candidate4Student,
    "c4b": Candidate4bStudent,
    "c5": Candidate5,
}


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)
