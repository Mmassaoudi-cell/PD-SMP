"""Benchmark suite (Sec. 14). Ten-plus strong, distinct baselines, all sharing
one replay-buffer/Q-learning trainer or one PPO trainer so that differences in
results are attributable to the stated architectural/algorithmic axis, not to
incidental implementation drift between baselines.

Deliberately out of scope: literal MAES (evolutionary search) and MAGDPG
(graph deterministic policy gradient) reproductions. Both are complex,
non-standard, and expected to add implementation risk without a distinct
scientific question beyond what QMIX (value decomposition) and Graph-QMIX
(explicit structure) already answer; this deprioritization is documented here
rather than silently substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .environment import RestorationEnv
from .candidates import global_input_dim, local_input_dims, switch_adjacency, SwitchGraphEncoder, greedy_plan

Mode = Literal["dqn_centralized", "ppo_centralized", "madqn", "meanfield_q", "qmix", "masked_iql", "graph_qmix"]


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def slice_local(global_obs: np.ndarray, idx: np.ndarray, n_switches: int) -> np.ndarray:
    status = global_obs[..., idx]
    fault = global_obs[..., n_switches + idx]
    context = global_obs[..., -3:]
    return np.concatenate([status, fault, context], axis=-1)


class QHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QMixer(nn.Module):
    """Monotonic mixing network (QMIX): hypernetwork weights are made
    non-negative so dQ_tot/dQ_k >= 0 for every agent."""

    def __init__(self, n_agents: int, state_dim: int, mixing_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, mixing_dim), nn.SiLU(), nn.Linear(mixing_dim, 1))
        self.mixing_dim = mixing_dim

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = q_values.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.mixing_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.mixing_dim)
        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.mixing_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        out = torch.bmm(hidden, w2) + b2
        return out.view(batch)


@dataclass
class BenchmarkConfig:
    hidden_dim: int = 64
    lr: float = 5e-4
    gamma: float = 0.95
    tau: float = 0.02
    buffer_size: int = 50_000
    batch_size: int = 256
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    mean_field_dim: int = 8


class ReplayBuffer:
    def __init__(self, capacity: int, global_dim: int, n_agents: int):
        self.capacity = capacity
        self.global_obs = np.zeros((capacity, global_dim), dtype=np.float32)
        self.next_global_obs = np.zeros((capacity, global_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, g, a, r, ng, d) -> None:
        i = self.position
        self.global_obs[i] = g
        self.actions[i] = a
        self.rewards[i] = r
        self.next_global_obs[i] = ng
        self.dones[i] = d
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=min(batch_size, self.size))
        return self.global_obs[idx], self.actions[idx], self.rewards[idx], self.next_global_obs[idx], self.dones[idx]


class QLearningAgentSet(nn.Module):
    """Covers madqn / qmix / masked_iql / meanfield_q / dqn_centralized via flags."""

    def __init__(self, env: RestorationEnv, mode: Mode, config: BenchmarkConfig, device: torch.device):
        super().__init__()
        self.env = env
        self.mode = mode
        self.config = config
        self.device = device
        self.n_switches = env.n_switches
        self.local_idx = [env.local_switch_indices(k) for k in range(env.spec.n_agents)]
        self.sizes = env.local_action_sizes
        dims = local_input_dims(env)
        shared_trunk = mode == "dqn_centralized"
        if shared_trunk:
            trunk_dim = max(dims)
            self.pad_to = trunk_dim
            self.shared = QHead(trunk_dim, config.hidden_dim, max(self.sizes))
            self.q_nets = nn.ModuleList([self.shared for _ in dims])
        elif mode == "graph_qmix":
            adjacency = switch_adjacency(env)
            self.encoder = SwitchGraphEncoder(env.n_switches, 2, config.hidden_dim, adjacency)
            self.target_encoder = SwitchGraphEncoder(env.n_switches, 2, config.hidden_dim, adjacency)
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            self.q_nets = nn.ModuleList([QHead(config.hidden_dim * len(idx), config.hidden_dim, s) for idx, s in zip(self.local_idx, self.sizes)])
        elif mode == "meanfield_q":
            self.action_embed = nn.ModuleList([nn.Linear(s, config.mean_field_dim) for s in self.sizes])
            self.q_nets = nn.ModuleList([QHead(d + config.mean_field_dim, config.hidden_dim, s) for d, s in zip(dims, self.sizes)])
        else:
            self.q_nets = nn.ModuleList([QHead(d, config.hidden_dim, s) for d, s in zip(dims, self.sizes)])
        self.target_nets = nn.ModuleList([QHead(q.net[0].in_features, config.hidden_dim, q.net[-1].out_features) for q in self.q_nets])
        for t, q in zip(self.target_nets, self.q_nets):
            t.load_state_dict(q.state_dict())
        self.use_mixer = mode in ("qmix", "graph_qmix")
        if self.use_mixer:
            self.mixer = QMixer(env.spec.n_agents, global_input_dim(env), config.hidden_dim)
            self.target_mixer = QMixer(env.spec.n_agents, global_input_dim(env), config.hidden_dim)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        self._prev_probs: list[np.ndarray] | None = None

    def _local_inputs(self, global_obs: np.ndarray) -> list[np.ndarray]:
        if self.mode == "dqn_centralized":
            outs = []
            for idx in self.local_idx:
                x = slice_local(global_obs, idx, self.n_switches)
                pad_width = self.pad_to - x.shape[-1]
                if pad_width <= 0:
                    outs.append(x)
                    continue
                pad_shape = x.shape[:-1] + (pad_width,)
                pad = np.zeros(pad_shape, dtype=np.float32)
                outs.append(np.concatenate([x, pad], axis=-1))
            return outs
        return [slice_local(global_obs, idx, self.n_switches) for idx in self.local_idx]

    def reset_episode(self) -> None:
        self._prev_probs = None

    def act(self, env: RestorationEnv, epsilon: float, rng: np.random.Generator, greedy: bool) -> list[int]:
        global_obs = env.observation(source_view=False)
        actions = []
        probs_this_step = []
        if self.mode == "graph_qmix":
            faults = np.zeros(env.n_switches, dtype=np.float32)
            faults[list(env.scenario.faulted_switches)] = 1.0
            features = torch.as_tensor(np.stack([env.status.astype(np.float32), faults], axis=-1)).float()
            with torch.no_grad():
                embeddings = self.encoder(features)
        for k, q_net in enumerate(self.q_nets):
            mask = env.valid_local_action_mask(k) if self.mode == "masked_iql" else np.ones(2 * len(self.local_idx[k]) + 1, dtype=bool)
            if not greedy and rng.random() < epsilon:
                valid = np.flatnonzero(mask)
                action = int(rng.choice(valid)) if len(valid) else 0
                actions.append(action)
                probs_this_step.append(np.eye(len(mask))[action])
                continue
            with torch.no_grad():
                if self.mode == "meanfield_q":
                    x = torch.as_tensor(slice_local(global_obs, self.local_idx[k], self.n_switches)).float().unsqueeze(0)
                    mean_field = self._mean_field(k)
                    q = q_net(torch.cat([x, mean_field], dim=-1))
                elif self.mode == "graph_qmix":
                    local_embed = embeddings[self.local_idx[k]].reshape(-1).unsqueeze(0)
                    q = q_net(local_embed)
                else:
                    x = torch.as_tensor(self._local_inputs(global_obs)[k]).float().unsqueeze(0)
                    q = q_net(x)[:, : len(mask)]
                q_masked = q.clone()
                q_masked[0, ~torch.as_tensor(mask)] = torch.finfo(q.dtype).min
                action = int(q_masked.argmax(dim=-1).item())
            probs = np.zeros(len(mask), dtype=np.float32)
            probs[action] = 1.0
            probs_this_step.append(probs)
            actions.append(action)
        self._prev_probs = probs_this_step
        return actions

    def _mean_field(self, exclude: int) -> torch.Tensor:
        if self._prev_probs is None:
            return torch.zeros(1, self.config.mean_field_dim)
        embeds = [
            self.action_embed[k](torch.as_tensor(p).float().unsqueeze(0)) for k, p in enumerate(self._prev_probs) if k != exclude
        ]
        return torch.stack(embeds, dim=0).mean(dim=0) if embeds else torch.zeros(1, self.config.mean_field_dim)

    def _batch_mean_field(self, a_t: torch.Tensor, exclude: int) -> torch.Tensor:
        """Mean one-hot action embedding of every agent other than ``exclude``,
        computed from the joint action actually taken in the sampled
        transitions (standard mean-field Q-learning estimator; the same
        estimate is reused for the bootstrapped next-state term since the
        next joint action is not yet known)."""
        embeds = [
            self.action_embed[k](F.one_hot(a_t[:, k], num_classes=size).float())
            for k, size in enumerate(self.sizes)
            if k != exclude
        ]
        return torch.stack(embeds, dim=0).mean(dim=0) if embeds else torch.zeros(a_t.shape[0], self.config.mean_field_dim)

    def update(self, buffer: ReplayBuffer, rng: np.random.Generator) -> dict[str, float] | None:
        if buffer.size < self.config.batch_size:
            return None
        g, a, r, ng, d = buffer.sample(self.config.batch_size, rng)
        g_t, ng_t = torch.as_tensor(g), torch.as_tensor(ng)
        r_t, d_t = torch.as_tensor(r), torch.as_tensor(d)
        a_t = torch.as_tensor(a)
        current_qs, target_qs = [], []
        if self.mode == "graph_qmix":
            status_batch = torch.as_tensor(g[:, : self.n_switches])
            node_feats = torch.stack([status_batch, torch.as_tensor(g[:, self.n_switches : 2 * self.n_switches])], dim=-1)
            next_status_batch = torch.as_tensor(ng[:, : self.n_switches])
            next_node_feats = torch.stack([next_status_batch, torch.as_tensor(ng[:, self.n_switches : 2 * self.n_switches])], dim=-1)
        for k, (q_net, size) in enumerate(zip(self.q_nets, self.sizes)):
            if self.mode == "graph_qmix":
                embed = self.encoder(node_feats)
                local_embed = embed[:, self.local_idx[k], :].reshape(g.shape[0], -1)
                q_all = q_net(local_embed)
                with torch.no_grad():
                    next_embed = self.target_encoder(next_node_feats)
                    next_local_embed = next_embed[:, self.local_idx[k], :].reshape(g.shape[0], -1)
                    next_q_all = self.target_nets[k](next_local_embed)
            elif self.mode == "meanfield_q":
                x = torch.as_tensor(slice_local(g, self.local_idx[k], self.n_switches))
                nx = torch.as_tensor(slice_local(ng, self.local_idx[k], self.n_switches))
                mean_field = self._batch_mean_field(a_t, k)
                q_all = q_net(torch.cat([x, mean_field], dim=-1))
                with torch.no_grad():
                    next_q_all = self.target_nets[k](torch.cat([nx, mean_field], dim=-1))
            else:
                x = torch.as_tensor(self._local_inputs(g)[k])
                nx = torch.as_tensor(self._local_inputs(ng)[k])
                q_all = q_net(x)[:, :size]
                with torch.no_grad():
                    next_q_all = self.target_nets[k](nx)[:, :size]
            chosen_q = q_all.gather(1, a_t[:, k : k + 1]).squeeze(-1)
            next_q = next_q_all.max(dim=-1).values
            current_qs.append(chosen_q)
            target_qs.append(next_q)
        current = torch.stack(current_qs, dim=1)
        nxt = torch.stack(target_qs, dim=1)
        if self.use_mixer:
            q_tot = self.mixer(current, g_t)
            with torch.no_grad():
                target_tot = self.target_mixer(nxt, ng_t)
            target = r_t + self.config.gamma * (1 - d_t) * target_tot
            loss = F.mse_loss(q_tot, target)
        else:
            target = r_t.unsqueeze(1) + self.config.gamma * (1 - d_t.unsqueeze(1)) * nxt
            loss = F.mse_loss(current, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        with torch.no_grad():
            for t, q in zip(self.target_nets, self.q_nets):
                for tp, p in zip(t.parameters(), q.parameters()):
                    tp.mul_(1 - self.config.tau).add_(p, alpha=self.config.tau)
            if self.mode == "graph_qmix":
                for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                    tp.mul_(1 - self.config.tau).add_(p, alpha=self.config.tau)
            if self.use_mixer:
                for tp, p in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                    tp.mul_(1 - self.config.tau).add_(p, alpha=self.config.tau)
        return {"loss": float(loss.detach())}


class PPOAgentSet(nn.Module):
    """Centralized shared-trunk actor-critic with per-agent heads, clipped
    surrogate objective (represents the paper's 'single-agent PPO' baseline
    class applied to the factorized joint action)."""

    def __init__(self, env: RestorationEnv, config: BenchmarkConfig):
        super().__init__()
        dims = local_input_dims(env)
        self.pad_to = max(dims)
        self.sizes = env.local_action_sizes
        self.local_idx = [env.local_switch_indices(k) for k in range(env.spec.n_agents)]
        self.n_switches = env.n_switches
        self.trunk = nn.Sequential(nn.Linear(self.pad_to, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(config.hidden_dim, s) for s in self.sizes])
        self.critic = nn.Sequential(nn.Linear(global_input_dim(env), config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        self.clip = 0.2

    def _local(self, global_obs: np.ndarray, k: int) -> torch.Tensor:
        x = slice_local(global_obs, self.local_idx[k], self.n_switches)
        pad = np.zeros(self.pad_to - x.shape[-1], dtype=np.float32)
        return torch.as_tensor(np.concatenate([x, pad], axis=-1) if pad.size else x).float()

    def act(self, env: RestorationEnv, rng: np.random.Generator, greedy: bool):
        global_obs = env.observation(source_view=False)
        actions, logps = [], []
        for k, head in enumerate(self.heads):
            x = self._local(global_obs, k).unsqueeze(0)
            logits = head(self.trunk(x))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
            logps.append(dist.log_prob(action).squeeze(0).detach())
            actions.append(int(action.item()))
        value = self.critic(torch.as_tensor(global_obs).float().unsqueeze(0)).squeeze()
        return actions, logps, value

    def evaluate_actions(self, global_obs: torch.Tensor, actions: torch.Tensor):
        logps, entropies = [], []
        for k, head in enumerate(self.heads):
            x = torch.stack([self._local(g.numpy(), k) for g in global_obs])
            logits = head(self.trunk(x))
            dist = torch.distributions.Categorical(logits=logits)
            logps.append(dist.log_prob(actions[:, k]))
            entropies.append(dist.entropy())
        value = self.critic(global_obs).squeeze(-1)
        return torch.stack(logps, dim=1).sum(dim=1), torch.stack(entropies, dim=1).sum(dim=1), value


class RandomPolicy:
    name = "random"

    def __init__(self, env: RestorationEnv):
        self.env = env

    def act(self, env: RestorationEnv, rng: np.random.Generator, greedy: bool = True) -> list[int]:
        return [int(rng.integers(0, size)) for size in env.local_action_sizes]


class GreedyPlannerOnly:
    """No learning: re-plans with the beam-search oracle at every step."""

    name = "greedy_planner_only"

    def __init__(self, env: RestorationEnv, beam_width: int = 4):
        self.beam_width = beam_width

    def act(self, env: RestorationEnv, rng: np.random.Generator, greedy: bool = True) -> list[int]:
        return greedy_plan(env, beam_width=self.beam_width)[0]
