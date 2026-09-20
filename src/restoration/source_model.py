from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AttentionRegularizedLayer(nn.Module):
    """Intended A2Reg interpretation with a valid masked softmax."""

    def __init__(self, hidden_dim: int, attention_dim: int, drop_probability: float):
        super().__init__()
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, attention_dim)
        self.output = nn.Linear(attention_dim, hidden_dim)
        self.drop_probability = float(drop_probability)
        self.scale = attention_dim**-0.5

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        q = self.query(hidden)
        k = self.key(hidden)
        v = self.value(hidden)
        scores = q.unsqueeze(-1) * k.unsqueeze(-2) * self.scale
        if self.training and self.drop_probability > 0:
            keep = torch.rand_like(scores) >= self.drop_probability
            diagonal = torch.eye(scores.shape[-1], device=scores.device, dtype=torch.bool).unsqueeze(0)
            keep = keep | diagonal
            scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(weights, v.unsqueeze(-1)).squeeze(-1)
        return hidden + self.output(attended)


class Actor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, attention_dim: int, drop_probability: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.a2reg = AttentionRegularizedLayer(hidden_dim, attention_dim, drop_probability)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input(state))
        hidden = self.a2reg(hidden)
        hidden = F.silu(self.hidden(hidden))
        return self.output(hidden)


class CentralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int, drop_probability: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.a2reg = AttentionRegularizedLayer(hidden_dim, attention_dim, drop_probability)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input(features))
        hidden = self.a2reg(hidden)
        hidden = F.silu(self.hidden(hidden))
        return self.output(hidden).squeeze(-1)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class SourceConfig:
    hidden_dim: int
    actor_lr: float
    critic_lr: float
    discriminator_lr: float = 5e-4
    gamma: float = 0.95
    tau: float = 0.01
    alpha: float = 0.05
    cgag_weight: float = 0.2
    a2reg_drop_probability: float = 0.2
    attention_dim: int = 32
    batch_size: int = 2048
    buffer_size: int = 100_000


class ReplayBuffer:
    def __init__(self, capacity: int, n_switches: int, n_agents: int):
        self.capacity = int(capacity)
        self.n_switches = n_switches
        self.n_agents = n_agents
        self.states = np.zeros((capacity, n_switches), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_switches), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.valid = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, actions, reward, next_state, done, valid) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = actions
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.valid[i] = valid
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        indices = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.valid[indices],
        )


class CGenMARL(nn.Module):
    def __init__(
        self,
        n_switches: int,
        local_switch_indices: Sequence[np.ndarray],
        local_action_sizes: Sequence[int],
        config: SourceConfig,
        device: torch.device,
    ):
        super().__init__()
        self.n_switches = n_switches
        self.local_indices = [torch.as_tensor(x, dtype=torch.long, device=device) for x in local_switch_indices]
        self.action_sizes = tuple(int(x) for x in local_action_sizes)
        self.action_offsets = np.cumsum((0,) + self.action_sizes)
        self.config = config
        self.device = device
        self.actors = nn.ModuleList(
            [
                Actor(len(indices), config.hidden_dim, actions, config.attention_dim, config.a2reg_drop_probability)
                for indices, actions in zip(local_switch_indices, self.action_sizes)
            ]
        )
        central_input = n_switches + sum(self.action_sizes)
        self.critic = CentralNetwork(central_input, config.hidden_dim, config.attention_dim, config.a2reg_drop_probability)
        self.target_critic = CentralNetwork(central_input, config.hidden_dim, config.attention_dim, 0.0)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.discriminator = Discriminator(central_input, config.hidden_dim)
        self.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actors.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config.discriminator_lr)

    def action_vector(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [F.one_hot(actions[:, k], num_classes=size).float() for k, size in enumerate(self.action_sizes)], dim=-1
        )

    def select_actions(self, state: np.ndarray, epsilon: float, rng: np.random.Generator, deterministic: bool) -> list[int]:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = []
        self.eval()
        with torch.no_grad():
            for actor, indices, size in zip(self.actors, self.local_indices, self.action_sizes):
                if not deterministic and rng.random() < epsilon:
                    actions.append(int(rng.integers(0, size)))
                    continue
                logits = actor(tensor.index_select(1, indices))
                if deterministic:
                    actions.append(int(logits.argmax(dim=-1).item()))
                else:
                    actions.append(int(torch.distributions.Categorical(logits=logits).sample().item()))
        self.train()
        return actions

    def update(self, replay: ReplayBuffer, rng: np.random.Generator) -> dict[str, float] | None:
        if replay.size < self.config.batch_size:
            return None
        states, actions, rewards, next_states, dones, valid = replay.sample(self.config.batch_size, rng)
        s = torch.as_tensor(states, device=self.device)
        a = torch.as_tensor(actions, device=self.device)
        r = torch.as_tensor(rewards, device=self.device)
        ns = torch.as_tensor(next_states, device=self.device)
        done = torch.as_tensor(dones, device=self.device)
        validity = torch.as_tensor(valid, device=self.device)
        joint = self.action_vector(a)

        with torch.no_grad():
            next_probabilities = []
            entropy = torch.zeros(len(s), device=self.device)
            for actor, indices in zip(self.actors, self.local_indices):
                logits = actor(ns.index_select(1, indices))
                probability = torch.softmax(logits, dim=-1)
                next_probabilities.append(probability)
                entropy += -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=-1)
            next_joint = torch.cat(next_probabilities, dim=-1)
            target_q = self.target_critic(torch.cat([ns, next_joint], dim=-1))
            target = r + self.config.gamma * (1.0 - done) * (target_q + self.config.alpha * entropy)
        prediction = self.critic(torch.cat([s, joint], dim=-1))
        critic_loss = F.mse_loss(prediction, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        discriminator_logits = self.discriminator(torch.cat([s, joint], dim=-1))
        discriminator_loss = F.binary_cross_entropy_with_logits(discriminator_logits, validity)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
        self.discriminator_optimizer.step()

        current_probabilities = []
        actor_entropy = torch.zeros(len(s), device=self.device)
        for actor, indices in zip(self.actors, self.local_indices):
            logits = actor(s.index_select(1, indices))
            probability = torch.softmax(logits, dim=-1)
            current_probabilities.append(probability)
            actor_entropy += -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=-1)
        policy_joint = torch.cat(current_probabilities, dim=-1)
        policy_features = torch.cat([s, policy_joint], dim=-1)
        policy_q = self.critic(policy_features)
        feasibility_loss = -F.logsigmoid(self.discriminator(policy_features)).mean()
        actor_loss = -policy_q.mean() - self.config.alpha * actor_entropy.mean() + self.config.cgag_weight * feasibility_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 5.0)
        self.actor_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.mul_(1.0 - self.config.tau).add_(parameter, alpha=self.config.tau)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "discriminator_loss": float(discriminator_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "feasibility_loss": float(feasibility_loss.detach().cpu()),
        }


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
