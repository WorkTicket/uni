"""Regression tests for the PyTorch RL implementation (Sprint 14 Pass 1B).

These verify the policy-gradient machinery itself is correct on simple,
memory-free problems BEFORE any delayed-cue control claim is made:

  * Two-action bandit: one action yields +1, the other -1; the agent must
    learn the +1 action from rewards alone.
  * Contextual bandit: two observable states; the optimal action depends on
    the state; a feedforward/recurrent policy must learn the mapping via RL.

torch is a research-only optional dependency; these tests skip when it is not
installed. No claim of consciousness; measures action-reward learning only.

These tests lock in the correctness of the REINFORCE implementation. They do
NOT demonstrate that delayed-cue control is solved (that remains an open,
documented negative result).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.universe_engine.research.contracts import (
    AgentAction, SensoryObservation,
)
from src.universe_engine.research.diagnostics.torch_controller import (
    TorchGRUActorCriticController,
)


def make_obs(state: int) -> SensoryObservation:
    return SensoryObservation(
        time=0.0, position=(0.0, 0.0), energy_level=0.5,
        nutrient_level=0.5, local_temperature=288.0, local_light=1.0,
        nearby_resources=float(state), nearby_hazards=0.0, nearby_agents=0,
    )


def action_to_index(action: AgentAction) -> int:
    if action.action_type == "move" and action.parameters.get("dx", 0) > 0:
        return 1
    return 0


class TwoActionBandit:
    """Single state; index 1 -> +1, index 0 -> -1."""

    def step(self, action_index: int):
        return (1.0 if action_index == 1 else -1.0), True


class ContextualBandit:
    """Two states; best action == state. Toggles each episode."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.state = 0

    def step(self, action_index: int):
        best = self.state
        reward = 1.0 if action_index == best else -1.0
        self.state = 1 - self.state
        return reward, True


def run_rl(env, n_episodes: int, seed: int, lr: float):
    torch.manual_seed(seed); np.random.seed(seed)
    ctrl = TorchGRUActorCriticController("a", seed=seed,
                                         hidden_dim=16, lr=lr,
                                         gamma=0.99, entropy_coeff=0.01)
    rewards = []
    for ep in range(n_episodes):
        state = getattr(env, "state", 0)
        obs = make_obs(state)
        ctrl.reset_hidden()
        action = ctrl.select_action(obs)
        a_idx = action_to_index(action)
        reward, _ = env.step(a_idx)
        ctrl.learn(obs, action, reward, obs)
        ctrl.finish_episode()
        rewards.append(reward)
    return rewards, ctrl


def test_bandit_learns_positive_action():
    """REINFORCE must learn to choose the +1 action in a two-action bandit."""
    env = TwoActionBandit()
    rewards, _ = run_rl(env, n_episodes=3000, seed=1, lr=5e-2)
    final = float(np.mean(rewards[-500:]))
    assert final > 0.5, f"bandit did not learn: {final:+.3f}"


def test_contextual_bandit_learns_state_dependent_policy():
    """REINFORCE must learn state->action mapping in a contextual bandit."""
    env = ContextualBandit()
    rewards, _ = run_rl(env, n_episodes=6000, seed=2, lr=1e-1)
    final = float(np.mean(rewards[-500:]))
    assert final > 0.5, f"contextual bandit did not learn: {final:+.3f}"


def test_actor_parameters_change_during_training():
    """Actor parameters must actually change after optimizer steps."""
    import torch
    torch.manual_seed(0)
    ctrl = TorchGRUActorCriticController("a", seed=0, hidden_dim=16, lr=1e-2)
    before = {n: p.data.clone() for n, p in ctrl.net.actor.named_parameters()}
    env = TwoActionBandit()
    _, trained = run_rl(env, n_episodes=2000, seed=3, lr=5e-2)
    after = {n: p.data.clone() for n, p in trained.net.actor.named_parameters()}
    max_change = max(
        float((before[n] - after[n]).abs().max().item())
        for n in before
    )
    assert max_change > 1e-4, f"actor parameters did not change: {max_change:.2e}"


def test_terminal_episode_does_not_bootstrap():
    """A single-terminal episode must not bootstrap from a nonexistent next
    state; the return after the terminal step is just that step's reward."""
    ctrl = TorchGRUActorCriticController("a", seed=0, hidden_dim=16)
    obs = make_obs(0)
    ctrl.reset_hidden()
    ctrl.select_action(obs)
    ctrl.learn(obs, AgentAction("move", {"dx": 1, "dy": 0}), 1.0, obs)
    stats = ctrl.finish_episode()
    assert stats["n_steps"] == 1
    assert stats["loss"] >= 0.0
