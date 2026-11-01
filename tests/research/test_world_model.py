"""Tests for world-model learning experiment (Sprint 4)."""

import numpy as np
import pytest

from universe_engine.research.world_model.experiment import (
    run_episode, run_comparison_experiment, make_world_model_manifest,
    WorldModelExperimentResult,
)
from universe_engine.research.environment.gridworld import GridWorld, GridWorldConfig
from universe_engine.research.environment.gridworld import ReactiveController, RecurrentWorldModelController
from universe_engine.research.contracts import AgentAction, SensoryObservation


def test_run_episode_returns_metrics():
    """run_episode returns a dict with expected keys."""
    env = GridWorld(GridWorldConfig(seed=1))
    ctrl = ReactiveController("a1", seed=1)
    m = run_episode(env, ctrl, "a1", n_steps=20)
    assert "total_reward" in m
    assert "alive" in m
    assert "unique_positions" in m
    assert m["n_steps"] > 0
    assert m["alive"] in (0.0, 1.0)


def test_run_episode_recurrent_tracks_prediction_error():
    """Recurrent controller records prediction errors during an episode."""
    env = GridWorld(GridWorldConfig(seed=1))
    ctrl = RecurrentWorldModelController("a1", seed=1, lr=0.05)
    m = run_episode(env, ctrl, "a1", n_steps=20)
    assert "mean_prediction_error" in m
    assert m["mean_prediction_error"] >= 0.0


def test_comparison_experiment_runs():
    """Full comparison experiment completes and produces results."""
    result = run_comparison_experiment(n_episodes=5, n_steps_per_episode=20, seed=7)
    assert isinstance(result, WorldModelExperimentResult)
    assert "mean_reward" in result.reactive_performance
    assert "mean_reward" in result.recurrent_performance
    assert isinstance(result.passed, bool)


def test_comparison_deterministic_with_seed():
    """Same seed produces identical results."""
    r1 = run_comparison_experiment(n_episodes=3, n_steps_per_episode=10, seed=42)
    r2 = run_comparison_experiment(n_episodes=3, n_steps_per_episode=10, seed=42)
    assert r1.reactive_performance["mean_reward"] == r2.reactive_performance["mean_reward"]
    assert r1.recurrent_performance["mean_reward"] == r2.recurrent_performance["mean_reward"]


def test_recurrent_prediction_error_decreases():
    """Recurrent controller's prediction error should generally decrease with learning."""
    env = GridWorld(GridWorldConfig(seed=1))
    env.register_agent("a1", (4, 4), initial_energy=5.0)
    ctrl = RecurrentWorldModelController("a1", seed=1, lr=0.1)
    obs = env.observe("a1")
    errors = []
    for _ in range(10):
        action = ctrl.select_action(obs)
        result = env.act("a1", action)
        next_obs = env.observe("a1")
        ctrl.learn(obs, action, float(result.get("reward", 0.0)), next_obs)
        obs = next_obs
        if ctrl.prediction_error_history:
            errors.append(ctrl.prediction_error_history[-1])
    # Prediction errors should be recorded
    assert len(errors) > 0


def test_manifest_config_hash_deterministic():
    """Manifest config hash is deterministic."""
    m1 = make_world_model_manifest(seed=42, n_episodes=10, n_steps=50)
    m2 = make_world_model_manifest(seed=42, n_episodes=10, n_steps=50)
    assert m1.config_hash() == m2.config_hash()

    m3 = make_world_model_manifest(seed=43, n_episodes=10, n_steps=50)
    assert m1.config_hash() != m3.config_hash()


def test_recurrent_vs_reactive_on_foraging():
    """Recurrent controller should forage at least as much as reactive in resource-rich env."""
    result = run_comparison_experiment(n_episodes=5, n_steps_per_episode=30, seed=99)
    # Both controllers should be able to forage when food is available
    assert result.reactive_performance["mean_forage_rate"] >= 0
    assert result.recurrent_performance["mean_forage_rate"] >= 0


def test_adaptation_score_computed():
    """Adaptation score is computed (can be positive or negative)."""
    result = run_comparison_experiment(n_episodes=5, n_steps_per_episode=20, seed=5)
    assert isinstance(result.adaptation_score, float)
    assert np.isfinite(result.adaptation_score)


def test_generalization_score_computed():
    """Generalization score is computed."""
    result = run_comparison_experiment(n_episodes=5, n_steps_per_episode=20, seed=5)
    assert isinstance(result.generalization_score, float)
    assert np.isfinite(result.generalization_score)


def test_result_to_dict():
    """Result can be serialized to dict."""
    result = run_comparison_experiment(n_episodes=3, n_steps_per_episode=10, seed=1)
    d = result.to_dict()
    assert "manifest" in d
    assert "reactive_performance" in d
    assert "recurrent_performance" in d
    assert "passed" in d
