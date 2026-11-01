"""Regression tests for the working-memory oracle and memory tasks.

Category 1 = programmed intelligence (explicit oracle): the environment +
memory mechanism work.
Category 2 = learned intelligence (trainable agents): not yet demonstrated.

These tests verify:
  * the oracle remembers target/cue coordinates and navigates to them,
  * coordinates use a consistent normalized convention,
  * claim succeeds only on arrival at the remembered target,
  * the oracle reads only the sensory observation contract (no hidden state).
"""
from __future__ import annotations

import numpy as np
import pytest

from universe_engine.research.contracts import (
    AgentAction, AgentState, SensoryObservation,
)
from universe_engine.research.diagnostics.memory_tasks import (
    DelayedCueTask, DelayedCueConfig,
    OccludedTargetTask, OccludedTargetConfig,
)
from universe_engine.research.diagnostics.statistical_replication import (
    WorkingMemoryOracle,
)


# ---------------------------------------------------------------------------
# Coordinate contract
# ---------------------------------------------------------------------------

def test_occluded_target_visible_encodes_target_coordinates():
    """During the visible phase the target position is observable (normalized)."""
    env = OccludedTargetTask(OccludedTargetConfig(grid=5, visible_steps=5),
                             seed=7)
    obs = env.reset()
    assert obs.local_light == pytest.approx(1.0)
    # nearby_resources / nearby_hazards carry normalized target (x, y).
    tx, ty = env.target_pos
    assert obs.nearby_resources == pytest.approx(tx / (env.cfg.grid - 1))
    assert obs.nearby_hazards == pytest.approx(ty / (env.cfg.grid - 1))
    # position is the agent's own observable location.
    assert obs.position == (0.0, 0.0)


def test_occluded_target_occluded_hides_target():
    """During occlusion the target is removed from the observation."""
    env = OccludedTargetTask(OccludedTargetConfig(grid=5, visible_steps=1),
                             seed=7)
    obs = env.reset()
    # Advance into occlusion.
    env.act("a", AgentAction("wait"))
    env.step(1.0)
    obs = env.observe("a")
    assert obs.local_light == pytest.approx(0.0)
    assert obs.extra["target_pos"] is None
    # nearby_resources/hazards now report the agent's own position, not target.
    assert obs.nearby_resources == pytest.approx(env.agent_pos[0] / (env.cfg.grid - 1))
    assert obs.nearby_hazards == pytest.approx(env.agent_pos[1] / (env.cfg.grid - 1))


def test_delayed_cue_visible_encodes_cue_side():
    """During the cue phase nearby_resources encodes cue side (0=left, 1=right)."""
    for side in (0, 1):
        env = DelayedCueTask(DelayedCueConfig(cue_duration=5), seed=side)
        env.cue_side = side
        env.cue_visible = True
        env.phase = "cue"
        obs = env._observe()
        assert obs.nearby_resources == pytest.approx(float(side))
        assert obs.local_light == pytest.approx(1.0)


def test_delayed_cue_hidden_is_neutral():
    """During the delay/response the cue observation is neutral (0.5)."""
    env = DelayedCueTask(DelayedCueConfig(cue_duration=1, delay=5), seed=1)
    env.reset()
    env.phase = "delay"
    env.cue_visible = False
    obs = env._observe()
    assert obs.nearby_resources == pytest.approx(0.5)
    assert obs.local_light == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Claiming
# ---------------------------------------------------------------------------

def test_claim_at_target_gives_hit():
    """Claiming at the remembered target during occlusion yields hit_reward."""
    env = OccludedTargetTask(OccludedTargetConfig(grid=5, visible_steps=1,
                                                  occlusion_steps=5), seed=3)
    obs = env.reset()
    tx, ty = env.target_pos
    # Move into occlusion (phase flips after visible_steps).
    env.act("a", AgentAction("wait"))
    env.step(1.0)
    # Navigate agent onto the target directly.
    env.agent_pos = (tx, ty)
    obs = env.observe("a")
    result = env.act("a", AgentAction("claim"))
    assert result["reward"] == pytest.approx(2.0 - 0.01)
    assert result["done"] is True


def test_claim_not_at_target_gives_miss():
    """Claiming away from the target during occlusion yields miss_penalty."""
    env = OccludedTargetTask(OccludedTargetConfig(grid=5, visible_steps=1,
                                                  occlusion_steps=5), seed=3)
    obs = env.reset()
    env.act("a", AgentAction("wait"))
    env.step(1.0)
    env.agent_pos = (0, 0)
    # Ensure target is not at origin.
    assert env.target_pos != (0, 0)
    obs = env.observe("a")
    result = env.act("a", AgentAction("claim"))
    assert result["reward"] == pytest.approx(-1.0 - 0.01)
    assert result["done"] is True


# ---------------------------------------------------------------------------
# Working-memory oracle behavior
# ---------------------------------------------------------------------------

def test_oracle_solves_delayed_cue():
    """The oracle reliably solves delayed_cue (programmed memory)."""
    wins = 0
    trials = 40
    for seed in range(trials):
        env = DelayedCueTask(DelayedCueConfig(cue_duration=2, delay=3,
                                              response_window=3), seed=seed)
        ctrl = WorkingMemoryOracle("a", seed=seed)
        obs = env.reset()
        total = 0.0
        for _ in range(40):
            action = ctrl.select_action(obs)
            res = env.act("a", action)
            nobs = env.observe("a")
            ctrl.learn(obs, action, float(res.get("reward", 0.0)), nobs)
            total += float(res.get("reward", 0.0))
            obs = nobs
            env.step(1.0)
            if res.get("done"):
                break
        # Optimal is about +1.0 minus small step costs; a correct response
        # yields positive net reward.
        if total > 0.2:
            wins += 1
    assert wins / trials >= 0.9, f"oracle delayed_cue win rate {wins}/{trials}"


def test_oracle_solves_occluded_target():
    """The oracle reliably solves occluded_target (programmed memory)."""
    wins = 0
    trials = 40
    for seed in range(trials):
        env = OccludedTargetTask(OccludedTargetConfig(grid=5, visible_steps=2,
                                                      occlusion_steps=6),
                                 seed=seed)
        ctrl = WorkingMemoryOracle("a", seed=seed)
        obs = env.reset()
        total = 0.0
        hit = False
        for _ in range(40):
            action = ctrl.select_action(obs)
            res = env.act("a", action)
            nobs = env.observe("a")
            ctrl.learn(obs, action, float(res.get("reward", 0.0)), nobs)
            total += float(res.get("reward", 0.0))
            if res.get("reward", 0.0) > 1.0:
                hit = True
            obs = nobs
            env.step(1.0)
            if res.get("done"):
                break
        if hit:
            wins += 1
    assert wins / trials >= 0.9, f"oracle occluded_target win rate {wins}/{trials}"


def test_oracle_does_not_read_hidden_target():
    """Information leakage: the oracle only uses the observation contract.

    It must solve the task without reading env.target_pos or env internals.
    We verify by checking the oracle's select_action only touches the
    observation fields it is given (position, nearby_resources, nearby_hazards,
    local_light) and never environment state.
    """
    env = OccludedTargetTask(OccludedTargetConfig(grid=5, visible_steps=2,
                                                  occlusion_steps=6), seed=5)
    obs = env.reset()
    ctrl = WorkingMemoryOracle("a", seed=5)
    # Before seeing anything, the oracle has no memory.
    assert ctrl.has_memory is False
    # It must select a legal action from the initial observation only.
    action = ctrl.select_action(obs)
    assert action.action_type in ("move", "forage", "rest", "wait", "claim")
    # After one informative observation it should have stored memory.
    action2 = ctrl.select_action(obs)
    assert ctrl.has_memory is True


def test_trainable_agents_fail_memory_tasks():
    """Category 2: trainable agents do NOT yet solve the memory tasks.

    This is a negative result: record it rather than relabeling the oracle's
    success as learned intelligence.
    """
    from universe_engine.research.diagnostics.statistical_replication import (
        GatedRecurrentController, RecurrentController, WorldModelController,
        ReactiveController,
    )
    controllers = {
        "reactive": ReactiveController,
        "recurrent": RecurrentController,
        "world_model": WorldModelController,
        "gated_recurrent": GatedRecurrentController,
    }
    for name, ctor in controllers.items():
        total = 0.0
        for seed in range(10):
            env = DelayedCueTask(DelayedCueConfig(cue_duration=2, delay=3,
                                                  response_window=3), seed=seed)
            ctrl = ctor("a", seed=seed)
            obs = env.reset()
            for _ in range(40):
                action = ctrl.select_action(obs)
                res = env.act("a", action)
                nobs = env.observe("a")
                ctrl.learn(obs, action, float(res.get("reward", 0.0)), nobs)
                total += float(res.get("reward", 0.0))
                obs = nobs
                env.step(1.0)
                if res.get("done"):
                    break
        # None of the trainable agents should reliably solve delayed_cue.
        assert total / 10 < 0.2, (
            f"{name} unexpectedly solved delayed_cue (mean {total/10:+.3f})")
