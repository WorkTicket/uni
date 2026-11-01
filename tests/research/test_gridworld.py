"""Tests for the minimal embodied agent environment (Sprint 3).

Tests GridWorld, ReactiveController, RecurrentWorldModelController,
and the agent/environment contracts.
"""

import numpy as np
import pytest

from universe_engine.research.environment.gridworld import (
    GridWorld, GridWorldConfig, ReactiveController,
    RecurrentWorldModelController,
)
from universe_engine.research.contracts import (
    AgentState, SensoryObservation, AgentAction,
)


# ---------------------------------------------------------------------------
# GridWorld tests
# ---------------------------------------------------------------------------

def test_gridworld_creation_and_reset():
    """GridWorld creates with proper grid and resets cleanly."""
    gw = GridWorld(GridWorldConfig(width=4, height=4, seed=1))
    assert gw.cfg.width == 4
    assert gw.cfg.height == 4
    assert len(gw.patches) == 16
    obs = gw.reset()
    assert gw.t == 0.0
    assert gw.step_count == 0
    assert obs.time == 0.0


def test_gridworld_register_and_observe():
    """Agent can register and receive a sensory observation."""
    gw = GridWorld(GridWorldConfig(seed=1))
    gw.register_agent("agent-1", (2, 2), initial_energy=5.0)
    obs = gw.observe("agent-1")
    assert obs.position == (2, 2)
    assert obs.energy_level == pytest.approx(0.5)  # 5.0 / 10.0
    assert obs.local_temperature > 0
    assert 0.0 <= obs.nearby_resources <= 1.0
    assert obs.nearby_hazards >= 0.0


def test_gridworld_observe_unknown_agent_fails():
    """Observing an unregistered agent raises an error."""
    gw = GridWorld(GridWorldConfig(seed=1))
    with pytest.raises(ValueError):
        gw.observe("nonexistent")


def test_gridworld_move_action():
    """Move action changes agent position and costs energy."""
    gw = GridWorld(GridWorldConfig(seed=1))
    gw.register_agent("agent-1", (2, 2), initial_energy=5.0)
    e0 = gw._agent_states["agent-1"].energy
    result = gw.act("agent-1", AgentAction("move", {"dx": 1, "dy": 0}))
    assert result["success"]
    assert gw._agent_positions["agent-1"] == (3, 2)
    assert gw._agent_states["agent-1"].energy < e0


def test_gridworld_forage_action():
    """Forage action consumes food and gains energy/nutrients."""
    gw = GridWorld(GridWorldConfig(seed=1))
    gw.register_agent("agent-1", (2, 2), initial_energy=1.0)
    patch = gw.patches[(2, 2)]
    food0 = patch.food
    e0 = gw._agent_states["agent-1"].energy
    result = gw.act("agent-1", AgentAction("forage"))
    assert result["success"]
    assert patch.food < food0  # food consumed
    assert gw._agent_states["agent-1"].energy > e0  # energy gained


def test_gridworld_rest_action():
    """Rest action recovers energy."""
    gw = GridWorld(GridWorldConfig(seed=1))
    gw.register_agent("agent-1", (2, 2), initial_energy=1.0)
    e0 = gw._agent_states["agent-1"].energy
    result = gw.act("agent-1", AgentAction("rest"))
    assert result["success"]
    assert gw._agent_states["agent-1"].energy > e0


def test_gridworld_energy_death():
    """Agent dies when energy reaches zero."""
    gw = GridWorld(GridWorldConfig(seed=1, move_cost=10.0))
    gw.register_agent("agent-1", (2, 2), initial_energy=0.1)
    result = gw.act("agent-1", AgentAction("move", {"dx": 1, "dy": 0}))
    assert not gw._agent_states["agent-1"].alive


def test_gridworld_step_regenerates_food():
    """Step advances time and regenerates food."""
    gw = GridWorld(GridWorldConfig(seed=1))
    gw.register_agent("agent-1", (2, 2), initial_energy=5.0)
    # Consume all food at a patch
    pos = (2, 2)
    gw.patches[pos].food = 0.0
    gw.step(dt=10.0)
    assert gw.patches[pos].food > 0.0  # regenerated
    assert gw.t == 10.0


def test_gridworld_checkpoint_restore():
    """Checkpoint and restore preserve environment state."""
    gw = GridWorld(GridWorldConfig(seed=1))
    gw.register_agent("agent-1", (2, 2), initial_energy=5.0)
    gw.step(dt=5.0)
    gw.act("agent-1", AgentAction("move", {"dx": 1, "dy": 0}))
    cp = gw.checkpoint()

    gw2 = GridWorld(GridWorldConfig(seed=1))
    gw2.register_agent("agent-1", (0, 0), initial_energy=1.0)
    gw2.restore(cp)

    assert gw2.t == gw.t
    assert gw2.step_count == gw.step_count
    assert gw2._agent_positions["agent-1"] == gw._agent_positions["agent-1"]
    assert gw2._agent_states["agent-1"].energy == gw._agent_states["agent-1"].energy


# ---------------------------------------------------------------------------
# ReactiveController tests
# ---------------------------------------------------------------------------

def test_reactive_controller_forages_when_energy_low_and_food_present():
    """Reactive controller forages when energy low and resources nearby."""
    ctrl = ReactiveController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.2,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.8,
                             nearby_hazards=0.0, nearby_agents=0)
    action = ctrl.select_action(obs)
    assert action.action_type == "forage"


def test_reactive_controller_moves_when_energy_low_no_food():
    """Reactive controller moves when energy low but no food."""
    ctrl = ReactiveController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.2,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.1,
                             nearby_hazards=0.0, nearby_agents=0)
    action = ctrl.select_action(obs)
    assert action.action_type == "move"


def test_reactive_controller_rest_when_energy_high():
    """Reactive controller rests when energy high and no food."""
    ctrl = ReactiveController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.9,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.1,
                             nearby_hazards=0.0, nearby_agents=0)
    action = ctrl.select_action(obs)
    assert action.action_type == "rest"


def test_reactive_controller_no_learning():
    """Reactive controller learn() is a no-op."""
    ctrl = ReactiveController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.5,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.0,
                             nearby_hazards=0.0, nearby_agents=0)
    action = AgentAction("wait")
    ctrl.learn(obs, action, 0.0, obs)
    assert ctrl.actions_taken == 0  # learn doesn't increment


# ---------------------------------------------------------------------------
# RecurrentWorldModelController tests
# ---------------------------------------------------------------------------

def test_recurrent_controller_selects_action():
    """Recurrent world model controller returns a valid action."""
    ctrl = RecurrentWorldModelController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.5,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    action = ctrl.select_action(obs)
    assert action.action_type in ("move", "forage", "rest", "wait")


def test_recurrent_controller_learns_and_reduces_error():
    """Recurrent controller learns from experience, reducing prediction error."""
    ctrl = RecurrentWorldModelController("agent-1", seed=1, lr=0.1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.5,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    next_obs = SensoryObservation(time=0.1, position=(0, 0), energy_level=0.6,
                                  nutrient_level=1.0, local_temperature=288.0,
                                  local_light=1.0, nearby_resources=0.5,
                                  nearby_hazards=0.0, nearby_agents=0)
    action = AgentAction("rest")
    errors_before = len(ctrl.prediction_error_history)
    ctrl.learn(obs, action, 0.1, next_obs)
    assert len(ctrl.prediction_error_history) == errors_before + 1


def test_recurrent_controller_checkpoint_restore():
    """Recurrent controller checkpoint/restore preserves state."""
    ctrl = RecurrentWorldModelController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.5,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    action = AgentAction("rest")
    ctrl.select_action(obs)
    ctrl.learn(obs, action, 0.1, obs)
    cp = ctrl.checkpoint()

    ctrl2 = RecurrentWorldModelController("agent-1", seed=99)
    ctrl2.restore(cp)
    assert ctrl2.hidden.tolist() == ctrl.hidden.tolist()
    assert ctrl2.actions_taken == ctrl.actions_taken


# ---------------------------------------------------------------------------
# Integrated environment-agent loop
# ---------------------------------------------------------------------------

def test_agent_environment_loop_survives():
    """Agent and environment can interact over multiple steps."""
    gw = GridWorld(GridWorldConfig(seed=1, initial_food=1.0))
    gw.register_agent("agent-1", (2, 2), initial_energy=5.0)
    ctrl = ReactiveController("agent-1", seed=1)
    alive_steps = 0
    for _ in range(50):
        obs = gw.observe("agent-1")
        action = ctrl.select_action(obs)
        result = gw.act("agent-1", action)
        gw.step()
        if not gw._agent_states["agent-1"].alive:
            break
        alive_steps += 1
    assert alive_steps > 0

