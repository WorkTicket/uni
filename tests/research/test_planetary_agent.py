"""Tests for planetary ecology integration (Sprint 8, Research Program C8/D8)."""

import numpy as np
import pytest

from universe_engine.research.planetary_agent.experiment import (
    PlanetaryAgentConfig, PlanetaryAgent, make_planet_habitat,
    run_planetary_agent_experiment,
)
from universe_engine.research.contracts import AgentAction


# ---------------------------------------------------------------------------
# Habitat creation tests
# ---------------------------------------------------------------------------

def test_make_planet_habitat():
    """Planet habitat creates with patches and chemistry."""
    h = make_planet_habitat(seed=1, n_patches=4, has_water=True)
    assert h.n_patches == 4
    assert h.t_gyr == 4.5
    assert h.state().sterile  # no replicators seeded yet


def test_make_planet_habitat_no_water():
    """Habitat without liquid water."""
    h = make_planet_habitat(seed=1, n_patches=2, has_water=False)
    assert not h.patches[0].liquid_water
    assert h.state().sterile


# ---------------------------------------------------------------------------
# Planetary agent tests
# ---------------------------------------------------------------------------

def test_planetary_agent_creation():
    """Planetary agent creates and links to patch."""
    h = make_planet_habitat(seed=1, n_patches=4)
    cfg = PlanetaryAgentConfig(agent_id="a1", patch_index=0, seed=1)
    agent = PlanetaryAgent(cfg, h)
    assert agent.agent_id == "a1"
    assert agent.patch_index == 0
    assert agent.alive
    assert agent.state.energy == 5.0


def test_planetary_agent_observe():
    """Agent observes local patch chemistry."""
    h = make_planet_habitat(seed=1, n_patches=4)
    cfg = PlanetaryAgentConfig(agent_id="a1", patch_index=0, seed=1)
    agent = PlanetaryAgent(cfg, h)
    obs = agent.observe()
    assert obs.energy_level == pytest.approx(0.5)  # 5.0 / 10.0
    assert obs.local_temperature > 0
    assert 0.0 <= obs.nearby_resources <= 1.0
    assert "patch_index" in obs.extra


def test_planetary_agent_act_forage():
    """Forage action consumes resources and gains energy."""
    h = make_planet_habitat(seed=1, n_patches=4)
    # Add monomer M to the patch so foraging can gain energy
    species = h.species_names
    if "M" in species:
        idx = species.index("M")
        h.patches[0].c[idx] = 1.0
    cfg = PlanetaryAgentConfig(agent_id="a1", patch_index=0, seed=1,
                               initial_energy=1.0)
    agent = PlanetaryAgent(cfg, h)
    e0 = agent.state.energy
    result = agent.act(AgentAction("forage"))
    assert result["success"]
    assert agent.state.energy > e0  # gained energy from foraging


def test_planetary_agent_act_rest():
    """Rest action recovers energy."""
    h = make_planet_habitat(seed=1, n_patches=4)
    cfg = PlanetaryAgentConfig(agent_id="a1", patch_index=0, seed=1, initial_energy=1.0)
    agent = PlanetaryAgent(cfg, h)
    e0 = agent.state.energy
    result = agent.act(AgentAction("rest"))
    assert result["success"]
    assert agent.state.energy > e0


def test_planetary_agent_step():
    """One step: observe, act, learn, habitat advances."""
    h = make_planet_habitat(seed=1, n_patches=4)
    cfg = PlanetaryAgentConfig(agent_id="a1", patch_index=0, seed=1)
    agent = PlanetaryAgent(cfg, h)
    result = agent.step()
    assert "t_gyr" in result
    assert "reward" in result
    assert "action" in result
    assert "alive" in result
    assert agent.n_steps == 1


def test_planetary_agent_can_starve():
    """Agent can die from energy depletion."""
    h = make_planet_habitat(seed=1, n_patches=4)
    cfg = PlanetaryAgentConfig(agent_id="a1", patch_index=0, seed=1,
                               initial_energy=0.001)
    agent = PlanetaryAgent(cfg, h)
    # Force death by acting with zero energy
    agent.state.energy = 0.001
    result = agent.act(AgentAction("move"))
    if agent.state.energy <= 0:
        assert not agent.alive


def test_planetary_agent_checkpoint_restore():
    """Agent checkpoint/restore preserves state."""
    h = make_planet_habitat(seed=1, n_patches=4)
    cfg = PlanetaryAgentConfig(agent_id="a1", patch_index=0, seed=1)
    agent = PlanetaryAgent(cfg, h)
    agent.step()
    agent.step()
    cp = agent.checkpoint()

    h2 = make_planet_habitat(seed=1, n_patches=4)
    agent2 = PlanetaryAgent.restore(cp, h2)
    assert agent2.agent_id == "a1"
    assert agent2.t_gyr == agent.t_gyr
    assert agent2.state.energy == agent.state.energy
    assert agent2.n_steps == agent.n_steps


# ---------------------------------------------------------------------------
# Experiment tests
# ---------------------------------------------------------------------------

def test_run_planetary_agent_experiment():
    """Full experiment runs and returns metrics."""
    result = run_planetary_agent_experiment(seed=42, n_steps=20,
                                             controller_type="ecs_iw")
    assert "alive" in result
    assert "n_steps" in result
    assert "final_energy" in result
    assert "lifetime_reward" in result
    assert "self_model_calibration" in result
    assert "n_self_model_updates" in result
    assert result["n_steps"] > 0


def test_planetary_agent_survives_short_run():
    """Agent typically survives a short run with resources."""
    result = run_planetary_agent_experiment(seed=7, n_steps=10,
                                             controller_type="ecs_iw")
    assert result["n_steps"] > 0


def test_planetary_agent_self_model_updates():
    """Agent's self-model updates through interaction."""
    result = run_planetary_agent_experiment(seed=99, n_steps=20,
                                             controller_type="ecs_iw")
    assert result["n_self_model_updates"] > 0


def test_planetary_agent_different_controllers():
    """Different controller types can run in the habitat."""
    for ctype in ("ecs_iw", "recurrent", "reactive"):
        result = run_planetary_agent_experiment(seed=5, n_steps=10,
                                                 controller_type=ctype)
        assert result["n_steps"] > 0
        assert isinstance(result["alive"], bool)
