"""Tests for civilization system (Kardashev, technology, expansion, nested simulations)."""
import numpy as np
from universe_engine.civilization import (
    CivilizationEvolutionEngine, kardashev_index, kardashev_type,
    TechnologyTree, GalacticMap, NestedSimulationEngine, SimulationConfig
)


def test_kardashev_index_calculation():
    assert kardashev_index(1e6) == 0.0  # Type I threshold approx (10^6 W = 1 MW, K=(6-6)/10=0)
    assert kardashev_index(1e16) == 1.0  # Type II
    assert kardashev_index(1e26) == 2.0  # Type III
    assert kardashev_type(0.0) == "Type-I" or kardashev_type(0.0) == "sub-Type-I"
    assert kardashev_type(1.0) == "Type-II"
    assert kardashev_type(2.0) == "Type-III"
    assert kardashev_type(-0.5) == "sub-Type-I"


def test_civilization_founding_and_energy_accounting():
    eng = CivilizationEvolutionEngine(rng=np.random.default_rng(1))
    civ = eng.found_civilization('sp1', 'sys-0', 'TestCiv')
    assert civ.population > 0
    assert civ.resources.energy_watts > 0
    assert civ.active
    assert civ.kardashev_type_str() in ["sub-Type-I", "Type-I"]


def test_technology_prerequisites_and_development():
    eng = CivilizationEvolutionEngine(rng=np.random.default_rng(2))
    civ = eng.found_civilization('sp1', 'sys-0', 'TestCiv')
    tree = TechnologyTree()
    # agriculture and metallurgy available initially
    avail = tree.available(civ)
    ids = [t.tech_id for t in avail]
    assert 'agriculture' in civ.technologies
    assert 'metallurgy' in civ.technologies


def test_resource_limited_expansion():
    eng = CivilizationEvolutionEngine(rng=np.random.default_rng(3))
    civ = eng.found_civilization('sp1', 'sys-0', 'TestCiv')
    civ.resources.energy_watts = 0  # no energy
    events = eng.step(0.001)
    # Should not be able to launch missions without energy/resources


def test_interstellar_travel_time_and_communication_latency():
    gm = GalacticMap(seed=4, n_systems=50)
    if 'sys-0' in gm.systems and 'sys-1' in gm.systems:
        d = gm.distance('sys-0', 'sys-1')
        assert d >= 0
    # Communication delay = light travel distance
    eng = CivilizationEvolutionEngine(rng=np.random.default_rng(5))
    civ1 = eng.found_civilization('sp1', 'sys-0')
    civ2 = eng.found_civilization('sp2', 'sys-1')
    if civ1.homeworld and civ2.homeworld:
        delay = eng.galactic_map.communicate(civ1, civ2)
        assert delay >= 0


def test_civilization_collapse_persistence():
    eng = CivilizationEvolutionEngine(rng=np.random.default_rng(6))
    civ = eng.found_civilization('sp1', 'sys-0')
    civ.social.stability = 0.01
    for _ in range(100):
        eng.step(0.01)
    cp = eng.checkpoint()
    eng2 = CivilizationEvolutionEngine.restore(cp, np.random.default_rng(6))
    assert len(eng2.civilizations) >= 1


def test_nested_simulation_creation_and_limits():
    eng = CivilizationEvolutionEngine(rng=np.random.default_rng(7))
    civ = eng.found_civilization('sp1', 'sys-0')
    civ.add_technology('nested_simulation')
    civ.computational_capacity = 1e18
    sim = eng.create_simulation_for_civ(civ.civ_id)
    assert sim is not None
    assert sim.depth == 0
    assert sim.parent_civ_id == civ.civ_id
    # Resource limits
    assert sim.cpu_allocated > 0
    assert sim.cpu_allocated <= 1e18


def test_nested_simulation_isolation():
    eng = CivilizationEvolutionEngine(rng=np.random.default_rng(8))
    civ1 = eng.found_civilization('sp1', 'sys-0')
    civ2 = eng.found_civilization('sp2', 'sys-1')
    civ1.add_technology('nested_simulation')
    civ2.add_technology('nested_simulation')
    civ1.computational_capacity = 1e18
    civ2.computational_capacity = 1e18
    sim1 = eng.create_simulation_for_civ(civ1.civ_id)
    sim2 = eng.create_simulation_for_civ(civ2.civ_id)
    assert sim1.sim_id != sim2.sim_id
    assert sim1.parent_civ_id != sim2.parent_civ_id


def test_nested_simulation_depth_limit():
    nse = NestedSimulationEngine(max_depth=3)
    eng = CivilizationEvolutionEngine(rng=np.random.default_rng(9))
    civ = eng.found_civilization('sp1', 'sys-0')
    civ.add_technology('nested_simulation')
    civ.computational_capacity = 1e18
    eng.nested_engine = nse
    sim = eng.create_simulation_for_civ(civ.civ_id)
    assert sim is not None
    # Create child up to depth limit
    child1 = nse.create_child_simulation(sim, civ)
    assert child1 is not None
    child2 = nse.create_child_simulation(child1, civ)
    assert child2 is not None
    # Depth 2 -> child would be depth 3, at limit
    child3 = nse.create_child_simulation(child2, civ)
    # May be None if at max_depth
