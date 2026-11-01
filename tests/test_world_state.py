"""PASS 4: authoritative world-state spine.

Verifies stable IDs, explicit parent-child relationships, units/provenance on
quantities, local clocks, and versioned checkpoint/restore for the chain
universe -> planet -> habitat -> organism without replacing the existing models.
"""

import numpy as np
import pytest

from universe_engine import world_state as ws_mod
from universe_engine.core.universe import Universe
from universe_engine.habitat import Habitat, make_default_network
from universe_engine.planet.model import Planet, PlanetSpec
from universe_engine.research.persistent_organism import make_organism
from universe_engine.world_state import Provenance, Quantity, WorldState, connect_chain, chain_state, WORLD_SCHEMA


def test_stable_ids_and_parent_chain():
    ws = WorldState(universe_id="uni-1")
    uni = Universe(universe_id="uni-1", name="Demo", seed=1,
                   cosmology={"H0_km_s_Mpc": 67.4, "Omega_m": 0.315, "Omega_r": 9.2e-5,
                              "Omega_Lambda": 0.684908})
    planet = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    net = make_default_network()
    hab = Habitat(planet, net, n_patches=4, seed=3)
    org = make_organism(architecture="ecs_iw", seed=11, habitat=hab, patch_index=0)

    ids = connect_chain(ws, uni, planet, hab, org)
    assert ids["universe"] and ids["planet"] and ids["habitat"] and ids["organism"]

    chain = ws.chain(ids["organism"])
    kinds = [e.kind for e in chain]
    assert kinds == ["organism", "habitat", "planet", "universe"]
    # parent links are explicit
    assert ws.entity(ids["organism"]).parent_id == ids["habitat"]
    assert ws.entity(ids["habitat"]).parent_id == ids["planet"]
    assert ws.entity(ids["planet"]).parent_id == ids["universe"]
    assert ws.entity(ids["universe"]).parent_id is None


def test_quantity_units_and_provenance():
    q = Quantity(288.0, "K", Provenance.SIMULATED, "planet.model", uncertainty=0.5)
    d = q.to_dict()
    assert d["unit"] == "K" and d["provenance"] == "simulated" and d["model_id"] == "planet.model"
    assert d["uncertainty"] == 0.5
    q2 = Quantity.from_dict(d)
    assert q2.value == 288.0 and q2.uncertainty == 0.5

    # illustrative vs simulated remain distinguishable
    qi = Quantity(1.0, "au", Provenance.ILLUSTRATIVE, "illustrative.model")
    assert qi.provenance == "illustrative" and qi.provenance != Provenance.SIMULATED


def test_planet_fields_carry_units_and_provenance():
    ws = WorldState(universe_id="uni-2")
    uni = Universe(universe_id="uni-2", name="Demo", seed=1,
                   cosmology={"H0_km_s_Mpc": 67.4, "Omega_m": 0.315, "Omega_r": 9.2e-5,
                              "Omega_Lambda": 0.684908})
    planet = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    net = make_default_network()
    hab = Habitat(planet, net, n_patches=4, seed=3)
    ids = connect_chain(ws, uni, planet, hab)
    pe = ws.entity(ids["planet"])
    assert pe.fields["T_global_K"].unit == "K"
    assert pe.fields["pCO2_bar"].unit == "bar"
    assert pe.fields["a_au"].unit == "AU"
    assert pe.fields["a_au"].provenance == Provenance.ANALYTIC   # derived from spec, not simulated
    assert pe.fields["T_global_K"].provenance == Provenance.SIMULATED


def test_checkpoint_restore_reproduces_chain():
    ws = WorldState(universe_id="uni-3")
    uni = Universe(universe_id="uni-3", name="Demo", seed=1,
                   cosmology={"H0_km_s_Mpc": 67.4, "Omega_m": 0.315, "Omega_r": 9.2e-5,
                              "Omega_Lambda": 0.684908})
    planet = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    planet.evolve(5.0, dt_gyr=0.05)
    net = make_default_network()
    hab = Habitat(planet, net, n_patches=4, seed=3)
    for _ in range(4):
        hab.step(1e-6)
    org = make_organism(architecture="ecs_iw", seed=11, habitat=hab, patch_index=0)
    for _ in range(3):
        org.step()
    connect_chain(ws, uni, planet, hab, org)

    cp = ws.checkpoint(code_sha="abc123")
    assert cp["schema"] == WORLD_SCHEMA
    assert cp["code_sha"] == "abc123"
    assert cp["entities"]

    ws2 = WorldState.restore(cp)
    assert len(ws2.entities) == 4
    assert ws2.clock("uni-3").integration_time_s == ws.clock("uni-3").integration_time_s
    p2 = ws2.entity("planet-uni-3")
    p1 = ws.entity("planet-uni-3")
    assert p2.fields["pCO2_bar"].value == p1.fields["pCO2_bar"].value
    assert p2.fields["pCO2_bar"].unit == p1.fields["pCO2_bar"].unit
    assert p2.fields["pCO2_bar"].provenance == p1.fields["pCO2_bar"].provenance


def test_chain_state_includes_habitat_with_planet():
    ws = WorldState(universe_id="uni-4")
    uni = Universe(universe_id="uni-4", name="Demo", seed=1,
                   cosmology={"H0_km_s_Mpc": 67.4, "Omega_m": 0.315, "Omega_r": 9.2e-5,
                              "Omega_Lambda": 0.684908})
    planet = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    net = make_default_network()
    hab = Habitat(planet, net, n_patches=4, seed=3)
    ids = connect_chain(ws, uni, planet, hab)
    st = chain_state(ws, hab)
    assert st["schema"] == WORLD_SCHEMA
    assert st["habitat"]["schema"] == "habitat/v1"
    assert "planet" in st["habitat"] and st["habitat"]["planet"]["schema"] == "planet/v1"
