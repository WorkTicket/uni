"""Tests for the Planck-scale theoretical research layer (Requirement 9 / Test I)."""
import math
import pytest

from universe_engine.frontier.planck_scale import (
    PlanckUnits, PLANCK_UNITS, C_LIGHT, H_BAR, G_NEWTON, K_BOLTZMANN,
    QuantumGeometryModel, DiscreteGeometryModel, SpinNetworkInspiredModel,
    PlanckScaleLayer, PLANCK_LAYER,
)


# ---------------------------------------------------------------- Planck units
def test_planck_length_calculation():
    """l_P = sqrt(hbar * G / c^3)"""
    expected = math.sqrt(H_BAR * G_NEWTON / C_LIGHT**3)
    assert PLANCK_UNITS.planck_length_m == pytest.approx(expected, rel=1e-10)
    assert PLANCK_UNITS.planck_length_m > 0
    # Planck length is ~1.6e-35 m
    assert PLANCK_UNITS.planck_length_m == pytest.approx(1.616e-35, rel=5e-3)


def test_planck_time_calculation():
    """t_P = sqrt(hbar * G / c^5)"""
    expected = math.sqrt(H_BAR * G_NEWTON / C_LIGHT**5)
    assert PLANCK_UNITS.planck_time_s == pytest.approx(expected, rel=1e-10)
    assert PLANCK_UNITS.planck_time_s > 0
    assert PLANCK_UNITS.planck_time_s == pytest.approx(5.391e-44, rel=5e-3)


def test_planck_mass_calculation():
    """m_P = sqrt(hbar * c / G)"""
    expected = math.sqrt(H_BAR * C_LIGHT / G_NEWTON)
    assert PLANCK_UNITS.planck_mass_kg == pytest.approx(expected, rel=1e-10)
    assert PLANCK_UNITS.planck_mass_kg > 0
    assert PLANCK_UNITS.planck_mass_kg == pytest.approx(2.176e-8, rel=5e-3)


def test_planck_energy_calculation():
    """E_P = m_P * c^2"""
    expected = PLANCK_UNITS.planck_mass_kg * C_LIGHT**2
    assert PLANCK_UNITS.planck_energy_J == pytest.approx(expected, rel=1e-10)
    assert PLANCK_UNITS.planck_energy_J > 0


def test_planck_temperature():
    """T_P = E_P / k_B"""
    expected = PLANCK_UNITS.planck_energy_J / K_BOLTZMANN
    assert PLANCK_UNITS.planck_temperature_K == pytest.approx(expected, rel=1e-10)
    assert PLANCK_UNITS.planck_temperature_K > 0


def test_planck_density():
    """rho_P = m_P / l_P^3"""
    expected = PLANCK_UNITS.planck_mass_kg / PLANCK_UNITS.planck_length_m**3
    assert PLANCK_UNITS.planck_density_kg_m3 == pytest.approx(expected, rel=1e-10)


def test_planck_units_dict():
    d = PLANCK_UNITS.to_dict()
    assert "planck_length_m" in d
    assert "planck_time_s" in d
    assert "planck_mass_kg" in d
    assert "planck_energy_J" in d


# ---------------------------------------------------------------- dimensionless
def test_dimensionless_scale():
    layer = PlanckScaleLayer()
    l_p = layer.planck_length()
    # 1 Planck length = 1 in Planck units
    assert layer.dimensionless_scale(l_p) == pytest.approx(1.0)
    # 10 Planck lengths = 10
    assert layer.dimensionless_scale(10 * l_p) == pytest.approx(10.0)


def test_dimensionless_time():
    layer = PlanckScaleLayer()
    t_p = layer.planck_time()
    assert layer.dimensionless_time(t_p) == pytest.approx(1.0)


# ---------------------------------------------------------------- discrete geometry
def test_discrete_geometry_initial_state():
    model = DiscreteGeometryModel(max_elements=100)
    state = model.initial_state()
    assert state["n_elements"] == 1
    assert state["adjacency"].shape == (1, 1)
    assert state["t"] == 0.0


def test_discrete_geometry_step():
    model = DiscreteGeometryModel(max_elements=1000)
    state = model.initial_state()
    state = model.step(state, dt=1e-43)
    assert state["n_elements"] == 2
    assert state["adjacency"].shape == (2, 2)
    assert state["t"] == pytest.approx(1e-43)


def test_discrete_geometry_observables():
    model = DiscreteGeometryModel(max_elements=1000)
    state = model.initial_state()
    state = model.step(state, dt=1e-43)
    obs = model.observables(state)
    assert obs["n_elements"] == 2
    assert "link_density" in obs


def test_discrete_geometry_max_elements():
    model = DiscreteGeometryModel(max_elements=5)
    state = model.initial_state()
    for _ in range(100):
        state = model.step(state, dt=1e-43)
    assert state["n_elements"] <= 5  # bounded


# ---------------------------------------------------------------- spin network
def test_spin_network_initial_state():
    model = SpinNetworkInspiredModel()
    state = model.initial_state()
    assert len(state["edges"]) == 1
    assert state["edges"][0]["spin"] == 0.5


def test_spin_network_area_spectrum():
    model = SpinNetworkInspiredModel(gamma=0.234)
    l_p2 = PLANCK_UNITS.planck_length_m**2
    # j=1/2: A = 8*pi*gamma*l_P^2*sqrt(0.5*1.5)
    expected = 8.0 * math.pi * 0.234 * l_p2 * math.sqrt(0.5 * 1.5)
    assert model.area_spectrum(0.5) == pytest.approx(expected, rel=1e-10)


def test_spin_network_observables():
    model = SpinNetworkInspiredModel()
    state = model.initial_state()
    obs = model.observables(state)
    assert obs["n_edges"] == 1
    assert obs["total_area_m2"] > 0


def test_spin_network_step():
    model = SpinNetworkInspiredModel()
    state = model.initial_state()
    state = model.step(state, dt=1e-43)
    assert len(state["edges"]) == 2


# ---------------------------------------------------------------- Planck layer
def test_planck_layer_register_model():
    layer = PlanckScaleLayer()
    assert len(layer.models) == 0
    model = DiscreteGeometryModel()
    layer.register_model(model)
    assert "discrete_geometry" in layer.models


def test_global_planck_layer_has_models():
    assert "discrete_geometry" in PLANCK_LAYER.models
    assert "spin_network_inspired" in PLANCK_LAYER.models


def test_planck_layer_to_dict():
    d = PLANCK_LAYER.to_dict()
    assert "planck_units" in d
    assert "models" in d


def test_model_theory_classification_speculative():
    """All Planck-scale models are labeled speculative/research, not established."""
    for model in PLANCK_LAYER.models.values():
        assert model.theory_classification in ("speculative", "research")

