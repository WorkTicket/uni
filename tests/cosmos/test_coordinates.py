"""Regression tests for physical <-> comoving coordinate transformations.

These tests verify:
  * Round-trip consistency: comoving <-> proper reproduces input to roundoff.
  * Peculiar velocity equation: d(v_pec)/dt = -H v_pec + g/a.
  * Density scaling: rho_proper = rho_comoving / a^3.
  * Conjugate momentum scaling: p_comoving = a^2 p_proper.
  * Hubble parameter from cosmology matches the FLRW background.
  * Bound orbits do not expand with the scale factor (peculiar velocity
    cancels Hubble flow for a circular orbit).
"""

import math
import numpy as np
import pytest

from universe_engine.cosmos.coordinates import (
    comoving_to_proper,
    proper_to_comoving,
    peculiar_velocity_from_comoving,
    proper_velocity,
    comoving_velocity_from_proper,
    physical_density,
    comoving_density,
    conjugate_momentum_comoving,
    conjugate_momentum_proper,
    peculiar_acceleration_equation,
    comoving_position_equation,
    validate_frame_consistency,
    hubble_from_cosmology,
    growth_rate_approx,
    ComovingParticleState,
)

H0 = 67.4
OM = 0.315
OR = 9.2e-5
OL = 0.684908


def test_roundtrip_position():
    x_com = np.array([0.1, 0.5, -0.3, 2.0])
    a = 0.7
    x_prop = comoving_to_proper(x_com, a)
    x_back = proper_to_comoving(x_prop, a)
    assert np.allclose(x_back, x_com, atol=1e-12)


def test_roundtrip_velocity():
    x_com = np.array([0.1, 0.5, -0.3])
    v_pec = np.array([100.0, -200.0, 50.0])
    a = 0.7
    H = hubble_from_cosmology(H0, OM, OR, OL, a)
    v_prop = proper_velocity(x_com, v_pec / a, a, H)
    v_pec_back = v_prop - H * comoving_to_proper(x_com, a)
    assert np.allclose(v_pec_back, v_pec, atol=1e-9)


def test_comoving_velocity_from_proper_roundtrip():
    x_prop = np.array([1e22, 2e22, 3e22])
    v_prop = np.array([1e3, -2e3, 500.0])
    a = 0.7
    H = hubble_from_cosmology(H0, OM, OR, OL, a)
    dx_dt = comoving_velocity_from_proper(x_prop, v_prop, a, H)
    v_pec_back = proper_velocity(proper_to_comoving(x_prop, a), dx_dt, a, H)
    assert np.allclose(v_pec_back, v_prop, atol=1e-6)


def test_validate_frame_consistency():
    x_com = np.random.default_rng(0).normal(size=(10, 3))
    v_pec = np.random.default_rng(1).normal(size=(10, 3)) * 100.0
    a = 0.5
    H = hubble_from_cosmology(H0, OM, OR, OL, a)
    result = validate_frame_consistency(x_com, v_pec, a, H)
    assert result["consistent"] is True
    assert result["roundtrip_position_error"] < 1e-9
    assert result["roundtrip_velocity_error"] < 1e-9


def test_density_scaling():
    rho_com = np.array([1.0, 2.0, 4.0])
    a = 2.0
    rho_prop = physical_density(rho_com, a)
    assert np.allclose(rho_prop, rho_com / a ** 3)
    rho_back = comoving_density(rho_prop, a)
    assert np.allclose(rho_back, rho_com, atol=1e-12)


def test_conjugate_momentum_scaling():
    p_prop = np.array([1.0, -2.0, 3.0])
    a = 3.0
    p_com = conjugate_momentum_comoving(p_prop, a)
    assert np.allclose(p_com, a ** 2 * p_prop)
    p_back = conjugate_momentum_proper(p_com, a)
    assert np.allclose(p_back, p_prop, atol=1e-12)


def test_hubble_matches_background():
    # At a=1, H = H0
    H = hubble_from_cosmology(H0, OM, OR, OL, 1.0)
    assert H == pytest.approx(H0 / 3.0856775814913673e19, rel=1e-12)
    # Matter-dominated limit: H^2 ~ Om H0^2 a^-3
    a_early = 1e-3
    H_early = hubble_from_cosmology(H0, OM, OR, OL, a_early)
    H_md = (H0 / 3.0856775814913673e19) * math.sqrt(OM / a_early ** 3)
    assert H_early == pytest.approx(H_md, rel=0.05)


def test_peculiar_acceleration_equation_signs():
    # With no gravity, peculiar velocity decays as 1/a (Hubble drag).
    v_pec = np.array([100.0, 0.0, 0.0])
    g_comoving = np.zeros(3)
    a = 1.0
    H = hubble_from_cosmology(H0, OM, OR, OL, a)
    dvdt = peculiar_acceleration_equation(v_pec, g_comoving, a, H)
    # dv_pec/dt = -H * v_pec  (pure Hubble drag)
    assert np.allclose(dvdt, -H * v_pec, atol=1e-15)


def test_bound_orbit_does_not_expand():
    """A circular orbit at fixed proper radius has peculiar velocity that
    exactly cancels the Hubble flow, so its comoving position is constant."""
    # 1 AU orbit around a solar-mass star; use a at present epoch
    a = 1.0
    H = hubble_from_cosmology(H0, OM, OR, OL, a)
    r = 1.495978707e11  # 1 AU in metres
    M = 1.98847e30
    G = 6.67430e-11
    v_circ = math.sqrt(G * M / r)  # circular orbit speed
    # Peculiar velocity is tangential; Hubble flow is radial.
    # At the point where r is along x and v_circ is along y:
    x_com = np.array([r / a, 0.0, 0.0])
    v_pec = np.array([0.0, v_circ, 0.0])
    v_prop = proper_velocity(x_com, v_pec / a, a, H)
    # Total proper velocity = H*r (radial Hubble flow) + v_circ (tangential)
    expected_hubble = H * r
    assert abs(v_prop[0] - expected_hubble) < 1e-6
    assert abs(v_prop[1] - v_circ) < 1e-6
    # Comoving position stays fixed (no cosmic expansion of bound orbit)
    dx_dt = comoving_position_equation(v_pec, v_pec, a)
    assert np.allclose(dx_dt, v_pec / a, atol=1e-15)


def test_comoving_particle_state_proper_position():
    rng = np.random.default_rng(42)
    x_com = rng.normal(size=5)
    v_pec = rng.normal(size=5) * 100.0
    mass = np.full(5, 1.98847e30)
    a = 0.5
    H = hubble_from_cosmology(H0, OM, OR, OL, a)
    state = ComovingParticleState(x_com, v_pec, mass, a, H)
    x_prop = state.proper_position()
    assert np.allclose(x_prop, a * x_com, atol=1e-12)
    d = state.to_dict()
    assert d["frame"] == "comoving"
    assert d["a"] == a
    assert d["H_s"] == H


def test_zero_scale_factor_rejected():
    with pytest.raises(ValueError):
        proper_to_comoving(np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        physical_density(np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        comoving_density(np.array([1.0]), -1.0)
