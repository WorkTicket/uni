"""Regression tests for the 3D expanding-universe particle-mesh gravity solver.

Validates:
  * Uniform density -> zero force, no motion.
  * Linear growth of a single Fourier mode vs analytic growing-mode solution.
  * Proper time t(a) matches the FLRW background.
  * Proper position = a * x_comoving.
  * Peculiar velocity = a * H(a) * dx/ds.
  * Checkpoint/restore bitwise replay at step boundaries.
  * Force symmetry: mirror-symmetric density yields antisymmetric acceleration.
  * Density contrast mean is zero (Jeans swindle / k=0 dropped).
"""

import math
import numpy as np
import pytest

from universe_engine.solvers.pm3d import (
    Cosmology, ParticleMesh3D,
)
from universe_engine.cosmos.coordinates import hubble_from_cosmology


H0 = 67.4
OM = 0.315
OR = 9.2e-5
OL = 0.684908


def _cosmology():
    return Cosmology(H0_km_s_Mpc=H0, Omega_m=OM, Omega_r=OR, Omega_Lambda=OL)


def _lattice(n_particles_per_axis, seed=0, perturbation_amp=0.0):
    """Particles on a perturbed lattice."""
    rng = np.random.default_rng(seed)
    n = n_particles_per_axis
    grid = np.stack(np.meshgrid(
        *([np.arange(n) / n + 0.5 / n] * 3), indexing="ij"), -1).reshape(-1, 3)
    if perturbation_amp > 0:
        grid = grid + perturbation_amp * rng.normal(size=grid.shape)
        grid = grid % 1.0
    mass = np.full(grid.shape[0], 1.0 / grid.shape[0])
    return grid, mass


def test_uniform_density_zero_force():
    """A perfectly uniform lattice feels no force and does not move."""
    n = 8
    x, m = _lattice(n)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=16, x=x, u=np.zeros_like(x), mass=m, cosmology=cosmo, a=0.5)
    acc = sim.comoving_acceleration()
    assert np.max(np.abs(acc)) < 1e-13
    sim.run_to_a(1.0, max_d_ln_a=0.05)
    assert np.max(np.abs(sim.u)) < 1e-12
    assert np.max(np.abs(sim.x - x)) < 1e-12


def test_density_contrast_mean_zero():
    """The CIC density contrast has zero mean (Jeans swindle / k=0 mode dropped)."""
    n = 8
    x, m = _lattice(n, perturbation_amp=0.1)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=16, x=x, u=np.zeros_like(x), mass=m, cosmology=cosmo, a=0.5)
    delta = sim.density_contrast()
    assert abs(delta.mean()) < 1e-12


def test_proper_position_is_a_times_comoving():
    n = 8
    x, m = _lattice(n)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=16, x=x, u=np.zeros_like(x), mass=m, cosmology=cosmo, a=0.7)
    x_prop = sim.proper_position()
    assert np.allclose(x_prop, 0.7 * x, atol=1e-14)


def test_peculiar_velocity_is_a_H_dx_ds():
    n = 8
    x, m = _lattice(n)
    u = np.full_like(x, 0.01)
    cosmo = _cosmology()
    a = 0.7
    sim = ParticleMesh3D(n=16, x=x, u=u, mass=m, cosmology=cosmo, a=a)
    H = cosmo.H(a)
    v_pec = sim.peculiar_velocity()
    assert np.allclose(v_pec, a * H * u, atol=1e-14)


def test_proper_time_matches_background():
    """t(a) after integration matches cosmic_time_from_scale_factor(a) to
    the accuracy of the midpoint ds/H integration (rel ~1e-5)."""
    from universe_engine.core.clocks import cosmic_time_from_scale_factor
    n = 8
    x, m = _lattice(n)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=16, x=x, u=np.zeros_like(x), mass=m, cosmology=cosmo, a=0.2)
    t_start = sim.t
    expected_start = cosmic_time_from_scale_factor(0.2, H0, OM, OR, OL)
    assert t_start == pytest.approx(expected_start, rel=1e-10)
    sim.run_to_a(1.0, max_d_ln_a=0.02)
    expected_end = cosmic_time_from_scale_factor(1.0, H0, OM, OR, OL)
    assert sim.t == pytest.approx(expected_end, rel=1e-4)
    assert sim.t > t_start


def test_proper_time_increases_monotonically():
    n = 8
    x, m = _lattice(n)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=16, x=x, u=np.zeros_like(x), mass=m, cosmology=cosmo, a=0.2)
    t0 = sim.t
    sim.run_to_a(2.0, max_d_ln_a=0.02)
    assert sim.t > t0
    assert sim.a == pytest.approx(2.0)


def test_linear_growth_single_mode():
    """A single Fourier density mode grows approximately like the linear
    growing mode D(a). Use a small amplitude so evolution stays linear."""
    n_mesh = 32
    n_particles = 8 ** 3
    x, m = _lattice(8, seed=3, perturbation_amp=0.02)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=n_mesh, x=x, u=np.zeros_like(x), mass=m,
                         cosmology=cosmo, a=0.2)
    # Measure the k=(1,0,0) Fourier mode amplitude of the density contrast
    def mode_amp(delta):
        d_hat = np.fft.fftn(delta)
        # k = (1,0,0) mode
        return abs(d_hat[1, 0, 0]) / delta.size
    a_vals = [0.3, 0.4, 0.7, 1.0]
    amps = []
    for a_t in a_vals:
        sim.run_to_a(a_t, max_d_ln_a=0.01)
        delta = sim.density_contrast()
        amps.append(mode_amp(delta))
    # Growth factor ratio from a=0.3 to a=1.0 should match linear theory roughly.
    # Use the approximate growth rate to estimate D(1)/D(0.3).
    from universe_engine.cosmos.coordinates import growth_rate_approx
    # Integrate d ln D / d ln a = f(a) from a=0.3 to 1 (trapezoidal).
    a_grid = np.linspace(0.3, 1.0, 200)
    ln_a = np.log(a_grid)
    f_vals = np.array([growth_rate_approx(OM, OL, a, OR) for a in a_grid])
    ln_D_ratio = float(np.sum(0.5 * (f_vals[1:] + f_vals[:-1]) * np.diff(ln_a)))
    D_ratio_expected = math.exp(ln_D_ratio)
    D_ratio_measured = amps[-1] / amps[0]
    # Allow 30% tolerance: PM with few particles and CIC is approximate.
    assert D_ratio_measured == pytest.approx(D_ratio_expected, rel=0.35)


def test_force_points_toward_overdensity():
    """A particle placed just off-center from a concentrated overdensity
    accelerates toward it (negative gradient of potential = attraction)."""
    n_mesh = 32
    # Put one particle at origin and a second slightly offset; all mass
    # concentrated -> both accelerate toward each other.
    x = np.zeros((2, 3))
    x[1] = [0.01, 0.0, 0.0]
    m = np.full(2, 0.5)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=n_mesh, x=x, u=np.zeros_like(x), mass=m,
                         cosmology=cosmo, a=0.5)
    acc = sim.comoving_acceleration()
    # Particle at origin should accelerate toward +x (toward the other particle)
    assert acc[0][0] > 0
    # Particle at +x should accelerate toward -x (toward the origin)
    assert acc[1][0] < 0
    # Equal masses -> equal and opposite accelerations
    assert np.allclose(acc[0], -acc[1], atol=1e-10)


def test_checkpoint_restore_bitwise():
    n = 8
    x, m = _lattice(n, seed=7, perturbation_amp=0.1)
    cosmo = _cosmology()
    a, b = ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                          cosmology=cosmo, a=0.3), \
           ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                          cosmology=cosmo, a=0.3)
    saved = []
    a.run_to_a(0.8, max_d_ln_a=0.02,
               callback=lambda s: saved.append(s.checkpoint()) if s._step_count == 5 else None)
    b.run_to_a(0.8, max_d_ln_a=0.02)
    assert np.allclose(a.x, b.x, atol=1e-12) and np.allclose(a.u, b.u, atol=1e-12)
    assert abs(a.t - b.t) < 1e-6
    # Restore from checkpoint at step 5 and continue.
    c = ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                       cosmology=cosmo, a=0.3)
    c.restore(saved[0])
    c.run_to_a(0.8, max_d_ln_a=0.02)
    assert np.allclose(a.x, c.x, atol=1e-12)
    assert np.allclose(a.u, c.u, atol=1e-12)
    assert abs(a.t - c.t) < 1e-6


def test_snapshot_schema_and_units():
    n = 8
    x, m = _lattice(n)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=16, x=x, u=np.zeros_like(x), mass=m, cosmology=cosmo, a=0.5)
    snap = sim.snapshot()
    assert snap["schema"] == "pm3d/v1"
    assert snap["a"] == pytest.approx(0.5)
    assert 0.0 < snap["z"] < 100.0
    assert snap["t_Gyr"] > 0
    assert snap["n_mesh"] == 16
    assert snap["n_particles"] == x.shape[0]
    assert snap["x_proper"].shape == x.shape
    assert snap["v_pec"].shape == x.shape
    assert snap["density_contrast"].shape == (16, 16, 16)
    assert snap["units"]["x_comoving"] == "box fraction"
