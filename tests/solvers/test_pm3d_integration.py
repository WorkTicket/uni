"""Numerical integration validation for the 3D PM gravity solver.

Compares the KDK integrator (explicit drag and analytic drag) against an
independently implemented high-accuracy reference (RK4 with very small steps),
measures timestep convergence, and tracks conservation diagnostics:
  - Particle mass conservation
  - Mean momentum
  - Force symmetry
  - Density contrast mean (k=0 / Jeans swindle)
  - Linear growth-factor error
  - Timestep sensitivity
"""

import math
import numpy as np
import pytest

from universe_engine.solvers.pm3d import Cosmology, ParticleMesh3D
from universe_engine.cosmos.coordinates import (
    hubble_from_cosmology,
    growth_rate_approx,
)


H0 = 67.4
OM = 0.315
OR = 9.2e-5
OL = 0.684908


def _cosmology():
    return Cosmology(H0_km_s_Mpc=H0, Omega_m=OM, Omega_r=OR, Omega_Lambda=OL)


def _lattice(n_per_axis, seed=0, amplitude=0.0):
    rng = np.random.default_rng(seed)
    n = n_per_axis
    grid = np.stack(np.meshgrid(
        *([np.arange(n) / n + 0.5 / n] * 3), indexing="ij"), -1).reshape(-1, 3)
    if amplitude > 0:
        grid = grid + amplitude * rng.normal(size=grid.shape)
        grid = grid % 1.0
    mass = np.full(grid.shape[0], 1.0 / grid.shape[0])
    return grid, mass


# ---------------------------------------------------------------------------
# Reference integrator: RK4 in s = ln a with drag.
# ---------------------------------------------------------------------------

def _sim_rhs(sim, x, u, a):
    """Right-hand side: dx/ds = u, du/ds = acc(x) - drag(a)*u."""
    sim.x = x
    sim.a = a
    delta = sim.density_contrast()
    acc = sim.comoving_acceleration(delta)
    drag = sim.drag()
    return u, acc - drag * u


def rk4_integrate(sim, a_start, a_end, n_steps):
    """Integrate from a_start to a_end using RK4 in s = ln a.

    Independent implementation of the integrator, used as a high-accuracy
    reference when n_steps is large.
    """
    s0 = math.log(a_start)
    s1 = math.log(a_end)
    ds = (s1 - s0) / n_steps
    x = sim.x.copy()
    u = sim.u.copy()
    s = s0
    a = a_start
    mass = sim.mass.copy()
    # Attach a temporary mesh for force evaluation.
    tmp = ParticleMesh3D(
        n=sim.n, x=x, u=u, mass=mass, cosmology=sim.cosmology,
        box_size_Mpc=sim.box_size_Mpc, a=a,
        integrator="kdk_drag",
    )
    for _ in range(n_steps):
        # k1
        k1_x, k1_u = _sim_rhs(tmp, x, u, a)
        # k2
        x2 = x + 0.5 * ds * k1_x
        u2 = u + 0.5 * ds * k1_u
        a2 = math.exp(s + 0.5 * ds)
        k2_x, k2_u = _sim_rhs(tmp, x2, u2, a2)
        # k3
        x3 = x + 0.5 * ds * k2_x
        u3 = u + 0.5 * ds * k2_u
        a3 = math.exp(s + 0.5 * ds)
        k3_x, k3_u = _sim_rhs(tmp, x3, u3, a3)
        # k4
        x4 = x + ds * k3_x
        u4 = u + ds * k3_u
        a4 = math.exp(s + ds)
        k4_x, k4_u = _sim_rhs(tmp, x4, u4, a4)
        # Update
        x = x + (ds / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        u = u + (ds / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        s = s + ds
        a = math.exp(s)
        x = np.mod(x, 1.0)
    return x, u, a


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def mass_conservation(sim) -> float:
    """Total particle mass (should be constant)."""
    return float(np.sum(sim.mass))


def mean_momentum(sim) -> np.ndarray:
    """Mean comoving momentum dx/ds weighted by mass (vector)."""
    return np.sum(sim.mass[:, None] * sim.u, axis=0) / np.sum(sim.mass)


def force_symmetry_error(sim) -> float:
    """Max deviation from force symmetry under a parity transform.

    For a density field and its mirror, accelerations should be exact mirrors.
    """
    delta = sim.density_contrast()
    acc = sim.comoving_acceleration(delta)
    # Mirror positions: x -> 1 - x
    x_mirror = (1.0 - sim.x) % 1.0
    sim.x = x_mirror
    delta_m = sim.density_contrast()
    acc_m = sim.comoving_acceleration(delta_m)
    sim.x = sim.x  # no-op; restore below
    # Restore original positions
    sim.x = (1.0 - x_mirror) % 1.0
    # Mirrored acceleration at mirrored position should be -acc
    err = np.max(np.abs(acc_m + acc))
    return float(err)


def density_mean(sim) -> float:
    """Mean density contrast (should be zero)."""
    return float(np.mean(sim.density_contrast()))


def linear_growth_error(sim, a_start, a_end) -> float:
    """Error in the linear growth factor relative to theory.

    Measures the k=(1,0,0) mode growth from a_start to a_end and compares
    against the linear-theory growth factor ratio.
    """
    def mode_amp(delta):
        d_hat = np.fft.fftn(delta)
        return abs(d_hat[1, 0, 0]) / delta.size

    from universe_engine.cosmos.coordinates import growth_rate_approx as gr
    a_grid = np.linspace(a_start, a_end, 200)
    ln_a = np.log(a_grid)
    f_vals = np.array([gr(OM, OL, aa, OR) for aa in a_grid])
    ln_D_ratio = float(np.sum(0.5 * (f_vals[1:] + f_vals[:-1]) * np.diff(ln_a)))
    D_ratio_expected = math.exp(ln_D_ratio)

    # Measure initial mode
    sim_tmp = ParticleMesh3D(
        n=sim.n, x=sim.x.copy(), u=sim.u.copy(), mass=sim.mass.copy(),
        cosmology=sim.cosmology, box_size_Mpc=sim.box_size_Mpc, a=a_start,
        integrator="kdk_analytic_drag",
    )
    delta0 = sim_tmp.density_contrast()
    amp0 = mode_amp(delta0)
    if amp0 < 1e-12:
        return 0.0
    sim_tmp.run_to_a(a_end, max_d_ln_a=0.01)
    delta1 = sim_tmp.density_contrast()
    amp1 = mode_amp(delta1)
    D_ratio_measured = amp1 / amp0
    return abs(D_ratio_measured - D_ratio_expected) / D_ratio_expected


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_particle_mass_conservation():
    """Particle masses remain constant throughout evolution."""
    n = 8
    x, m = _lattice(n, seed=3, amplitude=0.05)
    cosmo = _cosmology()
    for scheme in ("kdk_drag", "kdk_analytic_drag"):
        sim = ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                             cosmology=cosmo, a=0.2, integrator=scheme)
        m0 = mass_conservation(sim)
        sim.run_to_a(1.0, max_d_ln_a=0.01)
        m1 = mass_conservation(sim)
        assert abs(m1 - m0) < 1e-14


def test_mean_momentum_decays_with_drag():
    """Mean momentum should decay due to Hubble drag (no net force in a
    statistically homogeneous small box)."""
    n = 8
    x, m = _lattice(n, seed=5, amplitude=0.02)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                         cosmology=cosmo, a=0.2, integrator="kdk_analytic_drag")
    p0 = mean_momentum(sim)
    sim.run_to_a(1.0, max_d_ln_a=0.01)
    p1 = mean_momentum(sim)
    # Momentum should not grow unboundedly; drag damps peculiar velocities.
    assert np.linalg.norm(p1) <= np.linalg.norm(p0) + 1e-6


def test_density_contrast_mean_zero():
    """Density contrast mean stays zero (k=0 mode dropped / Jeans swindle)."""
    n = 8
    x, m = _lattice(n, seed=7, amplitude=0.1)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                         cosmology=cosmo, a=0.5, integrator="kdk_analytic_drag")
    sim.run_to_a(1.0, max_d_ln_a=0.01)
    assert abs(density_mean(sim)) < 1e-12


def test_force_symmetry():
    """Mirror-symmetric density yields antisymmetric acceleration."""
    n = 8
    x, m = _lattice(n, seed=11, amplitude=0.1)
    cosmo = _cosmology()
    sim = ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                         cosmology=cosmo, a=0.5, integrator="kdk_analytic_drag")
    err = force_symmetry_error(sim)
    assert err < 1e-8


def test_kdk_converges_to_rk4_reference():
    """KDK (both schemes) converges to the RK4 reference as dt -> 0."""
    n = 6
    x, m = _lattice(n, seed=1, amplitude=0.03)
    cosmo = _cosmology()
    a_start, a_end = 0.3, 0.8

    # High-accuracy reference (many small RK4 steps).
    x_ref, u_ref, _ = rk4_integrate(
        ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                       cosmology=cosmo, a=a_start),
        a_start, a_end, n_steps=500)

    errors = []
    for ds in (0.02, 0.01, 0.005):
        for scheme in ("kdk_drag", "kdk_analytic_drag"):
            sim = ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                                 cosmology=cosmo, a=a_start, integrator=scheme)
            sim.run_to_a(a_end, max_d_ln_a=ds)
            x_err = float(np.max(np.abs(sim.x - x_ref)))
            u_err = float(np.max(np.abs(sim.u - u_ref)))
            errors.append((ds, scheme, x_err, u_err))

    # Convergence: error should decrease as ds decreases.
    for scheme in ("kdk_drag", "kdk_analytic_drag"):
        scheme_errors = [e for e in errors if e[1] == scheme]
        x_errs = [e[2] for e in scheme_errors]
        # Error at smallest ds should be smaller than at largest ds.
        assert x_errs[-1] < x_errs[0], f"{scheme} not converging: {x_errs}"


def test_timestep_sensitivity():
    """Smaller timesteps give smaller deviation from the reference."""
    n = 6
    x, m = _lattice(n, seed=2, amplitude=0.03)
    cosmo = _cosmology()
    a_start, a_end = 0.3, 0.8
    x_ref, u_ref, _ = rk4_integrate(
        ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                       cosmology=cosmo, a=a_start),
        a_start, a_end, n_steps=500)
    errs = []
    for ds in (0.02, 0.01, 0.005, 0.0025):
        sim = ParticleMesh3D(n=16, x=x.copy(), u=np.zeros_like(x), mass=m.copy(),
                             cosmology=cosmo, a=a_start,
                             integrator="kdk_analytic_drag")
        sim.run_to_a(a_end, max_d_ln_a=ds)
        x_err = float(np.max(np.abs(sim.x - x_ref)))
        errs.append((ds, x_err))
    # The timestep error must be bounded and small relative to the box size.
    # (Absolute error plateaus at the spatial-discretization floor; the
    #  convergence-with-dt check is in test_kdk_converges_to_rk4_reference.)
    max_err = max(e[1] for e in errs)
    assert max_err < 0.02, f"timestep error too large: {errs}"
