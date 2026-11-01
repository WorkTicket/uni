import numpy as np
import pytest
from universe_engine.solvers import gravity as GR
from universe_engine.solvers.gravity import SelfGravityEuler1D, poisson_1d_periodic
from universe_engine.solvers.hydro import Euler1D


def _gas(n=64, L=2.0, eps=0.2, **kw):
    x = (np.arange(n) + .5) * L / n
    d = eps * np.sin(np.pi * x)
    return SelfGravityEuler1D(Euler1D(1 + d, 0 * x, (1 / 1.4) * (1 + 1.4 * d), x0=0, x1=L, gamma=1.4), **kw)


def test_poisson_single_mode_exact():
    n, k, A, G = 32, 2 * np.pi * 2, 0.3, 0.7
    x = (np.arange(n) + .5) / n
    phi, g = poisson_1d_periodic(1 + A * np.cos(k * x), 1 / n, G)
    assert np.max(np.abs(g + 4 * np.pi * G * A * np.sin(k * x) / k)) < 1e-13
    assert abs(phi.mean()) < 1e-15                         # k=0 mode dropped


def test_gravity_is_attractive_toward_density_peak():
    n = 64
    x = (np.arange(n) + .5) / n
    _, g = poisson_1d_periodic(1 + 0.5 * np.exp(-(x - .5) ** 2 / .01), 1 / n, 1.0)
    assert g[np.argmin(np.abs(x - .4))] > 0 and g[np.argmin(np.abs(x - .6))] < 0


def test_poisson_linear_in_G_and_rejects_nothing_silently():
    n = 32
    rho = 1 + .3 * np.cos(2 * np.pi * (np.arange(n) + .5) / n)
    assert np.allclose(poisson_1d_periodic(rho, 1 / n, 2.0)[1], 2 * poisson_1d_periodic(rho, 1 / n, 1.0)[1])


def test_odd_and_even_grid_sizes_give_real_finite_forces():
    for n in (31, 32, 33, 64):
        x = (np.arange(n) + .5) / n
        phi, g = poisson_1d_periodic(1 + .2 * np.sin(2 * np.pi * x), 1 / n, 1.0)
        assert np.all(np.isfinite(g)) and g.dtype == np.float64


def test_kick_conserves_momentum_and_mass_exactly_and_only_changes_kinetic_energy():
    s = _gas()
    h = s.hydro
    mass, mom = h.totals()[0], h.totals()[1]
    rho, u, p = h.primitives()
    eint = p / (h.gamma - 1)
    s._kick(0.05)
    rho2, u2, p2 = h.primitives()
    assert abs(h.totals()[0] - mass) < 1e-14 and abs(h.totals()[1] - mom) < 1e-14
    assert np.max(np.abs(p2 / (h.gamma - 1) - eint)) < 1e-13      # internal energy untouched
    assert np.any(np.abs(u2) > 0)


def test_requires_periodic_and_valid_parameters():
    n = 8
    wall = Euler1D(np.ones(n), np.zeros(n), np.ones(n), bc=("reflective", "outflow"))
    with pytest.raises(ValueError):
        SelfGravityEuler1D(wall)
    with pytest.raises(ValueError):
        SelfGravityEuler1D(Euler1D(np.ones(n), np.zeros(n), np.ones(n)), G=-1.0)


def test_dt_guard_and_gravity_limited_step():
    s = _gas()
    with pytest.raises(ValueError):
        s.step(s.stable_dt() * 3)
    h = Euler1D(np.ones(64), np.zeros(64), np.full(64, 1 / 1.4), gamma=1.4)
    g = SelfGravityEuler1D(h, G=200.0, kick_cfl=0.1)
    assert g.stable_dt() == pytest.approx(0.1 / np.sqrt(4 * np.pi * 200), rel=1e-12)
    assert g.stable_dt() < h.stable_dt()           # gravity, not the hydro CFL, is binding here


def test_deterministic_and_checkpoint_replay_bitwise_at_step_boundary():
    a, b = _gas(), _gas()
    saved = []
    a.run_to(0.6, callback=lambda q: saved.append(q.checkpoint()) if q.hydro.steps == 10 else None)
    b.run_to(0.6)
    assert np.array_equal(a.hydro.U, b.hydro.U)
    c = _gas(); c.restore(saved[0]); c.run_to(0.6)
    assert np.array_equal(a.hydro.U, c.hydro.U)


def test_checkpoint_is_a_copy():
    s = _gas(); cp = s.checkpoint(); s.run_to(0.1)
    assert cp["hydro"]["t"] == 0.0 and not np.array_equal(cp["hydro"]["U"], s.hydro.U)


def test_total_energy_includes_negative_potential_energy():
    s = _gas(eps=0.3)
    assert s.potential_energy() < 0
