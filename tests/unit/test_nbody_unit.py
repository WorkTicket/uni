import numpy as np
import pytest
from universe_engine.solvers import nbody as NB
from universe_engine.solvers.nbody import GravitySystem1D, cic_deposit, cic_interpolate
from universe_engine.solvers.hydro import Euler1D


def test_cic_single_particle_splits_between_two_cells():
    rho = cic_deposit(np.array([0.25]), 1.0, 0.0, 0.25, 4)       # exactly between centres 0.125 and 0.375
    assert rho == pytest.approx([2.0, 2.0, 0.0, 0.0])            # mass 0.5 each / dx 0.25


def test_cic_particle_on_a_cell_centre_goes_to_one_cell():
    rho = cic_deposit(np.array([0.375]), 1.0, 0.0, 0.25, 4)
    assert rho == pytest.approx([0.0, 4.0, 0.0, 0.0])


def test_cic_wraps_periodically():
    a = cic_deposit(np.array([-0.05]), 1.0, 0.0, 0.25, 4)
    b = cic_deposit(np.array([0.95]), 1.0, 0.0, 0.25, 4)
    assert np.allclose(a, b, atol=1e-13)


def test_cic_interpolation_reproduces_cell_values_at_centres():
    f = np.array([1., 2., 5., 3.])
    assert cic_interpolate(f, np.array([0.125, 0.375, 0.625, 0.875]), 0.0, 0.25) == pytest.approx(f)


def test_particle_only_system_requires_grid_and_valid_arrays():
    with pytest.raises(ValueError):
        GravitySystem1D(x=np.zeros(4), mass=1.0)                  # no n
    with pytest.raises(ValueError):
        GravitySystem1D(x=np.zeros(4), mass=-1.0, n=8)
    with pytest.raises(ValueError):
        GravitySystem1D()                                          # nothing to simulate
    with pytest.raises(ValueError):
        GravitySystem1D(hydro=Euler1D(np.ones(8), np.zeros(8), np.ones(8), bc=("outflow", "outflow")))


def test_positions_are_wrapped_into_the_box():
    s = GravitySystem1D(x=np.array([1.7, -0.2]), mass=1.0, n=16, box=(0.0, 1.0))
    assert np.all((s.x >= 0) & (s.x < 1))


def test_uniform_lattice_feels_no_force_and_does_not_move():
    N = 64
    s = GravitySystem1D(x=(np.arange(N) + .5) / N, mass=1.0 / N, n=16, G=1.0)
    s.run_to(0.5)
    assert np.max(np.abs(s.v)) < 1e-13


def test_attraction_between_sheets():
    s = GravitySystem1D(x=np.array([0.4, 0.6]), mass=0.1, n=64, G=1.0)
    s.run_to(0.2)
    assert s.v[0] > 0 and s.v[1] < 0                              # they accelerate toward each other


def test_total_momentum_exactly_zero_for_particles_only():
    rng = np.random.default_rng(3)
    s = GravitySystem1D(x=rng.random(128), mass=1.0 / 128, n=32, G=1.0)
    s.run_to(0.3)
    assert abs(s.momentum()[1]) < 1e-13


def test_dt_guard_rejects_large_steps():
    s = GravitySystem1D(x=(np.arange(32) + .5) / 32 + 0.01, mass=1.0 / 32, n=16)
    with pytest.raises(ValueError):
        s.step(s.stable_dt() * 3)


def test_snapshot_schema_contents_and_copy_semantics():
    N = 32
    s = GravitySystem1D(x=(np.arange(N) + .5) / N + .01, mass=1.0 / N, n=16)
    snap = s.snapshot()
    assert snap["schema"] == "nbody1d/v1" and "dimensionless" in snap["units"]
    assert snap["particle_x"].shape == (N,) and snap["density_total"].shape == (16,)
    snap["particle_x"][:] = 99
    assert np.all(s.x < 1)


def test_deterministic_and_bitwise_checkpoint_replay_with_gas_and_particles():
    L, n, N = 2.0, 32, 128
    x = (np.arange(n) + .5) * L / n
    def mk():
        d = .2 * np.sin(np.pi * x); q = (np.arange(N) + .5) * L / N
        return GravitySystem1D(hydro=Euler1D(.5 * (1 + d), 0 * x, (.5 / 1.4) * (1 + 1.4 * d), x0=0, x1=L, gamma=1.4),
                               x=q + .02 * np.sin(np.pi * q), v=np.zeros(N), mass=.5 * L / N)
    a, b, saved = mk(), mk(), []
    a.run_to(0.5, callback=lambda s: saved.append(s.checkpoint()) if s.steps == 6 else None)
    b.run_to(0.5)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.hydro.U, b.hydro.U)
    c = mk(); c.restore(saved[0]); c.run_to(0.5)
    assert np.array_equal(a.x, c.x) and np.array_equal(a.v, c.v) and np.array_equal(a.hydro.U, c.hydro.U)
