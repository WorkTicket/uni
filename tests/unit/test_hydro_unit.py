import numpy as np
import pytest
from universe_engine.solvers import hydro as H
from universe_engine.solvers.hydro import Euler1D, SolverFailure
from universe_engine.solvers.riemann_exact import star_state, sample


def _sod(n=64, **kw):
    x = (np.arange(n) + .5) / n
    return Euler1D(np.where(x < .5, 1, .125), 0 * x, np.where(x < .5, 1, .1), bc=("outflow", "outflow"), **kw)


# ---- exact reference solver is itself verified against published tables
@pytest.mark.parametrize("L,R,ps,us", [
    ((1, 0, 1), (.125, 0, .1), 0.30313, 0.92745),                                   # Toro Test 1
    ((1, 0, 1000), (1, 0, .01), 460.894, 19.5975),                                  # Test 3
    ((5.99924, 19.5975, 460.894), (5.99242, -6.19633, 46.0950), 1691.64, 8.68975),  # Test 5
])
def test_exact_solver_matches_published_star_states(L, R, ps, us):
    p, u = star_state(L, R, 1.4)
    assert abs(p - ps) / ps < 1e-5 and abs(u - us) / us < 1e-5


def test_exact_solver_sod_densities():
    r, _, _ = sample((1, 0, 1), (.125, 0, .1), np.array([0.3, 1.2]), 1.4)   # between contact/shock
    assert abs(r[0] - 0.42632) < 1e-4 and abs(r[1] - 0.26557) < 1e-4


def test_exact_solver_refuses_vacuum():
    with pytest.raises(ValueError):
        star_state((1, -6, .4), (1, 6, .4), 1.4)


# ---- conversions and construction
def test_cons_prim_roundtrip():
    U = H.prim_to_cons(np.array([1.2]), np.array([.3]), np.array([.7]), 1.4)
    r, u, p = H.cons_to_prim(U, 1.4)
    assert (r[0], u[0], p[0]) == pytest.approx((1.2, .3, .7), rel=1e-14)


@pytest.mark.parametrize("bad", [dict(flux="roe"), dict(limiter="vanleer"), dict(order=3),
                                 dict(cfl=0.0), dict(cfl=0.9), dict(bc=("periodic", "outflow")),
                                 dict(bc=("wall", "outflow"))])
def test_constructor_rejects_bad_settings(bad):
    n = 8
    with pytest.raises(ValueError):
        Euler1D(np.ones(n), np.zeros(n), np.ones(n), **bad)


def test_rejects_nonphysical_initial_state():
    for rho, p in [(-1, 1), (1, 0), (np.nan, 1)]:
        with pytest.raises(ValueError):
            Euler1D(np.full(8, rho), np.zeros(8), np.full(8, p))


def test_cfl_violation_rejected():
    s = _sod()
    with pytest.raises(ValueError):
        s.step(s.stable_dt() * 1.5)
    with pytest.raises(ValueError):
        s.step(-1.0)


def test_default_cfl_is_at_most_0_4():
    assert _sod().cfl <= 0.4


# ---- observer-facing API
def test_snapshot_is_a_copy_with_schema_and_units():
    s = _sod()
    snap = s.snapshot()
    snap["rho"][:] = -99
    assert np.all(s.primitives()[0] > 0)
    assert snap["schema"] == "euler1d/v1" and "dimensionless" in snap["units"]


def test_callback_receives_each_step_and_is_deterministic():
    a, b, seen = _sod(), _sod(), []
    a.run_to(0.1, callback=lambda sv: seen.append(sv.t))
    b.run_to(0.1)
    assert len(seen) == a.steps and np.all(np.diff(seen) > 0)
    assert np.array_equal(a.U, b.U)          # bitwise reproducible


def test_clipped_final_step_hits_t_end_exactly():
    s = _sod()
    s.run_to(0.123)
    assert abs(s.t - 0.123) < 1e-12


# ---- positivity-recovery paths (fault injection: the benchmarks never trigger these)
def test_step_rejection_and_retry_is_logged_and_conservative():
    s = _sod()
    real_valid, calls = s._valid, {"n": 0}

    def flaky(U):
        calls["n"] += 1
        return False if calls["n"] == 1 else real_valid(U)

    s._valid = flaky
    dt = s.stable_dt()
    s.step(dt)
    assert s.rejected_steps == 1
    assert all(v < 1e-13 for v in s.budget_residual())
    ref = _sod()                                       # retry logic == exactly two half steps
    ref.step(dt / 2); ref.step(dt / 2)
    assert np.max(np.abs(ref.U - s.U)) < 1e-14


def test_persistent_invalid_state_raises_solverfailure_not_clamping():
    s = _sod(max_retries=3)
    s._valid = lambda U: False
    with pytest.raises(SolverFailure, match="positivity"):
        s.step()
    assert np.array_equal(s.U, _sod().U)               # state untouched, nothing silently clamped


def _steep_density_solver(slope_multiplier):
    n = 32
    rho = np.where(np.arange(n) < 12, 1.0, 1e-3)
    s = Euler1D(rho, np.zeros(n), np.ones(n), bc=("outflow", "outflow"))
    s._lim = lambda a, b: slope_multiplier * a         # deliberately NON-TVD reconstruction
    return s


def test_reconstruction_fallback_is_triggered_logged_and_rescues_mild_overshoot():
    s = _steep_density_solver(2.0)
    s.step()
    assert s.fallback_interfaces > 0 and s.rejected_steps == 0
    rho_after, _, p_after = s.primitives()
    assert np.all(rho_after > 0) and np.all(p_after > 0)
    assert all(v < 1e-13 for v in s.budget_residual())


def test_severe_overshoot_fails_loudly_instead_of_clamping():
    s = _steep_density_solver(4.0)
    with pytest.raises(SolverFailure):
        s.step()
    assert s.rejected_steps >= 1                       # it did try step rejection first


# ---- boundary conditions
def test_reflective_ghost_cells_flip_momentum_only():
    n = 8
    s = Euler1D(np.arange(1., n + 1), np.full(n, .5), np.ones(n), bc=("reflective", "reflective"))
    g = s._pad(np.array(H.cons_to_prim(s.U, 1.4)))
    assert g[0, 1] == 1.0 and g[0, 0] == 2.0           # density mirrored
    assert g[1, 1] == -.5 and g[1, 0] == -.5           # velocity flipped
    assert g[2, 1] == 1.0                              # pressure mirrored
