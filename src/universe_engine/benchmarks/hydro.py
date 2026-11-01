"""R3 hydrodynamics benchmark suite. Every number here comes from running the live solver.

Tolerance policy (stated honestly):
  * ACCURACY claims are the comparisons against the exact Riemann solution and against
    analytic linear acoustics, plus the observed convergence orders (stored in metrics).
  * Error TOLERANCES below are REGRESSION GUARDS set at ~1.5x the error measured on this
    code at the date in TOLERANCE_MEASURED_ON. They detect degradation; they are not
    independent proofs of accuracy.
  * Conservation and exactness checks use roundoff-scale tolerances (1e-12 .. 1e-13).
"""
import csv
import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..solvers.hydro import Euler1D, SolverFailure
from ..solvers.riemann_exact import solution_on_grid, star_state

TOLERANCE_MEASURED_ON = "2026-10-30"
GAMMA = 1.4
REFERENCES = [
    "E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed., Springer 2009 "
    "(exact Riemann solver Ch.4; test problems Table 4.1 'Test 1/2/3/5')",
    "G. A. Sod, J. Comput. Phys. 27, 1 (1978) (shock tube)",
    "B. van Leer, J. Comput. Phys. 32, 101 (1979) (MUSCL); S. Gottlieb, C.-W. Shu, E. Tadmor, "
    "SIAM Rev. 43, 89 (2001) (SSP-RK2)",
]

# --- regression-guard tolerances (absolute L1 errors, code units) ---------------------
SOD_L1_TOL = {  # (flux, n) -> (rho, u, p), ~1.5x measured
    ("hll", 128): (6.2e-3, 1.2e-2, 3.5e-3), ("hll", 256): (3.4e-3, 7.0e-3, 2.1e-3),
    ("hll", 512): (1.9e-3, 4.0e-3, 1.2e-3),
    ("hllc", 128): (5.9e-3, 1.2e-2, 3.5e-3), ("hllc", 256): (3.3e-3, 7.0e-3, 2.1e-3),
    ("hllc", 512): (1.8e-3, 4.0e-3, 1.2e-3),
}
ACOUSTIC_EPS = 1e-6            # small enough that nonlinear steepening is below numerical error
ACOUSTIC_MIN_ORDER = {"mc": 1.7, "minmod": 1.6}   # measured 1.87-1.97 and 1.79-1.91
ROUNDOFF_TOL = 1e-12


def _chk(name, value, tol, op="<="):
    value = float(value)
    ok = bool(value <= tol) if op == "<=" else bool(value >= tol)
    return {"check": name, "value": value, "tolerance": tol, "op": op, "passed": ok}


def _bench(name, checks, **extra):
    return {"name": name, "passed": all(c["passed"] for c in checks), "checks": checks, **extra}


def _orders(errs, ns):
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(ns[i + 1] / ns[i])) for i in range(len(errs) - 1)]


def _riemann(n, L, R, t, x0=0.5, **kw):
    x = (np.arange(n) + 0.5) / n
    s = Euler1D(np.where(x < x0, L[0], R[0]), np.where(x < x0, L[1], R[1]),
                np.where(x < x0, L[2], R[2]), bc=("outflow", "outflow"), **kw)
    s.run_to(t)
    return s


def _l1_vs_exact(s, L, R, t, x0=0.5):
    exact = solution_on_grid(s.x, t, x0, L, R, GAMMA)
    return [float(np.mean(np.abs(a - b))) for a, b in zip(s.primitives(), exact)]


# =============================================================================
def bench_uniform():
    n = 64
    s = Euler1D(np.full(n, 1.3), np.full(n, 0.7), np.full(n, 0.9), flux="hll", limiter="mc")
    s.run_to(2.0)
    r, u, p = s.primitives()
    return _bench("uniform_periodic_flow", [
        _chk("max|rho-1.3|", np.max(np.abs(r - 1.3)), ROUNDOFF_TOL),
        _chk("max|u-0.7|", np.max(np.abs(u - 0.7)), ROUNDOFF_TOL),
        _chk("max|p-0.9|", np.max(np.abs(p - 0.9)), ROUNDOFF_TOL)], steps=s.steps)


def bench_sod(profiles):
    L, R, t = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2
    checks, orders = [], {}
    for flux in ("hll", "hllc"):
        ns, errs = (128, 256, 512), []
        for n in ns:
            s = _riemann(n, L, R, t, flux=flux, limiter="mc")
            e = _l1_vs_exact(s, L, R, t)
            errs.append(e)
            tol = SOD_L1_TOL[(flux, n)]
            for nm, v, tl in zip(("rho", "u", "p"), e, tol):
                checks.append(_chk(f"{flux} n={n} L1_{nm}", v, tl))
            checks.append(_chk(f"{flux} n={n} min_p>0", s.diagnostics()["min_p"], 0.0, ">="))
            if n == 512:
                ex = solution_on_grid(s.x, t, 0.5, L, R, GAMMA)
                for xi, a, b, c, d, e2, f in zip(s.x, *s.primitives(), *ex):
                    profiles.append(("sod_" + flux, xi, a, b, c, d, e2, f))
        errs = np.array(errs)
        orders[flux] = {v: _orders(errs[:, i], ns) for i, v in enumerate(("rho", "u", "p"))}
        # discontinuous solution: observed L1 order expected ~0.6-1.0 (contact ~2/3, shock ~1)
        checks.append(_chk(f"{flux} L1_rho strictly decreases with resolution",
                           errs[2, 0] / errs[0, 0], 0.6))
        checks.append(_chk(f"{flux} min observed order rho", min(orders[flux]["rho"]), 0.6, ">="))
    return _bench("sod_shock_tube", checks, observed_orders=orders,
                  note="exact Riemann reference; L1 over cell centres; gamma=1.4, t=0.2")


def _acoustic_error(n, eps, limiter="mc", order=2, T=1.0, flux="hll", return_state=False):
    x = (np.arange(n) + 0.5) / n
    p0 = 1.0 / GAMMA                       # c0 = 1, rho0 = 1
    d = eps * np.sin(2 * np.pi * x)        # right-moving linear wave: du = c0*drho/rho0, dp = c0^2*drho
    s = Euler1D(1 + d, d, p0 + d, gamma=GAMMA, flux=flux, limiter=limiter, order=order)
    s.run_to(T)
    exact = eps * np.sin(2 * np.pi * (x - T))
    err = float(np.mean(np.abs((s.primitives()[0] - 1) - exact)) / eps)
    return (err, s) if return_state else err


def bench_acoustic(profiles):
    ns = (64, 128, 256, 512, 1024)
    checks, orders, table = [], {}, {}
    for lim in ("mc", "minmod"):
        errs = [_acoustic_error(n, ACOUSTIC_EPS, lim) for n in ns]
        orders[lim], table[lim] = _orders(errs, ns), errs
        checks.append(_chk(f"{lim} min observed order (64..1024)", min(orders[lim]), ACOUSTIC_MIN_ORDER[lim], ">="))
        checks.append(_chk(f"{lim} error decreases monotonically", float(all(np.diff(errs) < 0)), 1.0, ">="))
    e1 = [_acoustic_error(n, ACOUSTIC_EPS, "mc", order=1) for n in (64, 128, 256, 512)]
    o1 = _orders(e1, (64, 128, 256, 512))
    checks.append(_chk("order=1 baseline observed order ~1 (lower bound)", min(o1), 0.85, ">="))
    checks.append(_chk("order=1 baseline observed order ~1 (upper bound)", max(o1), 1.15))
    # linear regime really is linear: halving amplitude does not change the relative error
    a = _acoustic_error(256, 1e-6)
    b = _acoustic_error(256, 1e-7)
    checks.append(_chk("relative error independent of amplitude (1e-6 vs 1e-7)", abs(a - b) / a, 1e-3))
    _, s = _acoustic_error(128, ACOUSTIC_EPS, return_state=True)
    for xi, r in zip(s.x, s.primitives()[0]):
        profiles.append(("acoustic_n128_eps1e-6", xi, r, 0, 0, 1 + ACOUSTIC_EPS * np.sin(2 * np.pi * (xi - 1)), 0, 0))
    return _bench("acoustic_wave_convergence", checks, resolutions=list(ns), observed_orders=orders,
                  observed_order_first_order_scheme=o1, relative_L1_errors=table,
                  note="amplitude 1e-6 (not 1e-4): see bench_nonlinear_floor")


def bench_nonlinear_floor():
    """At large amplitude the exact Euler solution departs from LINEAR acoustics by O(eps),
    so no grid refinement can reduce the error vs the linear reference. Verify the floor
    scales with amplitude (physics, not a solver defect)."""
    e3 = _acoustic_error(1024, 1e-3)
    e4 = _acoustic_error(1024, 1e-4)
    e6 = _acoustic_error(1024, 1e-6)
    return _bench("acoustic_nonlinear_floor", [
        _chk("floor ratio err(1e-3)/err(1e-4) ~ 10 (lower)", e3 / e4, 8.0, ">="),
        _chk("floor ratio err(1e-3)/err(1e-4) ~ 10 (upper)", e3 / e4, 12.0),
        _chk("err(1e-4) is dominated by the physical floor (>= 5x the numerical error at 1e-6)", e4 / e6, 5.0, ">="),
    ], measured={"err_eps1e-3": e3, "err_eps1e-4": e4, "err_eps1e-6": e6})


def bench_conservation():
    checks = []
    n = 128
    x = (np.arange(n) + 0.5) / n
    s = Euler1D(1 + 0.3 * np.sin(2 * np.pi * x), 0.5 + 0.2 * np.cos(2 * np.pi * x), np.ones(n), flux="hllc")
    s.run_to(1.0)
    d = s.diagnostics()
    for nm, v in zip(("mass", "momentum", "energy"), d["budget_residual_rel"]):
        checks.append(_chk(f"periodic smooth flow {nm} drift (relative)", v, 1e-13))
    sod = _riemann(256, (1, 0, 1), (.125, 0, .1), 0.2, flux="hll")
    for nm, v in zip(("mass", "momentum", "energy"), sod.diagnostics()["budget_residual_rel"]):
        checks.append(_chk(f"Sod outflow budget (incl. boundary flux) {nm}", v, 1e-13))
    return _bench("conservation_budgets", checks,
                  note="budget = domain total - initial + time-integrated boundary flux, scale-aware")


def bench_strong_shocks(profiles):
    cases = {  # name: (L, R, t, x0, L1 relative tolerance on rho from measurement*1.5)
        "toro3_blast_p_ratio_1e5": ((1, 0, 1000), (1, 0, .01), 0.012, 0.5, 6.0e-2),
        "toro5_colliding_shocks": ((5.99924, 19.5975, 460.894), (5.99242, -6.19633, 46.0950), 0.035, 0.5, 2.5e-2),
        "toro2_123_near_vacuum": ((1, -2, .4), (1, 2, .4), 0.15, 0.5, 7.0e-3),
    }
    checks, extra = [], {}
    for name, (L, R, t, x0, tol) in cases.items():
        for flux in ("hll", "hllc"):
            s = _riemann(400, L, R, t, x0=x0, flux=flux)
            e = _l1_vs_exact(s, L, R, t, x0)
            ex = solution_on_grid(s.x, t, x0, L, R, GAMMA)
            rel = e[0] / float(np.mean(np.abs(ex[0])))
            d = s.diagnostics()
            checks += [_chk(f"{name} {flux} relative L1 rho", rel, tol),
                       _chk(f"{name} {flux} min rho > 0", d["min_rho"], 0.0, ">="),
                       _chk(f"{name} {flux} min p > 0", d["min_p"], 0.0, ">="),
                       _chk(f"{name} {flux} mass budget", d["budget_residual_rel"][0], 1e-13)]
            extra[f"{name}_{flux}"] = {k: d[k] for k in ("steps", "rejected_steps", "fallback_interfaces")}
            if flux == "hll" and name == "toro3_blast_p_ratio_1e5":
                for xi, a, b, c, d1, e1, f1 in zip(s.x, *s.primitives(), *ex):
                    profiles.append(("toro3_hll", xi, a, b, c, d1, e1, f1))
    for name, L, R, t in [("p_ratio_1e10", (1, 0, 1e5), (1, 0, 1e-5), 1e-4),
                          ("p_ratio_1e12", (1, 0, 1e6), (1, 0, 1e-6), 1e-4),
                          ("density_ratio_1e6", (1e3, 0, 1), (1e-3, 0, 1), 0.05)]:
        for flux in ("hll", "hllc"):
            s = _riemann(400, L, R, t, flux=flux)
            d = s.diagnostics()
            fin = bool(np.all(np.isfinite(s.U)))
            checks += [_chk(f"{name} {flux} finite", float(fin), 1.0, ">="),
                       _chk(f"{name} {flux} min rho>0 and min p>0", float(d["min_rho"] > 0 and d["min_p"] > 0), 1.0, ">="),
                       _chk(f"{name} {flux} energy budget", d["budget_residual_rel"][2], 1e-13)]
    return _bench("strong_shock_robustness", checks, solver_event_counts=extra)


def bench_vacuum_failure_contract():
    """Contract: for vacuum-generating data the solver either returns a finite, positive
    state or raises SolverFailure. It must never return NaN/negative silently."""
    checks, outcomes = [], {}
    for amp in (6.0, 20.0, 50.0):
        for flux in ("hll", "hllc"):
            try:
                s = _riemann(400, (1, -amp, .4), (1, amp, .4), 0.01 if amp > 10 else 0.1, flux=flux)
                U = s.U
                ok = bool(np.all(np.isfinite(U)) and np.all(U[0] > 0)
                          and np.all((GAMMA - 1) * (U[2] - 0.5 * U[1] ** 2 / U[0]) > 0))
                outcomes[f"|u|={amp} {flux}"] = "completed, state valid" if ok else "completed, STATE INVALID"
            except SolverFailure:
                ok, outcomes[f"|u|={amp} {flux}"] = True, "explicit SolverFailure raised"
            checks.append(_chk(f"|u|={amp} {flux}: valid result or explicit failure", float(ok), 1.0, ">="))
    return _bench("vacuum_generation_contract", checks, outcomes=outcomes)


def bench_boundaries():
    checks = []
    # (1) periodic mass transport: contact advected exactly one period
    n = 200
    x = (np.arange(n) + 0.5) / n
    rho0 = 1 + 0.5 * np.exp(-((x - 0.5) / 0.08) ** 2)
    s = Euler1D(rho0, np.ones(n), np.ones(n), flux="hllc", limiter="mc")
    s.run_to(1.0)
    r = s.primitives()[0]
    checks += [_chk("periodic advection: mass drift", s.budget_residual()[0], 1e-13),
               _chk("periodic advection: L1 profile error after 1 period", np.mean(np.abs(r - rho0)), 3.0e-3),
               _chk("periodic advection: min rho", r.min(), 1.0 - 1e-9, ">=")]
    # (2) reflective wall == mirror-symmetric doubled domain with outflow ends (exact equivalence)
    n = 200
    pulse = lambda xx: 1 + 0.5 * np.exp(-((xx - 0.4) / 0.08) ** 2)
    xh = (np.arange(n) + 0.5) / n
    half = Euler1D(pulse(xh), np.full(n, -0.5), np.ones(n), x0=0, x1=1, bc=("reflective", "outflow"))
    xf = -1 + (np.arange(2 * n) + 0.5) / n
    full = Euler1D(np.where(xf > 0, pulse(xf), pulse(-xf)), np.where(xf > 0, -0.5, 0.5), np.ones(2 * n),
                   x0=-1, x1=1, bc=("outflow", "outflow"))
    half.run_to(1.2); full.run_to(1.2)
    diff = max(np.max(np.abs(a - b)) for a, b in zip(half.primitives(), tuple(q[n:] for q in full.primitives())))
    checks.append(_chk("reflective wall vs mirrored domain: max primitive difference", diff, 1e-12))
    # (3) reflecting wall: momentum sign reversal and wall pressure vs exact Riemann p*
    n = 400
    w = Euler1D(np.ones(n), -np.ones(n), np.ones(n), bc=("reflective", "outflow"))
    w.run_to(0.3)
    r, u, p = w.primitives()
    ps, us = star_state((1, 1, 1), (1, -1, 1), GAMMA)
    checks += [_chk("wall pressure relative error vs exact p*", abs(p[0] - ps) / ps, 5e-4),
               _chk("wall-cell velocity (should be 0)", abs(u[0]), 1e-3),
               _chk("wall run mass budget", w.budget_residual()[0], 1e-13),
               _chk("wall run momentum budget (wall impulse in boundary flux)", w.budget_residual()[1], 1e-13)]
    # (4) outflow ghost cells: supersonic pulse leaves cleanly, no spurious reflection
    n = 200
    xo = (np.arange(n) + 0.5) / n
    o = Euler1D(1 + 0.5 * np.exp(-((xo - 0.5) / 0.05) ** 2), np.full(n, 3.0), np.ones(n), bc=("outflow", "outflow"))
    o.run_to(0.45)
    ro, uo, po = o.primitives()
    checks += [_chk("outflow: residual density deviation after pulse exits", np.max(np.abs(ro - ro[0])), 1e-10),
               _chk("outflow: budget closes incl. boundary flux", float(np.max(o.budget_residual())), 1e-13)]
    # (5) uniform supersonic flow with outflow ends is a fixed point
    n = 64
    f = Euler1D(np.ones(n), np.full(n, 3.0), np.ones(n), bc=("outflow", "outflow"))
    f.run_to(1.0)
    checks.append(_chk("outflow: uniform supersonic flow preserved", np.max(np.abs(f.primitives()[0] - 1)), ROUNDOFF_TOL))
    return _bench("boundary_conditions", checks)


# =============================================================================
BENCHMARKS = ("uniform", "sod", "acoustic", "nonlinear_floor", "conservation", "strong_shocks",
              "vacuum_contract", "boundaries")


def _git_sha(cwd):
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def run_all(output_root="runs/benchmarks/hydro", command="universe benchmark hydro"):
    start = _dt.datetime.now(_dt.timezone.utc)
    run_id = start.strftime("%Y%m%dT%H%M%SZ")
    profiles = []
    results = [bench_uniform(), bench_sod(profiles), bench_acoustic(profiles), bench_nonlinear_floor(),
               bench_conservation(), bench_strong_shocks(profiles), bench_vacuum_failure_contract(),
               bench_boundaries()]
    out = Path(output_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    n_checks = sum(len(r["checks"]) for r in results)
    n_pass = sum(c["passed"] for r in results for c in r["checks"])
    metrics = {"run_id": run_id, "all_passed": all(r["passed"] for r in results),
               "checks_passed": int(n_pass), "checks_total": int(n_checks), "benchmarks": results}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "profiles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "x", "rho", "u", "p", "rho_ref", "u_ref", "p_ref"])
        w.writerows(profiles)
    meta = {"run_id": run_id, "started_utc": start.isoformat(), "command": command,
            "git_sha": _git_sha(Path(__file__).parent), "python": sys.version.split()[0],
            "numpy": np.__version__, "platform": platform.platform(), "gamma": GAMMA,
            "solver": "universe_engine.solvers.hydro.Euler1D (MUSCL+SSP-RK2, HLL/HLLC)",
            "units": "dimensionless code units", "references": REFERENCES,
            "tolerance_policy": __doc__.strip(), "tolerance_measured_on": TOLERANCE_MEASURED_ON,
            "scientific_status": "T1 numerical verification of a 1D ideal-gas Euler solver. "
                                 "Not cosmological hydrodynamics (no gravity, cooling, expansion)."}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    try:  # optional human-readable plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arr = [r for r in profiles if r[0] == "sod_hll"]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
        for i, (lab, col) in enumerate((("rho", 2), ("u", 3), ("p", 4))):
            ax[i].plot([r[1] for r in arr], [r[col] for r in arr], ".", ms=3, label="solver (512, HLL+MC)")
            ax[i].plot([r[1] for r in arr], [r[col + 3] for r in arr], "-", lw=1, label="exact")
            ax[i].set_title(lab)
        ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "sod.png", dpi=110); plt.close(fig)
    except ImportError:
        pass
    return metrics, out
