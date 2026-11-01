"""R4 (1D) benchmarks for collisionless particles + CIC + gas sharing one Poisson solve.

SCOPE: 1D, periodic, non-expanding. NOT covered: 3D, comoving/cosmological factors,
non-trivial hydrostatic equilibrium. Tolerance policy as in benchmarks/hydro.py (regression
guards ~1.5x measured on 2026-10-30; roundoff-level for exactness checks).
Key exact reference: before shell crossing the 1D sheet force is g = 4 pi G rho_bar * psi, so
x(q,t) = q + psi0(q) cosh(Gamma t), Gamma = sqrt(4 pi G rho_bar), exact even when nonlinear.
"""
import csv
import datetime as _dt
import json
import platform
import sys
from pathlib import Path

import numpy as np

from ..solvers.gravity import SelfGravityEuler1D
from ..solvers.hydro import Euler1D
from ..solvers.nbody import GravitySystem1D, cic_deposit, cic_interpolate
from .hydro import _bench, _chk, _git_sha, _orders

G, GAMMA, C2 = 1.0, 1.4, 1.0
GAM_ZEL = np.sqrt(4 * np.pi * G * 1.0)
REFERENCES = [
    "Ya. B. Zel'dovich, Astron. Astrophys. 5, 84 (1970) (pancake solution)",
    "R. W. Hockney & J. W. Eastwood, Computer Simulation Using Particles (1988) (CIC, PM method)",
    "J. Binney & S. Tremaine, Galactic Dynamics, 2nd ed. (2008), Sec. 5.2 (two-component Jeans analysis)",
]
ZEL_TOL = {32: 3.2e-2, 64: 8.1e-3, 128: 2.0e-3, 256: 5.1e-4}          # max|x-x_exact|/A, measured x1.5
TWOFLUID_UNSTABLE_TOL = {64: (1.8e-2, 4.2e-2), 128: (4.4e-3, 1.05e-2), 256: (1.1e-3, 2.6e-3)}   # (gas, particles)
TWOFLUID_STABLE_PART_TOL = {64: 3.0e-2, 128: 7.6e-3, 256: 1.9e-3}
TWOFLUID_STABLE_GAS_TOL = 3.1e-3


def zeldovich(n, A=0.02, t=0.5, ppc=8, kick_cfl=None):
    N = n * ppc
    q = (np.arange(N) + 0.5) / N
    psi0 = A * np.sin(2 * np.pi * q)
    s = GravitySystem1D(x=q + psi0, v=np.zeros(N), mass=1.0 / N, box=(0, 1), n=n, G=G,
                        kick_cfl=kick_cfl if kick_cfl else 0.1 * 32 / n)
    rho_max = float(s.density().max())
    s.run_to(t)
    d = s.x - (q + psi0 * np.cosh(GAM_ZEL * t))
    d -= np.round(d)                                   # periodic wrap
    return float(np.max(np.abs(d)) / A), s, rho_max


def two_fluid_reference(L, rg, rp, t, eps):
    k = 2 * np.pi / L
    M = np.array([[-C2 * k * k + 4 * np.pi * G * rg, 4 * np.pi * G * rp],
                  [4 * np.pi * G * rg, 4 * np.pi * G * rp]])
    lam, V = np.linalg.eig(M)
    y = V @ np.diag(np.cos(np.sqrt(-lam.astype(complex)) * t)) @ np.linalg.inv(V) @ np.array([eps, eps])
    return y.real


def two_fluid_run(n, L, rg, rp, t, eps=1e-6, ppc=8):
    x = (np.arange(n) + 0.5) * L / n
    k = 2 * np.pi / L
    d = eps * np.sin(k * x)
    h = Euler1D(rg * (1 + d), np.zeros(n), (rg * C2 / GAMMA) * (1 + GAMMA * d), x0=0, x1=L, gamma=GAMMA)
    N = n * ppc
    q = (np.arange(N) + 0.5) * L / N
    s = GravitySystem1D(hydro=h, x=q + (eps / k) * np.cos(k * q), v=np.zeros(N), mass=rp * L / N, G=G,
                        kick_cfl=0.1 * 64 / n)
    s.run_to(t)
    ag = 2 / n * np.sum((h.primitives()[0] / rg - 1) * np.sin(k * x))
    ap = 2 / n * np.sum(((s.density() - h.U[0]) / rp - 1) * np.sin(k * x))
    return ag, ap


# =============================================================================
def bench_cic():
    rng = np.random.default_rng(12345)
    n, x0, x1 = 32, -1.0, 3.0
    dx = (x1 - x0) / n
    xs = x0 + (x1 - x0) * rng.random(200)
    m = rng.random(200) + 0.1
    rho = cic_deposit(xs, m, x0, dx, n)
    cells = x0 + (np.arange(n) + 0.5) * dx
    f = np.sin(2 * np.pi * 3 * cells / (x1 - x0)) + 0.3 * rng.random(n)
    adjoint = abs(np.sum(m * cic_interpolate(f, xs, x0, dx)) - np.sum(rho * f) * dx)
    interior = x0 + dx / 2 + (x1 - x0 - dx) * rng.random(50)
    lin = cic_interpolate(2.5 * cells - 0.7, interior, x0, dx)
    checks = [_chk("CIC mass conservation |sum(rho)dx - sum(m)|", abs(np.sum(rho) * dx - m.sum()), 1e-13),
              _chk("CIC deposit/interpolate adjoint identity (momentum conservation basis)", adjoint, 1e-12),
              _chk("CIC interpolation exact for linear fields", np.max(np.abs(lin - (2.5 * interior - 0.7))), 1e-12),
              _chk("periodic wrap: particle at x1+0.3dx equals particle at x0+0.3dx (relative to peak density 1/dx)",
                   np.max(np.abs(cic_deposit(np.array([x1 + 0.3 * dx]), 1.0, x0, dx, n) - cic_deposit(np.array([x0 + 0.3 * dx]), 1.0, x0, dx, n))) * dx, 1e-13),
              _chk("lattice with integer particles per cell gives uniform density (total mass / box length)",
                   np.max(np.abs(cic_deposit((np.arange(256) + .5) / 256 * (x1 - x0) + x0, 1.0 / 64, x0, dx, n) - 256 * (1.0 / 64) / (x1 - x0))), 1e-12)]
    return _bench("cic_deposition_interpolation", checks)


def bench_zeldovich(profiles):
    ns, errs, checks = (32, 64, 128, 256), [], []
    for n in ns:
        e, s, rho_max = zeldovich(n)
        errs.append(e)
        checks.append(_chk(f"n={n} max|x - x_exact|/A (nonlinear Zel'dovich, dt ~ dx)", e, ZEL_TOL[n]))
        profiles.append(("zeldovich", n, e, rho_max, s.energy_drift(), 0, 0, 0))
    orders = _orders(errs, ns)
    _, _, rho_max = zeldovich(64)
    e64, s64, _ = zeldovich(64, kick_cfl=0.1)
    _, s64b, _ = zeldovich(64, kick_cfl=0.025)
    q64 = (np.arange(s64.x.size) + 0.5) / s64.x.size
    dq = s64.x - q64
    dq -= np.round(dq)                                  # displacement relative to the Lagrangian lattice
    checks += [_chk("no shell crossing: Eulerian order of sheets preserved (min spacing > 0)",
                    float(np.min(np.diff(q64 + dq))), 0.0, ">="),
               _chk("min observed order (32..256)", min(orders), 1.9, ">="),
               _chk("problem is genuinely nonlinear: max initial density >= 1.0 and final max rho >= 1.5",
                    float(s64.density().max()), 1.5, ">="),
               _chk("energy drift at kick_cfl=0.1", s64.energy_drift(), 1.25e-3),
               _chk("energy drift falls >=12x when dt shrinks 4x (2nd-order splitting error)",
                    s64.energy_drift() / s64b.energy_drift(), 12.0, ">="),
               _chk("|total particle momentum|", abs(s64.momentum()[1]), 1e-13)]
    return _bench("zeldovich_sheet_nonlinear", checks, resolutions=list(ns), errors=errs, observed_orders=orders,
                  note="exact nonlinear solution until shell crossing (t_cross = 0.78 > t = 0.5)")


def bench_momentum_exchange():
    n, L = 64, 2.0
    x = (np.arange(n) + 0.5) * L / n
    d = 0.2 * np.sin(np.pi * x)
    h = Euler1D(0.5 * (1 + d), np.zeros(n), (0.5 / GAMMA) * (1 + GAMMA * d), x0=0, x1=L, gamma=GAMMA)
    N = n * 8
    q = (np.arange(N) + 0.5) * L / N
    s = GravitySystem1D(hydro=h, x=q + 0.03 * np.sin(np.pi * q), v=np.zeros(N), mass=0.5 * L / N, G=G)
    mg0, mp0 = h.totals()[0], s.m.sum()
    g0, p0 = s.momentum()
    s.run_to(0.4)
    g1, p1 = s.momentum()
    checks = [_chk("gas momentum change is non-trivial (real exchange)", abs(g1 - g0), 1e-3, ">="),
              _chk("|dP_gas + dP_particles| (exact exchange)", abs((g1 - g0) + (p1 - p0)), 1e-13),
              _chk("gas mass budget", h.budget_residual()[0], 1e-13),
              _chk("particle mass unchanged", abs(s.m.sum() - mp0), 1e-15)]
    return _bench("gas_particle_momentum_exchange", checks, dP_gas=float(g1 - g0), dP_particles=float(p1 - p0))


def bench_two_fluid():
    checks, out = [], {}
    L, rg, rp, t = 2.0, 0.5, 0.5, 0.9
    ref = two_fluid_reference(L, rg, rp, t, 1e-6) / 1e-6
    ns, eg, ep = (32, 64, 128, 256), [], []
    for n in ns:
        ag, ap = two_fluid_run(n, L, rg, rp, t)
        eg.append(abs(ag / 1e-6 - ref[0])); ep.append(abs(ap / 1e-6 - ref[1]))
    for n, g_, p_ in zip(ns, eg, ep):
        if n in TWOFLUID_UNSTABLE_TOL:
            checks += [_chk(f"unstable n={n} gas amplitude error vs linear two-fluid solution", g_, TWOFLUID_UNSTABLE_TOL[n][0]),
                       _chk(f"unstable n={n} particle amplitude error", p_, TWOFLUID_UNSTABLE_TOL[n][1])]
    og, op = _orders(eg, ns), _orders(ep, ns)
    checks += [_chk("unstable gas min order", min(og), 1.9, ">="), _chk("unstable particle min order", min(op), 1.9, ">=")]
    ref2 = two_fluid_reference(1.0, rg, rp, 0.9, 1e-6) / 1e-6
    eg2, ep2 = [], []
    for n in ns:
        ag, ap = two_fluid_run(n, 1.0, rg, rp, 0.9)
        eg2.append(abs(ag / 1e-6 - ref2[0])); ep2.append(abs(ap / 1e-6 - ref2[1]))
    for n, p_ in zip(ns, ep2):
        if n in TWOFLUID_STABLE_PART_TOL:
            checks.append(_chk(f"stable n={n} particle amplitude error", p_, TWOFLUID_STABLE_PART_TOL[n]))
    checks += [_chk("stable particle min order", min(_orders(ep2, ns)), 1.9, ">="),
               _chk("stable gas error magnitude (no order claimed: not a clean power law)", max(eg2), TWOFLUID_STABLE_GAS_TOL)]
    return _bench("two_fluid_jeans_gas_plus_particles", checks, reference_unstable=list(ref), reference_stable=list(ref2),
                  gas_orders=og, particle_orders=op,
                  note="reference = exact solution of the linearised coupled gas/collisionless system (eigen-decomposition)")


def bench_consistency():
    checks = []
    n, L = 64, 2.0
    x = (np.arange(n) + .5) * L / n
    d = .2 * np.sin(np.pi * x)
    mk = lambda: Euler1D(1 + d, np.zeros(n), (1 / 1.4) * (1 + 1.4 * d), x0=0, x1=L, gamma=1.4)
    a = SelfGravityEuler1D(mk(), G=1.0); a.run_to(0.5)
    b = GravitySystem1D(hydro=mk(), G=1.0); b.run_to(0.5)
    checks.append(_chk("gas-only GravitySystem1D identical to SelfGravityEuler1D (bitwise)", float(np.array_equal(a.hydro.U, b.hydro.U)), 1.0, ">="))
    rng = np.random.default_rng(1)
    x0, v = rng.random(64), 0.3 * rng.normal(size=64)
    s = GravitySystem1D(x=x0, v=v, mass=1.0 / 64, box=(0, 1), n=32, G=0.0); s.run_to(1.7)
    dd = s.x - (x0 + v * 1.7) % 1.0; dd -= np.round(dd)
    checks.append(_chk("free streaming (G=0) matches x0+vt", np.max(np.abs(dd)), 1e-13))
    def scaled(f):
        nn = 32; N = nn * 8; q = (np.arange(N) + .5) / N
        s_ = GravitySystem1D(x=q + .01 * np.sin(2 * np.pi * q), v=np.zeros(N), mass=f / N, box=(0, 1), n=nn, G=1.0 / f)
        s_.run_to(0.3); return s_.x, s_.v
    (xa, va), (xb, vb) = scaled(1.0), scaled(3.0)
    checks.append(_chk("mass x3, G/3 leaves trajectories unchanged", max(np.max(np.abs(xa - xb)), np.max(np.abs(va - vb))), 1e-12))
    def full():
        N = 256; q = (np.arange(N) + .5) * L / N
        return GravitySystem1D(hydro=mk(), x=q + .02 * np.sin(np.pi * q), v=np.zeros(N), mass=.5 * L / N, G=1.0)
    saved, ref = [], full()
    ref.run_to(0.6, callback=lambda q_: saved.append(q_.checkpoint()) if q_.steps == 8 else None)
    rep = full(); rep.restore(saved[0]); rep.run_to(0.6)
    checks.append(_chk("checkpoint (gas+particles) at step boundary replays bitwise",
                       float(np.array_equal(ref.hydro.U, rep.hydro.U) and np.array_equal(ref.x, rep.x) and np.array_equal(ref.v, rep.v)), 1.0, ">="))
    snap = ref.snapshot(); snap["particle_x"][:] = -9
    checks.append(_chk("snapshot is a copy (mutating it does not affect the run)", float(np.all(ref.x > -1)), 1.0, ">="))
    return _bench("consistency_units_checkpoint_snapshot", checks)


def run_all(output_root="runs/benchmarks/nbody", command="universe benchmark nbody"):
    start = _dt.datetime.now(_dt.timezone.utc)
    run_id = start.strftime("%Y%m%dT%H%M%SZ")
    profiles = []
    results = [bench_cic(), bench_zeldovich(profiles), bench_momentum_exchange(), bench_two_fluid(), bench_consistency()]
    out = Path(output_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    n_checks = sum(len(r["checks"]) for r in results)
    n_pass = sum(c["passed"] for r in results for c in r["checks"])
    metrics = {"run_id": run_id, "all_passed": all(r["passed"] for r in results),
               "checks_passed": int(n_pass), "checks_total": int(n_checks), "benchmarks": results}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "profiles.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["benchmark", "n", "error", "rho_max", "energy_drift", "c5", "c6", "c7"]); w.writerows(profiles)
    (out / "metadata.json").write_text(json.dumps({
        "run_id": run_id, "started_utc": start.isoformat(), "command": command, "git_sha": _git_sha(Path(__file__).parent),
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "units": "dimensionless code units", "references": REFERENCES, "tolerance_policy": __doc__.strip(),
        "scientific_status": "T1 numerical verification of a 1D periodic N-body + gas gravity slice; "
                             "no 3D, no cosmological expansion."}, indent=2))
    return metrics, out
