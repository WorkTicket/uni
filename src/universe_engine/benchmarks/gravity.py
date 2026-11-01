"""R4 (1D slice) benchmark suite: periodic Poisson solver and gravity-hydro coupling.

SCOPE: 1D gas self-gravity only. NOT covered: collisionless particles, CIC deposition, 3D,
comoving/cosmological factors, free-fall and non-trivial hydrostatic-equilibrium tests.
Tolerance policy as in benchmarks/hydro.py: accuracy = comparison to analytic solutions and
observed orders; error tolerances = regression guards ~1.5x measured (2026-10-30);
conservation/exactness = roundoff scale.
"""
import csv
import datetime as _dt
import json
import platform
import sys
from pathlib import Path

import numpy as np

from ..solvers.gravity import SelfGravityEuler1D, poisson_1d_periodic
from ..solvers.hydro import Euler1D
from .hydro import _bench, _chk, _git_sha, _orders

GAMMA, RHO0, C2 = 1.4, 1.0, 1.0           # c_s = 1, rho0 = 1, G = 1 (code units)
P0 = RHO0 * C2 / GAMMA
REFERENCES = [
    "J. H. Jeans, Phil. Trans. R. Soc. A 199, 1 (1902) (dispersion relation w^2 = c^2 k^2 - 4 pi G rho0)",
    "J. Binney & S. Tremaine, Galactic Dynamics, 2nd ed. (2008), Sec. 5.2 (Jeans instability, Jeans swindle)",
    "R. Strang, SIAM J. Numer. Anal. 5, 506 (1968) (operator splitting)",
]
JEANS_UNSTABLE_TOL = {64: 5.9e-3, 128: 1.4e-3, 256: 3.3e-4}   # relative amplitude error, measured x1.5
JEANS_STABLE_TOL = {64: 9.2e-3, 128: 2.3e-3, 256: 5.8e-4, 512: 1.45e-4}   # |err|/eps, measured x1.5
ENERGY_DRIFT_TOL = 5.0e-6                                      # measured max 3.6e-6


def _perturbed(n, L, eps, G=1.0, cfl=0.4, scale=1.0):
    """Adiabatic (sound-mode) perturbation: du=0, dp = c^2 d(rho). Isobaric data would NOT be the Jeans mode."""
    x = (np.arange(n) + 0.5) * L / n
    d = eps * np.sin(2 * np.pi * x / L)
    h = Euler1D(scale * RHO0 * (1 + d), np.zeros(n), scale * P0 * (1 + GAMMA * d), x0=0.0, x1=L,
                gamma=GAMMA, cfl=cfl)
    return x, SelfGravityEuler1D(h, G=G)


def _amplitude(s, x, L):
    rho = s.hydro.primitives()[0]
    return 2.0 / x.size * float(np.sum((rho / RHO0 - 1) * np.sin(2 * np.pi * x / L)))


def _g_gaussian_exact(x, sig=0.05, x0=0.5, G=1.0, modes=400):
    """Exact Fourier-series acceleration of rho = 1 + exp(-(x-x0)^2/2sig^2) in a unit periodic box
    (independent of the FFT grid solution)."""
    g = np.zeros_like(x)
    for m in range(1, modes):
        k = 2 * np.pi * m
        c = sig * np.sqrt(2 * np.pi) * np.exp(-0.5 * sig**2 * k**2)
        g += -8 * np.pi * G * c * np.sin(k * (x - x0)) / k
    return g


def bench_force():
    checks = []
    n, G, A, k = 64, 1.3, 0.2, 2 * np.pi * 3
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    phi, g = poisson_1d_periodic(1 + A * np.cos(k * x), dx, G)
    checks.append(_chk("single-mode g vs exact -4 pi G A sin(kx)/k", np.max(np.abs(g + 4 * np.pi * G * A * np.sin(k * x) / k)), 1e-13))
    checks.append(_chk("single-mode phi vs exact -4 pi G A cos(kx)/k^2", np.max(np.abs(phi + 4 * np.pi * G * A * np.cos(k * x) / k**2)), 1e-13))
    errs = {}
    for n in (16, 32, 48, 64):
        dx = 1.0 / n
        x = (np.arange(n) + 0.5) * dx
        gn = poisson_1d_periodic(1 + np.exp(-(x - 0.5) ** 2 / (2 * 0.05**2)), dx, 1.0)[1]
        errs[n] = float(np.max(np.abs(gn - _g_gaussian_exact(x))))
    checks += [_chk("Gaussian force error n=32 (spectral convergence)", errs[32], 2.3e-7),
               _chk("Gaussian force error n=48", errs[48], 1e-12),
               _chk("Gaussian force error n=64 (roundoff)", errs[64], 1e-13),
               _chk("Gaussian force error decreases 16>32>48", float(errs[16] > errs[32] > errs[48]), 1.0, ">=")]
    n = 128
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    rho = 1 + np.exp(-(x - 0.5) ** 2 / (2 * 0.07**2))
    _, g = poisson_1d_periodic(rho, dx, 1.0)
    checks += [_chk("force symmetry: max|g(x)+g(L-x)| for symmetric rho", np.max(np.abs(g + g[::-1])), 1e-14),
               _chk("zero net force: |sum(rho*g)*dx|", abs(np.sum(rho * g) * dx), 1e-14),
               _chk("uniform density gives g = 0", np.max(np.abs(poisson_1d_periodic(np.full(n, 3.0), dx, 1.0)[1])), 1e-15)]
    return _bench("poisson_force", checks, gaussian_errors=errs)


def bench_jeans_unstable(profiles):
    L, t = 2.0, 1.2                       # k = pi < k_J = sqrt(4 pi G rho0/c^2) = 3.545
    k = 2 * np.pi / L
    gam = np.sqrt(4 * np.pi - C2 * k * k)
    checks, errs, amps = [], [], {}
    ns = (32, 64, 128, 256)
    for n in ns:
        x, s = _perturbed(n, L, 1e-6)
        s.run_to(t)
        ex = np.cosh(gam * t)
        err = abs(_amplitude(s, x, L) / 1e-6 - ex) / ex
        errs.append(err)
        amps[n] = _amplitude(s, x, L) / 1e-6
        if n in JEANS_UNSTABLE_TOL:
            checks.append(_chk(f"unstable n={n} relative amplitude error vs cosh(gamma t)", err, JEANS_UNSTABLE_TOL[n]))
        profiles.append(("jeans_unstable_amp", n, _amplitude(s, x, L) / 1e-6, ex, 0, 0, 0, 0))
    orders = _orders(errs, ns)
    checks.append(_chk("unstable min observed order (32..256)", min(orders), 1.9, ">="))
    x, s0 = _perturbed(128, L, 1e-6, G=0.0)
    s0.run_to(t)
    a0 = _amplitude(s0, x, L) / 1e-6
    ex0 = np.cos(np.sqrt(C2) * k * t)
    checks += [_chk("no-gravity control matches acoustic cos(c k t)", abs(a0 - ex0) / abs(ex0), 1.2e-3),
               _chk("gravity amplifies: measured |amp(G=1)| / |amp(G=0)| >= 3", amps[128] / abs(a0), 3.0, ">=")]
    return _bench("jeans_unstable_growth", checks, resolutions=list(ns), relative_errors=errs,
                  observed_orders=orders, growth_rate=float(gam),
                  note="k<k_J; amplitude 1e-6 keeps the evolution linear")


def bench_jeans_stable():
    L, t = 1.0, 0.9                       # k = 2 pi > k_J: gravity-modified sound wave, omega^2 = c^2k^2-4 pi G rho0
    k = 2 * np.pi / L
    om = np.sqrt(C2 * k * k - 4 * np.pi)
    ns, errs, checks = (32, 64, 128, 256, 512), [], []
    for n in ns:
        x, s = _perturbed(n, L, 1e-6)
        s.run_to(t)
        errs.append(abs(_amplitude(s, x, L) - 1e-6 * np.cos(om * t)) / 1e-6)   # normalised by eps (t chosen on a steep part)
    for n, e in zip(ns, errs):
        if n in JEANS_STABLE_TOL:
            checks.append(_chk(f"stable n={n} |amp error|/eps vs cos(omega t)", e, JEANS_STABLE_TOL[n]))
    orders = _orders(errs, ns)
    checks.append(_chk("stable min observed order (32..512)", min(orders), 1.9, ">="))
    return _bench("jeans_stable_oscillation", checks, resolutions=list(ns), errors=errs, observed_orders=orders,
                  note="evaluated at omega*t=4.67 (steep part of cos); at omega*t~2*pi the amplitude is insensitive to phase error")


def bench_nonlinear_conservation():
    checks, drift = [], {}
    for n in (64, 128, 256):
        x, s = _perturbed(n, 2.0, 0.3)
        s.run_to(1.0)
        d = s.hydro.diagnostics()
        drift[n] = s.energy_drift()
        checks += [_chk(f"n={n} mass budget", d["budget_residual_rel"][0], 1e-13),
                   _chk(f"n={n} |total momentum| (gravity does no net work on momentum)", abs(d["totals"][1]), 1e-13),
                   _chk(f"n={n} total-energy drift (kinetic+internal+potential)", drift[n], ENERGY_DRIFT_TOL),
                   _chk(f"n={n} min rho>0 and min p>0", float(d["min_rho"] > 0 and d["min_p"] > 0), 1.0, ">=")]
    dd = {}
    for cfl in (0.4, 0.1):
        _, s = _perturbed(128, 2.0, 0.3, cfl=cfl)
        s.run_to(1.0)
        dd[cfl] = s.energy_drift()
    checks.append(_chk("energy drift falls >=10x when dt shrinks 4x (splitting error)", dd[0.4] / dd[0.1], 10.0, ">="))
    return _bench("nonlinear_conservation", checks, energy_drift_by_n=drift, drift_vs_cfl=dd,
                  note="eps=0.3 collapse to t=1; drift is splitting error, not roundoff; not claimed to be roundoff-level")


def bench_consistency():
    checks = []
    # uniform gas under gravity stays uniform (Jeans swindle)
    n = 64
    s = SelfGravityEuler1D(Euler1D(np.full(n, 2.0), np.zeros(n), np.full(n, 0.9), gamma=GAMMA), G=1.0)
    s.run_to(1.0)
    checks.append(_chk("uniform gas stays uniform", np.max(np.abs(s.hydro.primitives()[0] - 2.0)), 1e-13))
    # G = 0 reduces bitwise to pure hydro
    x, a = _perturbed(64, 2.0, 0.2, G=0.0)
    _, b = _perturbed(64, 2.0, 0.2, G=0.0)
    a.run_to(0.5)
    b.hydro.run_to(0.5)
    checks.append(_chk("G=0 identical to pure hydro (bitwise)", float(np.array_equal(a.hydro.U, b.hydro.U)), 1.0, ">="))
    # unit / scaling consistency: rho*3, p*3, G/3 -> identical velocities, rho/3
    def run(sc):
        _, s = _perturbed(64, 2.0, 0.2, G=1.0 / sc, scale=sc)
        s.run_to(0.6)
        return s.hydro.primitives()
    a, b = run(1.0), run(3.0)
    checks.append(_chk("scaling (rho,p x3, G/3) leaves dynamics unchanged", max(np.max(np.abs(a[0] - b[0] / 3)),
                       np.max(np.abs(a[1] - b[1])), np.max(np.abs(a[2] - b[2] / 3))), 1e-12))
    # gravity actually limits dt when G*rho is large
    h = Euler1D(np.ones(64), np.zeros(64), np.full(64, P0), gamma=GAMMA)
    g = SelfGravityEuler1D(h, G=200.0, kick_cfl=0.1)
    lim = 0.1 / np.sqrt(4 * np.pi * 200.0)
    checks += [_chk("gravity-limited dt equals kick_cfl/sqrt(4 pi G rho)", abs(g.stable_dt() - lim) / lim, 1e-12),
               _chk("gravity limit is tighter than hydro CFL here", lim / h.stable_dt(), 1.0)]
    # checkpoint replay at a step boundary is bitwise; a clipped mid-step pause is not (documented)
    def mk():
        return _perturbed(64, 2.0, 0.2)[1]
    saved, full = [], mk()
    full.run_to(0.8, callback=lambda q: saved.append(q.checkpoint()) if q.hydro.steps == 25 else None)
    rep = mk(); rep.restore(saved[0]); rep.run_to(0.8)
    checks.append(_chk("checkpoint at step boundary replays bitwise", float(np.array_equal(full.hydro.U, rep.hydro.U)), 1.0, ">="))
    mid = mk(); mid.run_to(0.4); cp = mid.checkpoint()
    rep2 = mk(); rep2.restore(cp); rep2.run_to(0.8)
    off = float(np.max(np.abs(full.hydro.U - rep2.hydro.U)))
    checks.append(_chk("mid-step (clipped) pause perturbs trajectory only at truncation level", off, 5e-5))
    return _bench("consistency_units_checkpoint", checks, mid_step_pause_difference=off,
                  note="pause/resume in the observatory must occur on step boundaries to replay bitwise")


def run_all(output_root="runs/benchmarks/gravity", command="universe benchmark gravity"):
    start = _dt.datetime.now(_dt.timezone.utc)
    run_id = start.strftime("%Y%m%dT%H%M%SZ")
    profiles = []
    results = [bench_force(), bench_jeans_unstable(profiles), bench_jeans_stable(),
               bench_nonlinear_conservation(), bench_consistency()]
    out = Path(output_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    n_checks = sum(len(r["checks"]) for r in results)
    n_pass = sum(c["passed"] for r in results for c in r["checks"])
    metrics = {"run_id": run_id, "all_passed": all(r["passed"] for r in results),
               "checks_passed": int(n_pass), "checks_total": int(n_checks), "benchmarks": results}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "profiles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "n", "measured", "exact", "c4", "c5", "c6", "c7"])
        w.writerows(profiles)
    (out / "metadata.json").write_text(json.dumps({
        "run_id": run_id, "started_utc": start.isoformat(), "command": command,
        "git_sha": _git_sha(Path(__file__).parent), "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), "gamma": GAMMA, "G": 1.0, "units": "dimensionless code units",
        "solver": "SelfGravityEuler1D: spectral periodic Poisson + Strang kick-hydro-kick",
        "references": REFERENCES, "tolerance_policy": __doc__.strip(),
        "scientific_status": "T1 numerical verification of a 1D self-gravitating gas slice; "
                             "no particles, no 3D, no cosmological expansion."}, indent=2))
    return metrics, out
