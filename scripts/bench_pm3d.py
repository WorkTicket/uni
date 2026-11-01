"""PM3D performance benchmark: measure steps/sec, force-solver time,
density-deposition time, and snapshot serialization time at several
resolutions. Results printed to stdout and written to runs/benchmarks/pm3d."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from universe_engine.solvers.pm3d import Cosmology, ParticleMesh3D


H0 = 67.4
OM = 0.315
OR = 9.2e-5
OL = 0.684908


def _lattice(n_ppax, seed=0, amplitude=0.02):
    rng = np.random.default_rng(seed)
    n = n_ppax
    grid = np.stack(np.meshgrid(
        *([np.arange(n) / n + 0.5 / n] * 3), indexing="ij"), -1).reshape(-1, 3)
    grid = grid + amplitude * rng.normal(size=grid.shape)
    grid = grid % 1.0
    mass = np.full(grid.shape[0], 1.0 / grid.shape[0])
    return grid, mass


def bench_one(n_mesh, n_ppax, box_size_Mpc, n_steps, seed=7):
    cosmo = Cosmology(H0_km_s_Mpc=H0, Omega_m=OM, Omega_r=OR, Omega_Lambda=OL)
    x, m = _lattice(n_ppax, seed=seed, amplitude=0.02)
    sim = ParticleMesh3D(n=n_mesh, x=x, u=np.zeros_like(x), mass=m,
                         cosmology=cosmo, box_size_Mpc=box_size_Mpc, a=0.2)

    # Warm-up
    sim.run_to_a(0.21, max_d_ln_a=0.01)

    # Timed run: measure force + density + step time.
    t0 = time.perf_counter()
    for _ in range(n_steps):
        sim.step(ds=0.01)
    t1 = time.perf_counter()

    # Force-solver time (single comoving_acceleration call).
    tf0 = time.perf_counter()
    for _ in range(max(1, n_steps // 10)):
        sim.comoving_acceleration()
    tf1 = time.perf_counter()
    force_time = (tf1 - tf0) / max(1, n_steps // 10)

    # Density-deposition time (single density_contrast call).
    td0 = time.perf_counter()
    for _ in range(max(1, n_steps // 10)):
        sim.density_contrast()
    td1 = time.perf_counter()
    deposit_time = (td1 - td0) / max(1, n_steps // 10)

    # Snapshot serialization time.
    ts0 = time.perf_counter()
    for _ in range(max(1, n_steps // 10)):
        sim.snapshot()
    ts1 = time.perf_counter()
    snap_time = (ts1 - ts0) / max(1, n_steps // 10)

    elapsed = t1 - t0
    steps_per_sec = n_steps / elapsed if elapsed > 0 else float("inf")

    return {
        "n_mesh": n_mesh,
        "n_particles": n_ppax ** 3,
        "box_size_Mpc": box_size_Mpc,
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "steps_per_sec": steps_per_sec,
        "force_solver_s": force_time,
        "density_deposition_s": deposit_time,
        "snapshot_serialization_s": snap_time,
    }


def main():
    out_dir = Path("runs/benchmarks/pm3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        # (n_mesh, n_ppax, box_size_Mpc, n_steps)
        (16, 8, 100.0, 50),
        (32, 12, 100.0, 50),
        (32, 16, 250.0, 50),
        (64, 16, 500.0, 30),
    ]

    results = []
    for n_mesh, n_ppax, box, n_steps in configs:
        r = bench_one(n_mesh, n_ppax, box, n_steps)
        results.append(r)
        print(json.dumps(r))

    # Write summary.
    summary = {
        "benchmarks": results,
        "notes": {
            "integrator": "kdk_analytic_drag",
            "cosmology": {"H0": H0, "Omega_m": OM, "Omega_r": OR, "Omega_Lambda": OL},
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
