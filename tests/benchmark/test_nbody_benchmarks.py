import json
import subprocess
import sys

import numpy as np
import pytest

from universe_engine.benchmarks import nbody as B
from universe_engine.solvers import nbody as NB


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("nbench")
    proc = subprocess.run([sys.executable, "-m", "universe_engine.cli", "benchmark", "nbody", "--output-root", str(root)],
                          capture_output=True, text=True)
    run_dir = next(root.iterdir())
    return proc, run_dir, json.loads((run_dir / "metrics.json").read_text())


def test_cli_exit_code_and_artifacts(cli_run):
    proc, run_dir, m = cli_run
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert m["all_passed"] and m["checks_passed"] == m["checks_total"] >= 35
    assert "Zel'dovich" in json.loads((run_dir / "metadata.json").read_text())["references"][0]


@pytest.mark.parametrize("name", ["cic_deposition_interpolation", "zeldovich_sheet_nonlinear",
                                  "gas_particle_momentum_exchange", "two_fluid_jeans_gas_plus_particles",
                                  "consistency_units_checkpoint_snapshot"])
def test_each_benchmark_passes(cli_run, name):
    b = next(x for x in cli_run[2]["benchmarks"] if x["name"] == name)
    assert not [c["check"] for c in b["checks"] if not c["passed"]]


def test_observed_second_order_convergence(cli_run):
    bs = {x["name"]: x for x in cli_run[2]["benchmarks"]}
    assert min(bs["zeldovich_sheet_nonlinear"]["observed_orders"]) > 1.9
    assert min(bs["two_fluid_jeans_gas_plus_particles"]["gas_orders"]) > 1.9
    assert min(bs["two_fluid_jeans_gas_plus_particles"]["particle_orders"]) > 1.9


def test_exchange_is_real_and_exactly_balanced(cli_run):
    b = next(x for x in cli_run[2]["benchmarks"] if x["name"] == "gas_particle_momentum_exchange")
    assert abs(b["dP_gas"]) > 1e-3 and abs(b["dP_gas"] + b["dP_particles"]) < 1e-13


def _caught(mutate):
    saved = (NB.cic_deposit, NB.cic_interpolate, NB.GravitySystem1D._kick)
    try:
        mutate()
        prof = []
        return not all(r["passed"] for r in (B.bench_zeldovich(prof), B.bench_momentum_exchange(), B.bench_two_fluid()))
    finally:
        NB.cic_deposit, NB.cic_interpolate, NB.GravitySystem1D._kick = saved


def test_detects_ngp_instead_of_cic():
    def dep(x, m, x0, dx, n):
        i = np.floor((np.asarray(x) - x0) / dx).astype(int) % n
        return np.bincount(i, np.broadcast_to(np.asarray(m, float), np.shape(x)), n) / dx
    assert _caught(lambda: setattr(NB, "cic_deposit", dep))


def test_detects_flipped_particle_force():
    orig_i = NB.cic_interpolate
    def kick(self, dt):
        _, g = self.fields()
        if self.hydro is not None:
            U = self.hydro.U; mo = U[1].copy(); U[1] = mo + dt * U[0] * g; U[2] = U[2] + dt * g * .5 * (mo + U[1])
        if self.x is not None:
            self.v = self.v - dt * orig_i(g, self.x, self.x0, self.dx)
    assert _caught(lambda: setattr(NB.GravitySystem1D, "_kick", kick))


def test_detects_gas_ignoring_particles():
    orig_i = NB.cic_interpolate
    def kick(self, dt):
        _, g = self.fields()
        if self.hydro is not None:
            gg = NB.poisson_1d_periodic(self.hydro.U[0], self.dx, self.G)[1]
            U = self.hydro.U; mo = U[1].copy(); U[1] = mo + dt * U[0] * gg; U[2] = U[2] + dt * gg * .5 * (mo + U[1])
        if self.x is not None:
            self.v = self.v + dt * orig_i(g, self.x, self.x0, self.dx)
    assert _caught(lambda: setattr(NB.GravitySystem1D, "_kick", kick))
