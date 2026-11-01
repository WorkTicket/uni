"""R4 (1D slice) benchmarks via the real CLI, plus permanent sabotage regression tests."""
import json
import subprocess
import sys

import numpy as np
import pytest

from universe_engine.benchmarks import gravity as B
from universe_engine.solvers import gravity as GR


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("gbench")
    proc = subprocess.run([sys.executable, "-m", "universe_engine.cli", "benchmark", "gravity",
                           "--output-root", str(root)], capture_output=True, text=True)
    run_dir = next(root.iterdir())
    return proc, run_dir, json.loads((run_dir / "metrics.json").read_text())


def test_cli_exit_code_and_artifacts(cli_run):
    proc, run_dir, m = cli_run
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert m["all_passed"] and m["checks_passed"] == m["checks_total"] >= 35
    for f in ("metrics.json", "profiles.csv", "metadata.json"):
        assert (run_dir / f).stat().st_size > 0
    assert "Jeans" in json.loads((run_dir / "metadata.json").read_text())["references"][0]


@pytest.mark.parametrize("name", ["poisson_force", "jeans_unstable_growth", "jeans_stable_oscillation",
                                  "nonlinear_conservation", "consistency_units_checkpoint"])
def test_each_benchmark_passes(cli_run, name):
    b = next(x for x in cli_run[2]["benchmarks"] if x["name"] == name)
    assert not [c["check"] for c in b["checks"] if not c["passed"]]


def test_second_order_convergence_and_spectral_force(cli_run):
    bs = {x["name"]: x for x in cli_run[2]["benchmarks"]}
    assert min(bs["jeans_unstable_growth"]["observed_orders"]) > 1.9
    assert min(bs["jeans_stable_oscillation"]["observed_orders"]) > 1.9
    g = bs["poisson_force"]["gaussian_errors"]
    assert g["64"] < 1e-13 < g["16"]                          # spectral: roundoff by 64 cells


def _caught(mutate):
    saved = (GR.poisson_1d_periodic, GR.SelfGravityEuler1D._kick, GR.SelfGravityEuler1D.step)
    try:
        mutate()
        prof = []
        return not all(r["passed"] for r in (B.bench_force(), B.bench_jeans_unstable(prof),
                                             B.bench_nonlinear_conservation(), B.bench_consistency()))
    finally:
        GR.poisson_1d_periodic, GR.SelfGravityEuler1D._kick, GR.SelfGravityEuler1D.step = saved


def test_detects_repulsive_gravity():
    orig = GR.poisson_1d_periodic
    assert _caught(lambda: setattr(GR, "poisson_1d_periodic", lambda r, dx, G=1.0: (lambda p, g: (p, -g))(*orig(r, dx, G))))


def test_detects_missing_energy_update_in_kick():
    def kick(self, dt):
        U = self.hydro.U
        _, g = GR.poisson_1d_periodic(U[0], self.hydro.dx, self.G)
        U[1] = U[1] + dt * U[0] * g
    assert _caught(lambda: setattr(GR.SelfGravityEuler1D, "_kick", kick))


def test_detects_net_force_bias():
    orig = GR.poisson_1d_periodic
    assert _caught(lambda: setattr(GR, "poisson_1d_periodic", lambda r, dx, G=1.0: (lambda p, g: (p, g + 1e-6))(*orig(r, dx, G))))
