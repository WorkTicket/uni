"""R3 benchmarks executed through the real CLI, plus permanent sabotage regression tests."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from universe_engine.benchmarks import hydro as B
from universe_engine.solvers import hydro as H


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    proc = subprocess.run([sys.executable, "-m", "universe_engine.cli", "benchmark", "hydro",
                           "--output-root", str(root)], capture_output=True, text=True)
    run_dir = next(root.iterdir())
    return proc, run_dir, json.loads((run_dir / "metrics.json").read_text())


def test_cli_exit_code_and_artifacts(cli_run):
    proc, run_dir, metrics = cli_run
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert metrics["all_passed"] and metrics["checks_passed"] == metrics["checks_total"] >= 100
    for f in ("metrics.json", "profiles.csv", "metadata.json"):
        assert (run_dir / f).stat().st_size > 0
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["references"] and "Toro" in meta["references"][0] and meta["command"].startswith("universe benchmark")


@pytest.mark.parametrize("name", ["uniform_periodic_flow", "sod_shock_tube", "acoustic_wave_convergence",
                                  "acoustic_nonlinear_floor", "conservation_budgets",
                                  "strong_shock_robustness", "vacuum_generation_contract", "boundary_conditions"])
def test_each_benchmark_passes(cli_run, name):
    b = next(x for x in cli_run[2]["benchmarks"] if x["name"] == name)
    failed = [c["check"] for c in b["checks"] if not c["passed"]]
    assert not failed, failed


def test_observed_orders_are_what_theory_predicts(cli_run):
    b = next(x for x in cli_run[2]["benchmarks"] if x["name"] == "acoustic_wave_convergence")
    assert min(b["observed_orders"]["mc"]) > 1.8            # smooth flow: second order
    assert 0.9 < np.mean(b["observed_order_first_order_scheme"]) < 1.05   # baseline: first order


def test_profiles_csv_contains_exact_reference_columns(cli_run):
    header = (cli_run[1] / "profiles.csv").read_text().splitlines()[0]
    assert header == "benchmark,x,rho,u,p,rho_ref,u_ref,p_ref"


# ---- the suite must be able to FAIL: deliberately broken solvers are detected
def _sabotage_detected(mutate):
    saved = (dict(H.FLUXES), dict(H.LIMITERS), H.Euler1D._pad, H.Euler1D._advance)
    try:
        mutate()
        prof = []
        return not all(r["passed"] for r in (B.bench_sod(prof), B.bench_acoustic(prof), B.bench_boundaries(),
                                             B.bench_conservation()))
    finally:
        H.FLUXES.clear(); H.FLUXES.update(saved[0]); H.LIMITERS.clear(); H.LIMITERS.update(saved[1])
        H.Euler1D._pad, H.Euler1D._advance = saved[2], saved[3]


def test_detects_silent_first_order_degradation():
    assert _sabotage_detected(lambda: H.LIMITERS.update(mc=lambda a, b: 0 * a))


def test_detects_missing_momentum_flip_at_wall():
    orig = H.Euler1D._pad

    def bad(self, W):
        out = orig(self, W)
        if self.bc[0] == "reflective":
            out[1, :H.NG] *= -1
        return out
    assert _sabotage_detected(lambda: setattr(H.Euler1D, "_pad", bad))


def test_detects_tiny_mass_leak():
    orig = H.Euler1D._advance

    def leak(self, U, dt, depth=0):
        V = orig(self, U, dt, depth).copy()
        V[0] *= 1 - 1e-9
        return V
    assert _sabotage_detected(lambda: setattr(H.Euler1D, "_advance", leak))
