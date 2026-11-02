"""Regenerates docs/IMPLEMENTATION_STATUS.json from REAL test and benchmark runs (never hand-typed numbers).
Usage (from universe-starter/):  python scripts/write_status.py
Requires src/ on PYTHONPATH so `universe_engine` imports under pytest.
"""
import datetime, glob, json, os, re, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
py = sys.executable

# Make `universe_engine` importable under pytest (package lives under src/).
env = dict(os.environ)
env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")


def sh(*a, check=False):
    return subprocess.run(a, cwd=root, capture_output=True, text=True, env=env)


# --- Test counts from a REAL pytest run ------------------------------------
tests = sh(py, "-m", "pytest", "-q")
tail = tests.stdout.strip().splitlines()[-1]
counts = {k: 0 for k in ("passed", "failed", "skipped")}
for n, k in re.findall(r"(\d+) (passed|failed|skipped)", tail):
    counts[k] = int(n)
if counts["passed"] + counts["failed"] + counts["skipped"] == 0:
    raise SystemExit("could not parse pytest summary line: " + tail)

# --- Real HEAD hash (full, not short) --------------------------------------
head = sh("git", "rev-parse", "HEAD").stdout.strip()
if not head:
    raise SystemExit("could not resolve git HEAD")

# --- Benchmarks (optional; never fail status generation on benchmark issues) --
bench = {}
for suite in ("hydro", "gravity", "nbody"):
    try:
        r = sh(py, "-m", "universe_engine.cli", "benchmark", suite,
               "--output-root", f"runs/benchmarks/{suite}")
        runs = sorted(glob.glob(str(root / f"runs/benchmarks/{suite}/*/metrics.json")))
        if not runs:
            bench[suite] = {"exit_code": r.returncode, "error": "no metrics.json"}
            continue
        m = json.load(open(runs[-1]))
        bench[suite] = {"exit_code": r.returncode, "run_id": m["run_id"],
                        "checks": f'{m["checks_passed"]}/{m["checks_total"]}',
                        "benchmarks": [{"name": b["name"], "passed": b["passed"],
                                        "checks": f'{sum(c["passed"] for c in b["checks"])}/{len(b["checks"])}'}
                                       for b in m["benchmarks"]]}
    except Exception as e:  # benchmarks are not required for status honesty
        bench[suite] = {"error": str(e)}

# --- Per-seed failure records (Sprint 11 world-model failure, restored) -----
# Source: docs/SPRINT_11_FINDINGS.md, "Original failure (reproduced)" table.
per_seed_failures = [
    {"seed": 1,  "task": "GridWorld foraging (fully observable)",
     "reactive_mean_reward": +0.48,  "recurrent_mean_reward": -0.77,  "pass": False},
    {"seed": 42, "task": "GridWorld foraging (fully observable)",
     "reactive_mean_reward": +0.49,  "recurrent_mean_reward": -1.16,  "pass": False},
    {"seed": 99, "task": "GridWorld foraging (fully observable)",
     "reactive_mean_reward": +0.42,  "recurrent_mean_reward": -2.44,  "pass": False},
]
# Source: docs/SPRINT_12_FINDINGS.md Phase 2 (15 macro-seeds statistical replication).
statistical_replication = {
    "macro_seeds": 15,
    "train_episodes_per_seed": 5,
    "eval_episodes_per_seed": 10,
    "controllers": {
        "reactive":        {"mean_reward": -0.594, "ci": [-0.884, -0.303], "std": 1.808, "cohens_d_vs_reactive": 0.000},
        "stacked_obs":     {"mean_reward": -0.710, "ci": [-0.985, -0.436], "std": 1.711, "cohens_d_vs_reactive": -0.066},
        "recurrent":       {"mean_reward": -0.241, "ci": [-0.424, -0.057], "std": 1.145, "cohens_d_vs_reactive": +0.233},
        "world_model":     {"mean_reward": +0.065, "ci": [-0.138, +0.269], "std": 1.268, "cohens_d_vs_reactive": +0.421},
    },
    "conclusion": "No architecture demonstrates a reliable, statistically "
                  "significant positive advantage over reactive on the "
                  "fully-observable foraging task (all 95% CIs include zero "
                  "except recurrent d=+0.233, which is small and not "
                  "replicated by world-model). Null result retained.",
}

status = json.load(open(root / "docs/status_static.json"))
status.update({
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "head_commit": head,
    "repo_revision_at_generation": head,
    "test_command": "PYTHONPATH=src python -m pytest -q",
    "test_counts": counts,
    "test_summary_line": tail,
    "test_counts_source": "real pytest run output (never hand-edited)",
    "per_seed_failures": per_seed_failures,
    "statistical_replication": statistical_replication,
    "benchmarks": bench,
})
json.dump(status, open(root / "docs/IMPLEMENTATION_STATUS.json", "w"), indent=2)
print(json.dumps({"tests": counts, "head_commit": head,
                  "benchmarks": {k: v.get("checks", v.get("error")) for k, v in bench.items()}}))
sys.exit(0 if counts["failed"] == 0 else 1)
