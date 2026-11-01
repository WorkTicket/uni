"""Regenerates docs/IMPLEMENTATION_STATUS.json from REAL test and benchmark runs (never hand-typed numbers).
Usage (from universe-starter/):  .venv/bin/python scripts/write_status.py"""
import datetime, glob, json, re, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
py = sys.executable


def sh(*a):
    return subprocess.run(a, cwd=root, capture_output=True, text=True)


tests = sh(py, "-m", "pytest", "-q")
tail = tests.stdout.strip().splitlines()[-1]
counts = {k: 0 for k in ("passed", "failed", "skipped")}
for n, k in re.findall(r"(\d+) (passed|failed|skipped)", tail):
    counts[k] = int(n)
sha = sh("git", "rev-parse", "--short", "HEAD").stdout.strip()
bench = {}
for suite in ("hydro", "gravity", "nbody"):
    r = sh(py, "-m", "universe_engine.cli", "benchmark", suite, "--output-root", f"runs/benchmarks/{suite}")
    runs = sorted(glob.glob(str(root / f"runs/benchmarks/{suite}/*/metrics.json")))
    m = json.load(open(runs[-1]))
    bench[suite] = {"exit_code": r.returncode, "run_id": m["run_id"], "checks": f'{m["checks_passed"]}/{m["checks_total"]}',
                    "benchmarks": [{"name": b["name"], "passed": b["passed"],
                                    "checks": f'{sum(c["passed"] for c in b["checks"])}/{len(b["checks"])}'} for b in m["benchmarks"]]}
status = json.load(open(root / "docs/status_static.json"))
status.update({"updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "repo_revision_at_generation": sha, "test_command": "python -m pytest -q", "test_counts": counts,
               "test_summary_line": tail, "benchmarks": bench})
json.dump(status, open(root / "docs/IMPLEMENTATION_STATUS.json", "w"), indent=2)
print(json.dumps({"tests": counts, "benchmarks": {k: v["checks"] for k, v in bench.items()}}))
sys.exit(0 if counts["failed"] == 0 and all(b["exit_code"] == 0 for b in bench.values()) else 1)
