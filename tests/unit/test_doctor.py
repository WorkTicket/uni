import subprocess, sys
from universe_engine.doctor import run_checks


def test_doctor_reports_required_checks_and_is_read_only(tmp_path):
    out = tmp_path / "o"
    r = run_checks(str(out))
    names = {c["check"] for c in r["checks"]}
    assert {"python >= 3.10", "package numpy", "output directory writable", "hydro solver importable"} <= names
    assert r["ok"] is True
    assert list(out.iterdir()) == []            # leaves no files behind


def test_doctor_detects_unwritable_output(tmp_path):
    blocker = tmp_path / "file"; blocker.write_text("x")
    r = run_checks(str(blocker / "sub"))        # a path under a regular file cannot be created
    assert r["ok"] is False
    assert not next(c for c in r["checks"] if c["check"] == "output directory writable")["ok"]


def test_cli_doctor_exit_code():
    p = subprocess.run([sys.executable, "-m", "universe_engine.cli", "doctor"], capture_output=True, text=True)
    assert p.returncode == 0 and "doctor: PASS" in p.stdout
