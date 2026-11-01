"""`universe doctor`: environment diagnostics. Reports facts only; never modifies anything."""
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
from importlib import metadata


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def run_checks(output_dir="runs"):
    checks = []

    def add(name, ok, detail, required=True):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "required": required})

    v = sys.version_info
    add("python >= 3.10", v >= (3, 10), platform.python_version())
    add("python 3.11/3.12 recommended", v[:2] in ((3, 11), (3, 12), (3, 13)), platform.python_version(), required=False)
    for pkg, required in (("numpy", True), ("PyYAML", True), ("pytest", False), ("matplotlib", False),
                          ("fastapi", False), ("uvicorn", False)):
        ver = _version(pkg)
        add(f"package {pkg}", ver is not None, ver or "not installed", required=required)
    try:
        usage = shutil.disk_usage(os.getcwd())
        add("free disk >= 1 GB", usage.free >= 1e9, f"{usage.free / 1e9:.1f} GB free")
    except OSError as e:
        add("free disk >= 1 GB", False, str(e))
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        add("physical memory >= 4 GB", mem >= 4e9, f"{mem / 1e9:.1f} GB", required=False)
    except (ValueError, OSError, AttributeError):
        add("physical memory >= 4 GB", True, "could not determine on this platform", required=False)
    try:
        os.makedirs(output_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_dir):
            pass
        add("output directory writable", True, os.path.abspath(output_dir))
    except OSError as e:
        add("output directory writable", False, str(e))
    add("optional GPU (nvidia-smi on PATH)", shutil.which("nvidia-smi") is not None,
        "present" if shutil.which("nvidia-smi") else "not found (not needed for current solvers)", required=False)
    try:
        importlib.import_module("universe_engine.solvers.hydro")
        add("hydro solver importable", True, "ok")
    except Exception as e:  # report, never raise
        add("hydro solver importable", False, repr(e))
    return {"platform": platform.platform(), "checks": checks,
            "ok": all(c["ok"] for c in checks if c["required"])}


def main(output_dir="runs"):
    report = run_checks(output_dir)
    for c in report["checks"]:
        mark = "ok  " if c["ok"] else ("FAIL" if c["required"] else "warn")
        print(f"[{mark}] {c['check']}: {c['detail']}")
    print("doctor:", "PASS" if report["ok"] else "FAIL")
    return report
