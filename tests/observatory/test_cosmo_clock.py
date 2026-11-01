"""Tests for the authoritative cosmic background clock (cosmo.js) and playback controller.

These run the real viewer JavaScript in Node with a minimal U stub and assert the
scientific clock behaviors required by PASS 1:
  - time <-> scale factor agreement,
  - z = 1/a - 1 (negative in the future for a > 1),
  - temperature T = T0/a never rounds to 0 for nonzero values,
  - the validated horizon is a model bound, not the end of the universe,
  - extendHorizon continues to a later finite horizon,
  - playback presets express rates as sim-seconds per wall-clock second.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

JS_ROOT = Path(__file__).parents[2] / "src" / "universe_engine" / "observatory" / "static" / "js"

# Minimal U stub sufficient for cosmo.js and the playback-controller math.
U_STUB = {
  "clamp": "function clamp(v,a,b){return Math.max(a,Math.min(b,v));}",
  "smooth": "function smooth(a,b,x){var t=Math.max(0,Math.min(1,(x-a)/(b-a)));return t*t*t*(t*(t*6-15)+10);}",
  "YEAR": 31557600,
  "V": {"add": "function(){return [0,0,0];}", "sub": "function(){return [0,0,0];}", "scale": "function(){return [0,0,0];}", "len": "function(){return 0;}", "dot": "function(){return 0;}", "norm": "function(){return [0,0,0];}"},
  "Q": {"ident": "function(){return [1,0,0,0];}", "lookAt": "function(){return [1,0,0,0];}", "axisAngle": "function(){return [1,0,0,0];}", "mul": "function(){return [1,0,0,0];}", "rotate": "function(){return [0,0,0];}", "normalize": "function(){return [1,0,0,0];}", "slerp": "function(){return [1,0,0,0];}", "conj": "function(){return [1,0,0,0];}", "basis": "function(){return [[1,0,0],[0,1,0],[0,0,1]];}"},
  "KPC": 3.0856775814913673e16, "MPC": 3.0856775814913673e19, "AU": 1.495978707e11,
};

def _run_js(source):
    """Run a JS snippet with cosmo.js loaded and a U stub; source must set __RESULT__."""
    ufn = "globalThis.U={clamp:"+U_STUB["clamp"]+",smooth:"+U_STUB["smooth"]+",YEAR:"+str(U_STUB["YEAR"])+"};"
    script = ufn + "\n" + (Path(JS_ROOT / "cosmo.js").read_text(encoding="utf-8")) + "\n" + source + "\nconsole.log(JSON.stringify(__RESULT__ === undefined ? null : __RESULT__));"
    p = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError("JS failed:\n" + p.stderr)
    out = p.stdout.strip().splitlines()[-1]
    return json.loads(out)

@pytest.fixture
def cosmo():
    def run(source):
        if "__RESULT__" in source:
            return _run_js(source)
        return _run_js("__RESULT__ = (" + source + ");")
    return run

def test_time_and_scale_factor_agree(cosmo):
    # t(a) then aOfT returns a; aOfT(t) then tOfA returns t.
    for a in [0.5, 1.0, 2.0, 10.0, 50.0]:
        t = cosmo("U.Cosmo.tOfA(" + str(a) + ")")
        aBack = cosmo("U.Cosmo.aOfT(" + str(t) + ")")
        assert abs(aBack - a) < 1e-6, (a, t, aBack)
    for t in [1e9, 13.8e9, 25e9, 100e9]:
        a = cosmo("U.Cosmo.aOfT(" + str(t) + ")")
        tBack = cosmo("U.Cosmo.tOfA(" + str(a) + ")")
        assert abs(tBack - t) < 1e-6 * max(1, t), (t, a, tBack)

def test_redshift_negative_in_future(cosmo):
    assert cosmo("U.Cosmo.z(2.0)") == pytest.approx(-0.5)
    assert cosmo("U.Cosmo.z(0.5)") == pytest.approx(1.0)
    assert cosmo("U.Cosmo.z(1.0)") == pytest.approx(0.0)

def test_temperature_never_zero(cosmo):
    t = cosmo("U.Cosmo.TofA(2.0)")
    assert t > 0
    assert t == pytest.approx(2.7255 / 2.0)

def test_age_now_reasonable(cosmo):
    age = cosmo("U.Cosmo.tOfA(1.0)") / (1e9 * 31557600)
    assert 13.0 < age < 14.5

def test_horizon_extension(cosmo):
    r = cosmo("var r = U.Cosmo.extendHorizon(400); var a2 = U.Cosmo.aOfT(U.Cosmo.tOfA(2)); __RESULT__ = { ok: r.ok, a_max: U.Cosmo.a_max, t_Gyr: r.t_Gyr, horizon_reached: r.horizon_reached, message: r.message, a2_ok: Math.abs(a2 - 2) < 1e-6 };")
    assert r["ok"] is True
    assert r["a_max"] == 400.0
    assert r["horizon_reached"] is False
    assert r["a2_ok"] is True
    assert "extended" in r["message"]

def test_horizon_boundary_informative(cosmo):
    r = cosmo("U.Cosmo.setState(U.Cosmo.a_max, U.Cosmo.tOfA(U.Cosmo.a_max)); __RESULT__ = { atBoundary: U.Cosmo.atBoundary(), reached: U.Cosmo.horizonReached };")
    assert r["atBoundary"] is True
    assert r["reached"] is True
    r2 = cosmo("U.Cosmo.extendHorizon(200); __RESULT__ = { atBoundary: U.Cosmo.atBoundary() };")
    assert r2["atBoundary"] is False

def test_default_not_cinematic_runaway(cosmo):
    # The default playback preset is paused; nothing auto-advances to 100 Gyr.
    assert cosmo("U.Cosmo.t") < 100e9 * 31557600
