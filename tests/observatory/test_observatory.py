import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from universe_engine.observatory.server import create_app
from universe_engine.observatory.session import Session


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "saves")) as c:
        yield c


def _strip(snap):
    return {k: v for k, v in snap.items() if k not in ("version", "running")}


def _make(client, preset="zeldovich", n=32, **extra):
    r = client.post("/api/session", json={"preset": preset, "params": {"n": n, **extra}})
    assert r.status_code == 200, r.text
    return r.json()


def test_presets_and_default_state(client):
    p = client.get("/api/presets").json()
    assert set(p) == {"zeldovich", "gas_particles"} and "limits" in p["zeldovich"]
    s = client.get("/api/state").json()
    assert s["steps"] == 0 and s["running"] is False and "Not a 3D universe" in s["scientific_status"]


@pytest.mark.parametrize("body", [{"preset": "nope"}, {"preset": "zeldovich", "params": {"n": 4}},
                                  {"preset": "zeldovich", "params": {"n": "64"}}, {"preset": "zeldovich", "params": {"bogus": 1}},
                                  {"preset": "zeldovich", "params": {"A": 5}}])
def test_invalid_parameters_are_rejected_with_422(client, body):
    r = client.post("/api/session", json=body)
    assert r.status_code == 422 and "error" in r.json()


@pytest.mark.parametrize("token", ["Infinity", "-Infinity", "NaN"])
def test_non_standard_json_numbers_sent_as_raw_text_are_rejected(client, token):
    raw = '{"preset": "zeldovich", "params": {"A": %s}}' % token
    r = client.post("/api/session", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "error" in r.json()


@pytest.mark.parametrize("preset", ["zeldovich", "gas_particles"])
def test_step_advances_time_and_preserves_invariants(client, preset):
    _make(client, preset)
    s = client.post("/api/step", json={"n": 5}).json()
    assert s["steps"] == 5 and s["t"] > 0
    assert abs(s["total_momentum"]) < 1e-12
    snap = client.get("/api/snapshot").json()
    assert snap["schema"] == "observatory-snapshot/v1" and len(snap["density"]) == snap["n"] == 32
    assert ("gas_density" in snap) == (preset == "gas_particles") and len(snap["particle_x"]) > 0


@pytest.mark.parametrize("bad", [0, -1, 10 ** 6, "5", 1.5, True])
def test_step_count_validated(client, bad):
    assert client.post("/api/step", json={"n": bad}).status_code == 422


def test_same_inputs_give_identical_results(tmp_path):
    out = []
    for _ in range(2):
        with TestClient(create_app(tmp_path / "s")) as c:
            _make(c, "gas_particles"); c.post("/api/step", json={"n": 12}); out.append(_strip(c.get("/api/snapshot").json()))
    assert json.dumps(out[0], sort_keys=True) == json.dumps(out[1], sort_keys=True)


def test_run_pause_lands_on_step_boundary_and_resumes_bitwise(tmp_path):
    with TestClient(create_app(tmp_path / "a")) as c:
        _make(c); c.post("/api/run", json={"delay": 0.002})
        deadline = time.time() + 20
        while c.get("/api/state").json()["steps"] < 5 and time.time() < deadline:
            time.sleep(0.01)
        paused = c.post("/api/pause").json()
        assert paused["running"] is False and paused["steps"] >= 5
        k = paused["steps"]
        time.sleep(0.1)
        assert c.get("/api/state").json()["steps"] == k            # really stopped
        c.post("/api/step", json={"n": 10})
        a = _strip(c.get("/api/snapshot").json())
    with TestClient(create_app(tmp_path / "b")) as c2:             # uninterrupted reference
        _make(c2); c2.post("/api/step", json={"n": k + 10})
        b = _strip(c2.get("/api/snapshot").json())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_run_with_max_steps_stops_by_itself(client):
    _make(client); client.post("/api/run", json={"max_steps": 7})
    deadline = time.time() + 20
    while client.get("/api/state").json()["running"] and time.time() < deadline:
        time.sleep(0.01)
    assert client.get("/api/state").json()["steps"] == 7


def test_manual_step_while_running_is_refused(client):
    _make(client); client.post("/api/run", json={"delay": 0.01})
    assert client.post("/api/step", json={"n": 1}).status_code == 409
    client.post("/api/pause")


@pytest.mark.parametrize("preset", ["zeldovich", "gas_particles"])
def test_save_load_resume_is_bitwise_identical(client, preset):
    _make(client, preset); client.post("/api/step", json={"n": 8})
    sid = client.post("/api/save", json={"id": "ck"}).json()["id"]
    client.post("/api/step", json={"n": 8}); a = _strip(client.get("/api/snapshot").json())
    assert client.post("/api/load", json={"id": sid}).json()["steps"] == 8
    client.post("/api/step", json={"n": 8}); b = _strip(client.get("/api/snapshot").json())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert "ck" in client.get("/api/saves").json()["saves"]


def test_save_file_loads_in_a_fresh_process_state(tmp_path):
    with TestClient(create_app(tmp_path / "s")) as c:
        _make(c, "gas_particles"); c.post("/api/step", json={"n": 6}); c.post("/api/save", json={"id": "x"})
        c.post("/api/step", json={"n": 6}); a = _strip(c.get("/api/snapshot").json())
    with TestClient(create_app(tmp_path / "s")) as c2:             # new app instance, same directory
        c2.post("/api/load", json={"id": "x"}); c2.post("/api/step", json={"n": 6})
        assert json.dumps(a, sort_keys=True) == json.dumps(_strip(c2.get("/api/snapshot").json()), sort_keys=True)


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "", "x" * 80, "a b"])
def test_path_traversal_and_bad_ids_rejected(client, bad):
    assert client.post("/api/load", json={"id": bad}).status_code in (404, 422)
    assert client.post("/api/save", json={"id": bad or "ok"}).status_code in (200, 422) if bad else True
    if bad in ("../etc/passwd", "a/b", "x" * 80, "a b"):
        assert client.post("/api/save", json={"id": bad}).status_code == 422


def test_load_missing_save_is_404(client):
    assert client.post("/api/load", json={"id": "nothing"}).status_code == 404


def test_corrupted_or_tampered_save_is_rejected(tmp_path):
    ses = Session("zeldovich", {"n": 16}); path = tmp_path / "t.npz"; ses.save(path)
    import numpy as np
    z = dict(np.load(path)); meta = json.loads(str(z["meta"])); meta["params"]["A"] = 0.1
    z["meta"] = np.array(json.dumps(meta)); np.savez(tmp_path / "bad.npz", **z)
    with pytest.raises(Exception, match="mismatch"):
        Session.load(tmp_path / "bad.npz")


def test_inspect_returns_local_physical_values(client):
    _make(client, "gas_particles"); client.post("/api/step", json={"n": 3})
    r = client.get("/api/inspect", params={"x": 0.5}).json()
    assert r["gas_density"] > 0 and r["sound_speed"] > 0 and "acceleration" in r and r["particles_in_cell"] >= 0
    assert client.get("/api/inspect", params={"x": 99}).status_code == 422
    assert client.get("/api/inspect").status_code == 422


def test_websocket_streams_state_changes(client):
    _make(client)
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["schema"] == "observatory-snapshot/v1" and first["steps"] == 0
        client.post("/api/step", json={"n": 4})
        deadline, got = time.time() + 10, None
        while time.time() < deadline:
            got = ws.receive_json()
            if got["steps"] == 4:
                break
        assert got["steps"] == 4 and len(got["density"]) == 32


def test_new_session_replaces_old_and_stops_running_one(client):
    _make(client); client.post("/api/run", json={"delay": 0.01})
    s = _make(client, "gas_particles")
    assert s["preset"] == "gas_particles" and s["steps"] == 0 and s["running"] is False


# ---- the browser viewer: served, and its real script executed against a real snapshot in Node
def test_lab_viewer_is_served(client):
    r = client.get("/lab")
    assert r.status_code == 200 and "Universe Observatory" in r.text and "Not a 3D universe" in r.text


def test_universe_viewer_and_assets_are_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<canvas" in r.text
    import re as _re
    scripts = _re.findall(r'src="(/?js/[^"]+)"', r.text)
    assert len(scripts) >= 10
    for s in scripts + ["css/app.css"]:
        a = client.get("/" + s.lstrip("/"))
        assert a.status_code == 200 and len(a.content) > 100, s


def test_cosmology_endpoint(client):
    c = client.get("/api/cosmology").json()
    assert 60 < c["H0_km_s_Mpc"] < 80 and 0.2 < c["Omega_m"] < 0.4 and 0 < c["Omega_r"] < 1e-3


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_universe_viewer_scripts_parse():
    js = sorted((Path(__file__).parents[2] / "src/universe_engine/observatory/static/js").glob("*.js"))
    assert js
    for f in js:
        p = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert p.returncode == 0, f"{f.name}: {p.stderr}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("preset", ["zeldovich", "gas_particles"])
def test_viewer_script_executes_against_real_snapshot(client, tmp_path, preset):
    _make(client, preset); client.post("/api/step", json={"n": 3})
    snap = tmp_path / "snap.json"; snap.write_text(json.dumps(client.get("/api/snapshot").json()))
    html = Path(__file__).parents[2] / "src/universe_engine/observatory/static/lab.html"
    p = subprocess.run(["node", str(Path(__file__).parent / "viewer_harness.js"), str(snap), str(html)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert out["steps"] == "3" and out["calls"].get("stroke", 0) >= 1 and out["calls"].get("clearRect", 0) == 2
    assert out["calls"].get("fillRect", 0) > 0 if preset else True
    assert out["fmtNaN"] == "-" and out["fmtNum"] == "1.23e+3"


def test_warning_appears_only_when_leaving_the_verified_regime(client):
    _make(client, "gas_particles", n=64, A=0.08)
    assert client.post("/api/step", json={"n": 5}).json()["warnings"] == []
    assert client.post("/api/step", json={"n": 400}).json()["warnings"]      # deep nonlinear collapse
    assert "beyond the verified benchmarks" in client.get("/api/state").json()["warnings"][0]


# ---- research laboratory ----
def test_lab_create_and_list(client):
    r = client.post("/api/lab", json={"name": "Evo-1", "seed": 7, "config": {"n_founders": 10, "n_patches": 4}})
    assert r.status_code == 200, r.text
    exp = r.json()["experiment"]
    assert exp["name"] == "Evo-1" and exp["status"] == "created" and exp["steps"] == 0
    data = client.get("/api/lab").json()
    assert any(e["id"] == exp["id"] for e in data["experiments"])


def test_lab_step_advances_population(client):
    r = client.post("/api/lab", json={"name": "Evo-2", "seed": 3, "config": {"n_founders": 12, "dt_gyr": 1e-4}})
    eid = r.json()["experiment"]["id"]
    s = client.post(f"/api/lab/{eid}/step", json={"n": 5}).json()
    assert s["experiment"]["steps"] == 5
    st = client.get(f"/api/lab/{eid}/stats").json()
    assert st["n_organisms"] >= 0 and st["steps"] == 5


def test_lab_run_and_pause(client):
    r = client.post("/api/lab", json={"name": "Evo-3", "seed": 5, "config": {"n_founders": 8, "dt_gyr": 1e-4}})
    eid = r.json()["experiment"]["id"]
    client.post(f"/api/lab/{eid}/run", json={"max_steps": 6})
    st = client.get(f"/api/lab/{eid}/stats").json()
    assert st["steps"] == 6 and st["status"] == "finished"
    client.post(f"/api/lab/{eid}/pause")
    assert client.get(f"/api/lab/{eid}/stats").json()["status"] == "paused"


def test_lab_trajectory_and_history(client):
    r = client.post("/api/lab", json={"name": "Evo-4", "seed": 9, "config": {"n_founders": 6, "dt_gyr": 1e-4}})
    eid = r.json()["experiment"]["id"]
    client.post(f"/api/lab/{eid}/step", json={"n": 4})
    traj = client.get(f"/api/lab/{eid}/trajectory").json()
    assert len(traj["t"]) >= 1 and len(traj["organisms"]) == len(traj["t"])
    hist = client.get(f"/api/lab/{eid}/history").json()
    assert len(hist["history"]) >= 1


def test_lab_sterile_world_stays_sterile(client):
    # Zero founders -> no organisms ever appear
    r = client.post("/api/lab", json={"name": "Sterile", "seed": 1, "config": {"n_founders": 0, "dt_gyr": 1e-4}})
    eid = r.json()["experiment"]["id"]
    client.post(f"/api/lab/{eid}/step", json={"n": 10})
    st = client.get(f"/api/lab/{eid}/stats").json()
    assert st["n_organisms"] == 0 and st["sterile"] is True


def test_lab_checkpoint_and_restore(client):
    r = client.post("/api/lab", json={"name": "Evo-5", "seed": 2, "config": {"n_founders": 10, "dt_gyr": 1e-4}})
    eid = r.json()["experiment"]["id"]
    client.post(f"/api/lab/{eid}/step", json={"n": 3})
    cp = client.post(f"/api/lab/{eid}/checkpoint").json()
    assert "world" in cp and "experiment" in cp
    # advance more, then restore
    client.post(f"/api/lab/{eid}/step", json={"n": 3})
    after = client.get(f"/api/lab/{eid}/stats").json()
    assert after["steps"] == 6
    restored = client.post(f"/api/lab/{eid}/restore", json=cp).json()
    assert restored["experiment"]["steps"] == 3


def test_lab_lineage_and_extinctions(client):
    r = client.post("/api/lab", json={"name": "Evo-6", "seed": 4, "config": {"n_founders": 8, "dt_gyr": 1e-4}})
    eid = r.json()["experiment"]["id"]
    client.post(f"/api/lab/{eid}/step", json={"n": 5})
    lin = client.get(f"/api/lab/{eid}/lineage").json()
    assert "n_lineages" in lin
    ex = client.get(f"/api/lab/{eid}/extinctions").json()
    assert "extinctions" in ex


def test_lab_bad_id_rejected(client):
    assert client.post("/api/lab/../etc/passwd/step", json={"n": 1}).status_code in (404, 422)
    assert client.get("/api/lab/nonexistent-id/stats").status_code in (404, 422)


def test_research_lab_page_served(client):
    r = client.get("/research-lab")
    assert r.status_code == 200 and "<canvas" in r.text


# ---- read-only planetary habitat API ----
def test_habitat_create_and_get(client):
    r = client.post("/api/habitat", json={"id": "h1", "seed": 3, "n_patches": 4})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "h1" and body["status"] == "created"
    assert "state" in body and "sterile" in body["state"]

    g = client.get("/api/habitat/h1").json()
    assert g["sterile"] is True
    assert g["replicators"] == {}


def test_habitat_chemistry_endpoint(client):
    client.post("/api/habitat", json={"id": "h2", "seed": 4})
    g = client.get("/api/habitat/h2/chemistry").json()
    assert "t_gyr" in g and "species" in g and "patches" in g
    assert len(g["patches"]) == 8
    for p in g["patches"]:
        assert "index" in p and "c" in p and "temperature_K" in p
        assert "pH" in p and "liquid_water" in p
        assert "nutrient_mol" in p and "waste_mol" in p


def test_habitat_population_endpoint(client):
    client.post("/api/habitat", json={"id": "h3", "seed": 5})
    g = client.get("/api/habitat/h3/population").json()
    assert "sterile" in g and "replicators" in g
    assert "pool_monomer" in g and "total_replicators" in g
    assert g["sterile"] is True


def test_habitat_lineage_endpoint(client):
    client.post("/api/habitat", json={"id": "h4", "seed": 6})
    g = client.get("/api/habitat/h4/lineage").json()
    assert "t_gyr" in g and "lineages" in g and "n_lineages" in g


def test_habitat_resources_endpoint(client):
    client.post("/api/habitat", json={"id": "h5", "seed": 7})
    g = client.get("/api/habitat/h5/resources").json()
    assert "t_gyr" in g and "pool_monomer" in g
    assert "patches" in g and "mass_ledger_error" in g


def test_habitat_bad_id_rejected(client):
    assert client.get("/api/habitat/nonexistent-id/chemistry").status_code == 404
    assert client.post("/api/habitat", json={"id": "../etc/passwd"}).status_code == 422
    assert client.post("/api/habitat", json={"id": "a b"}).status_code == 422


def test_habitat_duplicate_id_rejected(client):
    client.post("/api/habitat", json={"id": "dup", "seed": 1})
    r = client.post("/api/habitat", json={"id": "dup", "seed": 2})
    assert r.status_code == 422
