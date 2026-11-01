"""Observatory endpoints for the persistent organism + Mind Inspector (Sprint 13 Task 6).

Verifies the organism can be created, stepped, inspected (Mind Inspector
inspector_state), and checkpointed through the actual observatory API.
"""
from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from universe_engine.observatory.server import create_app


def _client(save_dir="runs/observatory"):
    from universe_engine.observatory.server import create_app
    from fastapi.testclient import TestClient
    return TestClient(create_app(save_dir=save_dir))


def test_organism_create_step_inspect_checkpoint():
    c = _client()
    assert c.post("/api/habitat", json={"id": "h1", "seed": 3, "n_patches": 4}).status_code == 200
    r = c.post("/api/organism", json={"id": "o1", "habitat_id": "h1",
                                      "architecture": "ecs_iw", "seed": 7,
                                      "patch_index": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "o1"
    state = body["state"]
    assert state["alive"] is True
    assert state["controller_type"] == "ecs_iw"
    assert state["identity"]["architecture"] == "ecs_iw"

    # Step the organism.
    r = c.post("/api/organism/o1/step", json={"n": 5})
    assert r.status_code == 200
    assert r.json()["steps"] == 5
    assert r.json()["state"]["n_steps"] == 5

    # Mind Inspector view.
    r = c.get("/api/organism/o1/inspect")
    assert r.status_code == 200
    insp = r.json()
    for key in ("identity", "body", "patch", "controller", "self_model",
                "memory", "developmental_history"):
        assert key in insp, key

    # Checkpoint.
    r = c.post("/api/organism/o1/checkpoint")
    assert r.status_code == 200
    assert r.json()["checkpoint"]["schema"] == "persistent-organism/v1"

    # Organism GET.
    r = c.get("/api/organism/o1")
    assert r.status_code == 200
    assert r.json()["controller_type"] == "ecs_iw"


def test_organism_missing_404():
    c = _client()
    assert c.get("/api/organism/nope").status_code == 404
    assert c.post("/api/organism/nope/step").status_code == 404
    assert c.get("/api/organism/nope/inspect").status_code == 404


def test_organism_without_habitat_creates_default():
    c = _client()
    r = c.post("/api/organism", json={"id": "o2", "architecture": "reactive",
                                      "seed": 1})
    assert r.status_code == 200
    assert r.json()["state"]["controller_type"] == "reactive"


def test_organism_save_load_survives_process_restart(tmp_path):
    """Create organism -> advance -> save -> new app (restart) -> load -> inspect -> continue."""
    # --- First "process": create, advance, save ---
    c1 = _client(save_dir=str(tmp_path))
    assert c1.post("/api/habitat", json={"id": "h1", "seed": 3, "n_patches": 4}).status_code == 200
    r = c1.post("/api/organism", json={"id": "o1", "habitat_id": "h1",
                                       "architecture": "ecs_iw", "seed": 7,
                                       "patch_index": 0})
    assert r.status_code == 200
    r = c1.post("/api/organism/o1/step", json={"n": 6})
    assert r.status_code == 200
    assert r.json()["state"]["n_steps"] == 6
    pre = c1.get("/api/organism/o1/inspect").json()
    assert pre["alive"] is True

    r = c1.post("/api/organism/o1/save", json={"id": "cp-o1"})
    assert r.status_code == 200
    assert r.json()["id"] == "cp-o1"

    # --- Second "process": fresh app, same disk -> load -> inspect -> continue ---
    c2 = _client(save_dir=str(tmp_path))
    r = c2.post("/api/organism/load", json={"id": "cp-o1", "organism_id": "o1"})
    assert r.status_code == 200, r.text
    loaded = r.json()
    assert loaded["id"] == "o1"
    state = loaded["state"]
    assert state["n_steps"] == 6
    assert state["alive"] is True
    assert state["t_gyr"] == pytest.approx(pre["t_gyr"], rel=1e-9, abs=1e-9)
    assert state["body"]["energy"] == pytest.approx(pre["body"]["energy"],
                                                   rel=1e-9, abs=1e-9)

    # Continue the same organism; scientific state continues from the save point.
    r = c2.post("/api/organism/o1/step", json={"n": 4})
    assert r.status_code == 200
    assert r.json()["state"]["n_steps"] == 10

    # Saves are listable.
    r = c2.get("/api/organism/saves")
    assert r.status_code == 200
    assert "cp-o1" in r.json()["saves"]
