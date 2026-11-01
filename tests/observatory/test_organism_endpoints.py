"""Observatory endpoints for the persistent organism + Mind Inspector (Sprint 13 Task 6).

Verifies the organism can be created, stepped, inspected (Mind Inspector
inspector_state), and checkpointed through the actual observatory API.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from universe_engine.observatory.server import create_app


def _client() -> TestClient:
    return TestClient(create_app())


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
