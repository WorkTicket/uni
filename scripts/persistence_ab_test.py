"""Persistence A/B test for the persistent organism (Stage 0b.3).

Run A: organism advances uninterrupted for N steps.
Run B: organism advances N/2 steps, checkpoints, is restored into a fresh
habitat, then continues for the remaining N/2 steps.

Full state (identity, body, resources, controller, recurrent/self-model,
memory, developmental history, habitat, RNG-relevant fields, simulation clock)
is compared by canonical serialization + SHA-256 hash. Every mismatching
field is named.

Honest limitation (pre-existing, disclosed): PersistentOrganism.restore()
re-seeds the controller RNG from the stored seed; it does NOT continue the
RNG stream position. So stochastic action selection is deterministic *given
the seed* but is not bitwise continuation after the restore point. This test
reports that mismatch honestly rather than hiding it.

Usage (PowerShell):
  $env:PYTHONPATH="src"; python scripts/persistence_ab_test.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

# Ensure src is importable when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from universe_engine.research.persistent_organism import (
    PersistentOrganism, OrganismIdentity, make_default_habitat, make_organism,
)


# ---------------------------------------------------------------------------
# Canonical serialization helpers
# ---------------------------------------------------------------------------

def canonical(obj) -> str:
    """Deterministic JSON serialization for hashing.

    Floats are normalized via repr to full precision; dicts are key-sorted;
    numpy arrays become nested lists. This produces one canonical byte
    representation per logical state.
    """
    if isinstance(obj, np.ndarray):
        return canonical(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return canonical(float(obj))
    if isinstance(obj, (np.integer,)):
        return canonical(int(obj))
    if isinstance(obj, float):
        return json.dumps(obj, allow_nan=False)
    if isinstance(obj, bool):
        return json.dumps(obj)
    if isinstance(obj, int):
        return json.dumps(obj)
    if isinstance(obj, str):
        return json.dumps(obj)
    if obj is None:
        return "null"
    if isinstance(obj, dict):
        items = sorted((str(k), canonical(v)) for k, v in obj.items())
        return "{" + ",".join(k + ":" + v for k, v in items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(canonical(v) for v in obj) + "}"
    # Fallback for dataclasses / arbitrary objects: use __dict__.
    if hasattr(obj, "__dict__"):
        return canonical(obj.__dict__)
    return json.dumps(str(obj), allow_nan=False)


def sha256_of(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Full state capture
# ---------------------------------------------------------------------------

def full_state(org: PersistentOrganism) -> dict:
    """Capture every comparable field of organism + habitat + controller."""
    p = org.patch
    return {
        "schema": "persistence-ab/v1",
        "identity": org.identity.to_dict(),
        "t_gyr": org.t_gyr,
        "seed": org.seed,
        "patch_index": org.patch_index,
        "alive": org.alive,
        "n_steps": org.n_steps,
        "lifetime_reward": org.lifetime_reward,
        "total_cost_paid": org.total_cost_paid,
        "created_at": org.created_at,
        "body": org.body.to_dict(),
        "controller_type": org.controller_type,
        "memory": org.memory.to_dict(),
        "self_model": {
            "calibration_error": org.self_model.calibration_error,
            "n_updates": org.self_model.n_updates,
            "sensorimotor_map": {
                k: v.tolist() for k, v in org.self_model.body.sensorimotor_map.items()
            },
        },
        "controller": _controller_state(org.controller),
        "developmental_history": [
            {"t_gyr": e.t_gyr, "event": e.event, "details": e.details}
            for e in org.developmental_history
        ],
        "patch": (None if p is None else {
            "index": p.index,
            "nutrient_mol": float(p.nutrient_mol),
            "waste_mol": float(p.waste_mol),
            "temperature_K": float(p.temperature_K),
            "pH": float(p.pH),
            "liquid_water": bool(p.liquid_water),
            "concentrations": p.c.tolist(),
        }),
        "habitat": org.habitat.checkpoint(),
    }


def _controller_state(c) -> dict:
    out = {"controller_type": getattr(c, "controller_type", None)}
    if hasattr(c, "checkpoint"):
        out["checkpoint"] = c.checkpoint()
    if hasattr(c, "get_weights"):
        out["weights"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                          for k, v in c.get_weights().items()}
    if hasattr(c, "actions_taken"):
        out["actions_taken"] = c.actions_taken
    if hasattr(c, "workspace"):
        out["workspace"] = c.workspace.checkpoint()
    if hasattr(c, "self_model"):
        out["self_model_internal"] = c.self_model.checkpoint()
    if hasattr(c, "prediction_error_history"):
        out["recent_prediction_error"] = (
            c.prediction_error_history[-10:]
            if c.prediction_error_history else [])
    return out


# ---------------------------------------------------------------------------
# Recursive field comparison (names mismatching fields)
# ---------------------------------------------------------------------------

def compare_fields(a, b, path="", mismatches=None):
    if mismatches is None:
        mismatches = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(list(a.keys()) + list(b.keys()))):
            p = f"{path}.{k}" if path else k
            if k not in a:
                mismatches.append((p, "MISSING_IN_A", None, b[k]))
            elif k not in b:
                mismatches.append((p, "MISSING_IN_B", a[k], None))
            else:
                compare_fields(a[k], b[k], p, mismatches)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            mismatches.append((path, "LENGTH_MISMATCH", len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                compare_fields(x, y, f"{path}[{i}]", mismatches)
    else:
        a_norm = _norm(a)
        b_norm = _norm(b)
        if a_norm != b_norm:
            mismatches.append((path, "VALUE_DIFFERS", a_norm, b_norm))
    return mismatches


def _norm(x):
    if isinstance(x, float):
        return round(x, 12)
    if isinstance(x, np.floating):
        return round(float(x), 12)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def run_uninterrupted(n_steps: int, seed: int = 23, patch_index: int = 0):
    """Run A: organism advances uninterrupted for n_steps."""
    habitat = make_default_habitat(seed=11, n_patches=4)
    org = make_organism(architecture="ecs_iw", seed=seed,
                        habitat=habitat, patch_index=patch_index)
    for _ in range(n_steps):
        org.step()
    return org


def run_save_reload(n_steps: int, seed: int = 23, patch_index: int = 0):
    """Run B: n/2 steps, checkpoint, restore into fresh habitat, continue n/2."""
    habitat = make_default_habitat(seed=11, n_patches=4)
    org = make_organism(architecture="ecs_iw", seed=seed,
                        habitat=habitat, patch_index=patch_index)
    mid = n_steps // 2
    for _ in range(mid):
        org.step()
    cp = org.checkpoint()

    # Restore into a brand-new habitat with the same seed (mirrors save/close/reload).
    habitat_b = make_default_habitat(seed=11, n_patches=4)
    org_b = PersistentOrganism.restore(cp, habitat_b)
    for _ in range(n_steps - mid):
        org_b.step()
    return org_b


def main():
    n_steps = 40
    seed = 23

    # Fix thread counts for determinism (recorded).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    print("=== Persistence A/B test ===")
    print(f"machine: Windows 11 10.0.26100-SP0")
    print(f"python: {sys.version}")
    print(f"numpy: {np.__version__}")
    print(f"threads: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"steps: {n_steps} (midpoint {n_steps // 2})")
    print(f"seed: {seed}")
    print()

    org_a = run_uninterrupted(n_steps, seed=seed)
    state_a = full_state(org_a)
    hash_a = sha256_of(state_a)

    org_b = run_save_reload(n_steps, seed=seed)
    state_b = full_state(org_b)
    hash_b = sha256_of(state_b)

    print(f"Run A (uninterrupted) final hash: {hash_a}")
    print(f"Run B (save/reload/continue) final hash: {hash_b}")
    print(f"Hashes match: {hash_a == hash_b}")
    print()

    mismatches = compare_fields(state_a, state_b)
    print(f"Mismatching fields: {len(mismatches)}")
    for path, kind, va, vb in mismatches:
        print(f"  {path}: {kind}")
        if kind == "VALUE_DIFFERS":
            print(f"    A={va!r}  B={vb!r}")
        elif kind in ("MISSING_IN_A", "MISSING_IN_B"):
            print(f"    A={va!r}  B={vb!r}")
        else:
            print(f"    A={va!r}  B={vb!r}")
    print()

    # Also compare high-level scalar fields for a readable summary.
    summary = [
        ("t_gyr", state_a["t_gyr"], state_b["t_gyr"]),
        ("alive", state_a["alive"], state_b["alive"]),
        ("n_steps", state_a["n_steps"], state_b["n_steps"]),
        ("lifetime_reward", state_a["lifetime_reward"], state_b["lifetime_reward"]),
        ("total_cost_paid", state_a["total_cost_paid"], state_b["total_cost_paid"]),
        ("body_energy", state_a["body"]["energy"], state_b["body"]["energy"]),
        ("body_nutrients", state_a["body"]["nutrients"], state_b["body"]["nutrients"]),
        ("body_stress", state_a["body"]["stress"], state_b["body"]["stress"]),
        ("body_age", state_a["body"]["age"], state_b["body"]["age"]),
        ("memory_n_entries", len(state_a["memory"]["entries"]), len(state_b["memory"]["entries"])),
        ("n_dev_events", len(state_a["developmental_history"]), len(state_b["developmental_history"])),
    ]
    print("=== Scalar summary ===")
    for name, va, vb in summary:
        flag = "MATCH" if _norm(va) == _norm(vb) else "MISMATCH"
        print(f"  {name}: A={va!r} B={vb!r} -> {flag}")

    result = {
        "n_steps": n_steps,
        "midpoint": n_steps // 2,
        "seed": seed,
        "hash_a": hash_a,
        "hash_b": hash_b,
        "hashes_match": hash_a == hash_b,
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "python": sys.version,
        "numpy": np.__version__,
    }
    return result


if __name__ == "__main__":
    res = main()
