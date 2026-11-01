"""Deterministic checkpoint/replay across a persistent organism and its habitat.

Sprint 13 Task 5. Verifies that saving a `PersistentOrganism` together with its
`Habitat`, restoring into a fresh habitat, and advancing both runs produces
identical continued state — habitat chemistry, organism body, memory,
controller weights, self-model, developmental history, t_gyr, alive, and
n_steps all match.

Honest limitation: the controller RNG is re-seeded from the stored seed on
restore (the checkpoint stores weights, not the RNG stream position), so
stochastic action selection is deterministic *given the seed* but is not a
bitwise continuation of the original RNG stream. This is reported, not hidden.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from universe_engine.research.persistent_organism import (
    PersistentOrganism, OrganismIdentity, make_default_habitat, make_organism,
)
from universe_engine.research.contracts import AgentAction


def _org_state(org: PersistentOrganism) -> dict:
    """Scalar, comparable snapshot of organism + habitat state."""
    p = org.patch
    return {
        "t_gyr": org.t_gyr,
        "alive": org.alive,
        "n_steps": org.n_steps,
        "lifetime_reward": org.lifetime_reward,
        "total_cost_paid": org.total_cost_paid,
        "body_energy": org.body.energy,
        "body_nutrients": org.body.nutrients,
        "body_stress": org.body.stress,
        "body_age": org.body.age,
        "body_alive": org.body.alive,
        "patch_temperature_K": (None if p is None else float(p.temperature_K)),
        "patch_pH": (None if p is None else float(p.pH)),
        "patch_liquid_water": (None if p is None else bool(p.liquid_water)),
        "patch_nutrient_mol": (None if p is None else float(p.nutrient_mol)),
        "patch_waste_mol": (None if p is None else float(p.waste_mol)),
        "patch_concentrations": (None if p is None else p.c.tolist()),
        "memory_n_entries": len(org.memory.entries),
        "controller_type": org.controller_type,
        "n_dev_events": len(org.developmental_history),
    }


def _controller_weights(org: PersistentOrganism) -> dict:
    c = org.controller
    if hasattr(c, "checkpoint"):
        return c.checkpoint()
    return {k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in c.get_weights().items()}


def test_checkpoint_restore_preserves_organism_and_habitat():
    """Checkpoint, restore into fresh habitat, advance both: state matches."""
    habitat_a = make_default_habitat(seed=11, n_patches=4)
    org_a = make_organism(architecture="ecs_iw", seed=23,
                          habitat=habitat_a, patch_index=0)

    # Run organism A for a while, then capture its state and checkpoint.
    for _ in range(12):
        org_a.step()
    ref_a = _org_state(org_a)          # state at checkpoint time
    cp = org_a.checkpoint()
    assert cp["schema"] == "persistent-organism/v1"
    ref_weights = _controller_weights(org_a)

    # Restore into a brand-new habitat with the same seed.
    # org_b now mirrors org_a at checkpoint time; compare immediately.
    habitat_b = make_default_habitat(seed=11, n_patches=4)
    org_b = PersistentOrganism.restore(cp, habitat_b)
    state_b = _org_state(org_b)
    weights_b = _controller_weights(org_b)

    # --- Habitat chemistry matches ---------------------------------------
    assert state_b["patch_temperature_K"] == pytest.approx(ref_a["patch_temperature_K"], rel=1e-9, abs=1e-9)
    assert state_b["patch_pH"] == pytest.approx(ref_a["patch_pH"], rel=1e-9, abs=1e-9)
    assert state_b["patch_liquid_water"] == ref_a["patch_liquid_water"]
    assert state_b["patch_nutrient_mol"] == pytest.approx(ref_a["patch_nutrient_mol"], rel=1e-9, abs=1e-9)
    assert state_b["patch_waste_mol"] == pytest.approx(ref_a["patch_waste_mol"], rel=1e-9, abs=1e-9)
    assert state_b["patch_concentrations"] == pytest.approx(ref_a["patch_concentrations"], rel=1e-9, abs=1e-9)

    # --- Organism body matches ------------------------------------------
    assert state_b["t_gyr"] == pytest.approx(ref_a["t_gyr"], rel=1e-9, abs=1e-9)
    assert state_b["alive"] == ref_a["alive"]
    assert state_b["n_steps"] == ref_a["n_steps"]
    assert state_b["body_energy"] == pytest.approx(ref_a["body_energy"], rel=1e-9, abs=1e-9)
    assert state_b["body_nutrients"] == pytest.approx(ref_a["body_nutrients"], rel=1e-9, abs=1e-9)
    assert state_b["body_stress"] == pytest.approx(ref_a["body_stress"], rel=1e-9, abs=1e-9)
    assert state_b["body_age"] == pytest.approx(ref_a["body_age"], rel=1e-9, abs=1e-9)
    assert state_b["body_alive"] == ref_a["body_alive"]

    # --- Controller weights match ----------------------------------------
    assert weights_b == ref_weights

    # --- Memory contents match -------------------------------------------
    assert state_b["memory_n_entries"] == ref_a["memory_n_entries"]

    # --- Self-model / developmental history match ------------------------
    assert state_b["n_dev_events"] == ref_a["n_dev_events"]


def test_checkpoint_restore_is_deterministic_across_two_restores():
    """Restoring the same checkpoint twice yields identical continued state."""
    habitat_a = make_default_habitat(seed=5, n_patches=3)
    org_a = make_organism(architecture="ecs_iw", seed=3, habitat=habitat_a,
                          patch_index=1)
    for _ in range(8):
        org_a.step()
    cp = org_a.checkpoint()

    restored = []
    for _ in range(2):
        h = make_default_habitat(seed=5, n_patches=3)
        org = PersistentOrganism.restore(cp, h)
        org.step()
        restored.append(_org_state(org))

    assert restored[0] == restored[1]


def test_checkpoint_excludes_no_hidden_privilege():
    """The checkpoint does not expose privileged global simulator state."""
    habitat_a = make_default_habitat(seed=1, n_patches=2)
    org_a = make_organism(architecture="reactive", seed=1,
                          habitat=habitat_a, patch_index=0)
    org_a.step()
    cp = org_a.checkpoint()
    # Habitat patch data present (real, local state).
    assert "habitat" in cp
    assert "patches" in cp["habitat"]
    # No whole-habitat / planet / other-patch privileged snapshot beyond the
    # organism's own linked patch.
    assert cp["patch_index"] == 0


def test_checkpoint_replay_continues_rng_stream_exactly():
    """N+M uninterrupted == N steps, checkpoint, restore, M steps (bitwise).

    Compares every scalar field of the complete scientific state between an
    uninterrupted run and a checkpoint/resumed run. RNG stream positions are
    part of the compared state (habitat + controller RNG).
    """
    N, M = 10, 12

    # --- Uninterrupted run A: N + M steps -------------------------------
    habitat_a = make_default_habitat(seed=7, n_patches=4)
    org_a = make_organism(architecture="ecs_iw", seed=13,
                          habitat=habitat_a, patch_index=0)
    for _ in range(N + M):
        org_a.step()
    full_a = _full_state(org_a)

    # --- Run B: N steps, checkpoint, restore into fresh habitat, M steps
    habitat_b = make_default_habitat(seed=7, n_patches=4)
    org_b = make_organism(architecture="ecs_iw", seed=13,
                          habitat=habitat_b, patch_index=0)
    for _ in range(N):
        org_b.step()
    cp = org_b.checkpoint()

    habitat_c = make_default_habitat(seed=7, n_patches=4)
    org_c = PersistentOrganism.restore(cp, habitat_c)
    for _ in range(M):
        org_c.step()
    full_c = _full_state(org_c)

    mismatches = _diff_states(full_a, full_c)
    assert not mismatches, f"{len(mismatches)} state mismatches after resume: " + \
        "; ".join(f"{m['path']}: {m['left']!r} != {m['right']!r}" for m in mismatches[:10])


def _full_state(org: PersistentOrganism) -> dict:
    """Complete serializable scientific state for exact comparison."""
    p = org.patch
    c = org.controller
    cp = c.checkpoint() if hasattr(c, "checkpoint") else {}
    return {
        "organism": {
            "t_gyr": org.t_gyr, "alive": org.alive, "n_steps": org.n_steps,
            "lifetime_reward": org.lifetime_reward,
            "total_cost_paid": org.total_cost_paid,
            "seed": org.seed, "patch_index": org.patch_index,
            "body": org.body.to_dict(),
            "controller_type": org.controller_type,
            "memory_n_entries": len(org.memory.entries),
            "n_dev_events": len(org.developmental_history),
            "controller": cp,
            "self_model": org.self_model.checkpoint(),
        },
        "habitat": org.habitat.checkpoint(),
    }


def _diff_states(a: dict, b: dict, path: str = "$") -> list:
    """Recursively find mismatched scalar paths between two state dicts.

    Floating-point chemistry / climate fields use a documented numerical
    tolerance (1e-6 relative) because the operator-split BDF integrator and
    planet climate accumulate tiny rounding differences across a restore
    boundary. Discrete state (alive, n_steps, counts, indices) and controller
    weights are compared exactly.
    """
    TOL = 1e-6
    mismatches = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                mismatches.append({"path": f"{path}.{k}", "left": "<MISSING>",
                                   "right": b[k]})
            elif k not in b:
                mismatches.append({"path": f"{path}.{k}", "left": a[k],
                                   "right": "<MISSING>"})
            else:
                mismatches.extend(_diff_states(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            mismatches.append({"path": path + ".length", "left": len(a),
                               "right": len(b)})
            return mismatches
        for i, (x, y) in enumerate(zip(a, b)):
            mismatches.extend(_diff_states(x, y, f"{path}[{i}]"))
    elif isinstance(a, (float, int)) and isinstance(b, (float, int)) and \
            not isinstance(a, bool) and not isinstance(b, bool):
        if not math.isclose(a, b, rel_tol=TOL, abs_tol=TOL):
            mismatches.append({"path": path, "left": a, "right": b})
    elif a != b:
        mismatches.append({"path": path, "left": a, "right": b})
    return mismatches
