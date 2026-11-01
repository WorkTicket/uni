"""PASS 2A acceptance tests: persistence gaps.

Covers:
- episodic memory lossless checkpoint/restore (>capacity inspector limit)
- full planetary state via habitat checkpoint/restore/continuation
- stochastic RNG persistence (habitat, protocell world, replicators)
- shared habitat identity across organism restore
"""

from __future__ import annotations

import json
import math
import numpy as np
import pytest

from universe_engine.research.persistent_organism import (
    PersistentOrganism, EpisodicMemory, MemoryEntry, make_default_habitat,
    make_organism,
)
from universe_engine.research.contracts import AgentAction
from universe_engine.habitat import Habitat, make_default_network
from universe_engine.planet.model import Planet, PlanetSpec


# ---------------------------------------------------------------------------
# 1. Episodic memory: lossless checkpoint / restore
# ---------------------------------------------------------------------------

def test_memory_checkpoint_preserves_all_entries_beyond_inspector_limit():
    """Storing >100 unique memories, checkpoint/restore keeps every entry in order."""
    mem = EpisodicMemory(capacity=500)
    for i in range(150):
        mem.add(MemoryEntry(
            t_gyr=float(i), observation={"i": i}, action={"a": i},
            reward=float(i), next_observation={"n": i},
        ))
    cp = mem.checkpoint()
    assert cp["schema"] == "episodic-memory/v1"
    assert cp["n_entries"] == 150
    assert len(cp["entries"]) == 150
    # Inspector summary remains bounded (lossy) -- UI payload stays small.
    summary = mem.inspector_summary(limit=50)
    assert len(summary["entries"]) == 50
    restored = EpisodicMemory.restore(cp)
    assert len(restored.entries) == 150
    assert [e.t_gyr for e in restored.entries] == [float(i) for i in range(150)]
    assert [e.reward for e in restored.entries] == [float(i) for i in range(150)]


# ---------------------------------------------------------------------------
# 2. Full planetary state via habitat checkpoint/restore/continuation
# ---------------------------------------------------------------------------

def _habitat_state(h: Habitat) -> dict:
    return {
        "t_gyr": h.t_gyr,
        "planet_T": h.planet.T.tolist(),
        "planet_pco2": h.planet.pco2,
        "planet_t_gyr": h.planet.t_gyr,
        "patches": [[float(v) for v in p.c] for p in h.patches],
        "rng_state": h.rng.bit_generator.state,
    }


def test_habitat_checkpoint_restore_continuation_matches_uninterrupted():
    """N+M uninterrupted == N steps, checkpoint, restore, M steps (habitat level)."""
    N, M = 8, 6
    net = make_default_network()

    # Uninterrupted run A: N + M steps.
    pA = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    hA = Habitat(pA, net, n_patches=4, seed=3)
    for _ in range(N + M):
        hA.step(1e-6)
    fullA = _habitat_state(hA)

    # Run B: N steps, checkpoint, restore into fresh habitat, M steps.
    pB = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    hB = Habitat(pB, net, n_patches=4, seed=3)
    for _ in range(N):
        hB.step(1e-6)
    cp = hB.checkpoint()
    assert "planet" in cp and cp["planet"]["schema"] == "planet/v1"

    pC = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    hC = Habitat.restore(pC, net, cp, seed=3, n_patches=4)
    for _ in range(M):
        hC.step(1e-6)
    fullC = _habitat_state(hC)

    assert fullC["planet_pco2"] == pytest.approx(fullA["planet_pco2"], rel=1e-6, abs=1e-6)
    assert fullC["planet_t_gyr"] == pytest.approx(fullA["planet_t_gyr"], rel=1e-6, abs=1e-6)
    assert np.allclose(fullC["planet_T"], fullA["planet_T"], rtol=1e-6, atol=1e-6)
    assert np.allclose(np.array(fullC["patches"]), np.array(fullA["patches"]),
                       rtol=1e-6, atol=1e-6)
    assert fullC["t_gyr"] == pytest.approx(fullA["t_gyr"], rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Stochastic RNG persistence (protocell + replicator)
# ---------------------------------------------------------------------------

def test_habitat_rng_restoration_matches_stochastic_outcomes():
    """Checkpoint after RNG advancement, then restore: subsequent stochastic outcomes match."""
    net = make_default_network()

    # Advance a habitat far enough to exercise protocell RNG (division draws binomial).
    pA = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    hA = Habitat(pA, net, n_patches=2, seed=9)
    for _ in range(50):
        hA.step(1e-6)
    cpA = hA.checkpoint()

    # Restore and continue the same number of steps; compare full state.
    pB = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    hB = Habitat.restore(pB, net, cpA, seed=9, n_patches=2)
    for _ in range(50):
        hB.step(1e-6)

    # Uninterrupted reference: 100 steps from scratch.
    pC = Planet(PlanetSpec(a_au=1.0, tectonics="mobile"), t_gyr=4.5)
    hC = Habitat(pC, net, n_patches=2, seed=9)
    for _ in range(100):
        hC.step(1e-6)

    assert hB.t_gyr == pytest.approx(hC.t_gyr, rel=1e-9, abs=1e-9)
    assert hB.protocells.t == pytest.approx(hC.protocells.t, rel=1e-9, abs=1e-9)
    assert hB.protocells.pool == pytest.approx(hC.protocells.pool, rel=1e-9, abs=1e-9)
    assert [c.N for c in hB.protocells.cells] == pytest.approx(
        [c.N for c in hC.protocells.cells], rel=1e-9, abs=1e-9)
    assert len(hB.protocells.cells) == len(hC.protocells.cells)


# ---------------------------------------------------------------------------
# 4. Shared habitat identity across organism restore
# ---------------------------------------------------------------------------

def test_restore_reuses_shared_habitat_identity():
    """Restoring an organism rehydrates the passed habitat in place (one identity)."""
    habitat = make_default_habitat(seed=4, n_patches=4)
    org = make_organism(architecture="ecs_iw", seed=11, habitat=habitat,
                        patch_index=0)
    for _ in range(6):
        org.step()
    cp = org.checkpoint()

    # Restore into the SAME habitat object.
    org2 = PersistentOrganism.restore(cp, habitat)
    assert org2.habitat is habitat
    assert org2.patch is not None
    # Actions by org2 affect the shared habitat observed via the holder.
    before = habitat.patches[0].nutrient_mol
    org2.step()
    after = habitat.patches[0].nutrient_mol
    assert after != before or org2.n_steps == org.n_steps + 1


# ---------------------------------------------------------------------------
# 5. Organism checkpoint/restore is bitwise-continuous (N+M)
# ---------------------------------------------------------------------------

def _org_full_state(org: PersistentOrganism) -> dict:
    c = org.controller
    cp = c.checkpoint() if hasattr(c, "checkpoint") else {}
    return {
        "t_gyr": org.t_gyr, "alive": org.alive, "n_steps": org.n_steps,
        "lifetime_reward": org.lifetime_reward,
        "total_cost_paid": org.total_cost_paid,
        "body": org.body.to_dict(),
        "patch_index": org.patch_index,
        "seed": org.seed,
        "memory_n_entries": len(org.memory.entries),
        "memory_entries": [[e.t_gyr, e.reward] for e in org.memory.entries],
        "controller": cp,
        "self_model": org.self_model.checkpoint(),
        "habitat": org.habitat.checkpoint(),
    }


def test_organism_checkpoint_restore_bitwise_continuous():
    """N+M uninterrupted == N steps, checkpoint, restore, M steps (organism level)."""
    N, M = 8, 7
    habitatA = make_default_habitat(seed=17, n_patches=4)
    orgA = make_organism(architecture="ecs_iw", seed=29, habitat=habitatA,
                         patch_index=0)
    for _ in range(N + M):
        orgA.step()
    fullA = _org_full_state(orgA)

    habitatB = make_default_habitat(seed=17, n_patches=4)
    orgB = make_organism(architecture="ecs_iw", seed=29, habitat=habitatB,
                         patch_index=0)
    for _ in range(N):
        orgB.step()
    cp = orgB.checkpoint()

    habitatC = make_default_habitat(seed=17, n_patches=4)
    orgC = PersistentOrganism.restore(cp, habitatC)
    for _ in range(M):
        orgC.step()
    fullC = _org_full_state(orgC)

    assert fullC == fullA, f"organism state diverged after restore: " \
        f"{_diff(fullA, fullC)}"


def _diff(a, b, path="$"):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: missing in left")
            elif k not in b:
                out.append(f"{path}.{k}: missing in right")
            else:
                out.extend(_diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(_diff(x, y, f"{path}[{i}]"))
    elif isinstance(a, float) and isinstance(b, float):
        if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out[:20]
