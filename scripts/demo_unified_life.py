"""Reproducible demo: unified planetary life simulation.

Runs the three required scenarios through the shared HabitatState and prints
numerical results. No fake organisms, no procedural viewer, no civilization
systems. All outputs come from the live simulation.

Usage:
    python scripts/demo_unified_life.py
"""
from __future__ import annotations

import json
import time

from universe_engine.habitat import Habitat, make_default_network
from universe_engine.planet.model import Planet, PlanetSpec


def make_planet(t_gyr=4.5, has_water=True, seed=1):
    spec = PlanetSpec(mass_mearth=1.0, a_au=1.0, ocean_fraction=0.71,
                      has_surface_water=has_water, tectonics="mobile", seed=seed)
    return Planet(spec, t_gyr=t_gyr)


def scenario_1_sterile():
    """No replicators seeded -> habitat stays sterile forever."""
    h = Habitat(make_planet(has_water=True, seed=1), make_default_network(),
                n_patches=4, seed=11)
    for _ in range(50):
        h.step(1e-6)
    st = h.state()
    assert st.sterile is True
    assert st.replicators == {}
    assert st.mass_ledger_error < 1e-6
    return {"t_gyr": st.t_gyr, "sterile": st.sterile,
            "replicators": st.replicators,
            "mass_ledger_error": st.mass_ledger_error}


def scenario_2_sustained():
    """Seeded replicators with replenished resources -> population persists."""
    h = Habitat(make_planet(has_water=True, seed=2), make_default_network(),
                n_patches=4, seed=12)
    assert h.seed_replicator("ACGU", count=50, cell_index=0)
    h.replicators.pool = 10**6
    h.protocells.cells[0].N = 10**6
    h.protocells.solute_rate = 0.0
    h.protocells.Pf = 0.0
    h.protocells.lyse = 1.0
    start = h.replicators.total()
    minima = []
    for _ in range(40):
        h.replicators.pool = 10**6
        h.step(1e-10)
        minima.append(h.replicators.total())
    end = h.replicators.total()
    assert end > 0
    assert all(m > 0 for m in minima)
    return {"t_gyr": h.t_gyr, "start": start, "end": end,
            "min_over_run": min(minima), "n_variants": len(h.state().replicators),
            "sterile": h.state().sterile}


def scenario_3_extinction():
    """Replicators exhaust monomers -> population collapses to zero."""
    h = Habitat(make_planet(has_water=True, seed=3), make_default_network(),
                n_patches=4, seed=13)
    assert h.seed_replicator("ACGU", count=1, cell_index=0)
    h.replicators.pool = 0
    h.protocells.cells[0].N = 0
    h.protocells.solute_rate = 0.0
    h.protocells.Pf = 0.0
    h.protocells.lyse = 1.0
    for _ in range(100):
        h.step(1e-10)
    assert h.replicators.total() == 0
    assert h.state().sterile is True
    return {"t_gyr": h.t_gyr, "final_total": h.replicators.total(),
            "sterile": h.state().sterile}


def demo_checkpoint_restore():
    """Save/load restores the complete habitat consistently."""
    h = Habitat(make_planet(has_water=True, seed=4), make_default_network(),
                n_patches=4, seed=19)
    h.seed_replicator("ACGU", count=50, cell_index=0)
    h.replicators.pool = 10**5
    h.protocells.cells[0].N = 10**5
    h.protocells.solute_rate = 0.0
    h.protocells.Pf = 0.0
    h.protocells.lyse = 1.0
    for _ in range(5):
        h.step(0.01)
    before = h.state()
    cp = h.checkpoint()
    h2 = Habitat.restore(make_planet(t_gyr=4.5), make_default_network(), cp, seed=19)
    after = h2.state()
    assert abs(after.t_gyr - before.t_gyr) < 1e-9
    assert after.replicators == before.replicators
    assert abs(after.mass_ledger_error - before.mass_ledger_error) < 1e-6
    return {"restored_t_gyr": after.t_gyr,
            "restored_replicators_match": after.replicators == before.replicators,
            "ledger_error": after.mass_ledger_error}


def main():
    results = {
        "scenario_1_sterile": scenario_1_sterile(),
        "scenario_2_sustained": scenario_2_sustained(),
        "scenario_3_extinction": scenario_3_extinction(),
        "checkpoint_restore": demo_checkpoint_restore(),
    }
    print(json.dumps(results, indent=2))
    assert results["scenario_1_sterile"]["sterile"] is True
    assert results["scenario_2_sustained"]["end"] > 0
    assert results["scenario_3_extinction"]["final_total"] == 0
    assert results["checkpoint_restore"]["restored_replicators_match"] is True
    print("\nAll three scenarios and checkpoint/restore verified.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nElapsed: {time.time() - t0:.3f}s")
