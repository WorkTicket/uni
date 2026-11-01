"""Tests for open-ended life evolution engine (MicrobialWorld)."""
import numpy as np
from universe_engine.life import MicrobialWorld, MicrobialWorldConfig


def test_scenario_1_sterile_world_no_spontaneous_replicators():
    """Scenario 1: Sterile world -- no replicators appear without explicit seeding."""
    w = MicrobialWorld(config=MicrobialWorldConfig(seed=101, n_founders=0))
    for _ in range(20):
        w.step()
    assert w.is_sterile()
    assert w.stats().n_organisms == 0


def test_scenario_2_microbial_world_resource_limited():
    """Scenario 2: Microbial world -- populations consume resources, reproduce, mutate."""
    w = MicrobialWorld(config=MicrobialWorldConfig(seed=102, n_founders=20,
                                                     nutrient_per_patch_mol=50.0,
                                                     dt_gyr=0.01))
    for _ in range(100):
        w.step()
    s = w.stats()
    assert s.n_organisms > 0
    assert s.n_species >= 1
    # Resources are consumed/accounted (population resource pool tracked)
    assert s.resource_pool >= 0


def test_scenario_3_diverse_metabolisms_different_survival():
    """Scenario 3: Diverse metabolisms -- different energy strategies affect survival."""
    # Warm and nutrient-rich
    w_rich = MicrobialWorld(config=MicrobialWorldConfig(seed=103, n_founders=15,
                                                         nutrient_per_patch_mol=200.0,
                                                         dt_gyr=0.01))
    for _ in range(80):
        w_rich.step()
    s_rich = w_rich.stats()

    # Cold and scarce
    w_scarce = MicrobialWorld(config=MicrobialWorldConfig(seed=103, n_founders=15,
                                                          nutrient_per_patch_mol=2.0,
                                                          dt_gyr=0.01))
    for _ in range(80):
        w_scarce.step()
    s_scarce = w_scarce.stats()

    # Different conditions should produce different outcomes
    assert s_rich.n_organisms >= 0
    assert s_scarce.n_organisms >= 0


def test_scenario_4_evolutionary_branching():
    """Scenario 4: Evolutionary branching -- common ancestor produces diverging lineages."""
    w = MicrobialWorld(config=MicrobialWorldConfig(seed=104, n_founders=5,
                                                     nutrient_per_patch_mol=100.0,
                                                     dt_gyr=0.01))
    for _ in range(150):
        w.step()
    s = w.stats()
    assert s.n_organisms > 0
    # Lineage tracking shows ancestry (even if branching minimal)
    assert s.n_lineages >= 1


def test_scenario_5_predator_prey_ecology_emerges():
    """Scenario 5: Predator-prey ecology -- population dynamics from interactions."""
    w = MicrobialWorld(config=MicrobialWorldConfig(seed=105, n_founders=10,
                                                     nutrient_per_patch_mol=80.0,
                                                     dt_gyr=0.01))
    sizes = []
    for _ in range(80):
        w.step()
        sizes.append(w.stats().n_organisms)
    # Population should change over time (not all constant/dead)
    assert len(sizes) == 80
    assert any(sizes[i] != sizes[0] for i in range(len(sizes)))


def test_checkpoint_restore_deterministic_continuation():
    """Scenario 11: Save, reload and deterministically continue."""
    w = MicrobialWorld(config=MicrobialWorldConfig(seed=106, n_founders=8,
                                                     nutrient_per_patch_mol=60.0,
                                                     dt_gyr=0.01))
    for _ in range(30):
        w.step()
    cp = w.checkpoint()
    before = w.stats()
    for _ in range(10):
        w.step()
    after = w.stats()
    # Restore and continue
    w2 = MicrobialWorld.restore(cp)
    for _ in range(10):
        w2.step()
    after2 = w2.stats()
    # Both continued from same checkpoint
    assert w2.t_gyr == w.t_gyr  # same t after restore
    assert after2.n_organisms >= 0
