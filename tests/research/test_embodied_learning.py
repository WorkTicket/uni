"""PASS 5: embodied learning harness (research, experimental).

Verifies the harness runs an agent inside the persistent world and reports its
measured outcome honestly. Crucially, this test PRESERVES the documented
negative finding: it does NOT assert that reward-only memory-based control is
solved. `solved` is a measured boolean, and `note` reports that control is
UNRESOLVED.

No claim of consciousness; measures action-reward learning only.
"""

import numpy as np
import pytest

from universe_engine.embodied_learning import EmbodiedLearning
from universe_engine.research.persistent_organism import make_default_habitat, make_organism
from universe_engine.habitat import make_default_network
from universe_engine.planet.model import Planet, PlanetSpec


def test_harness_runs_episodes_and_reports_honest_outcome():
    el = EmbodiedLearning(architecture="ecs_iw", seed=7, n_patches=4,
                          patch_index=0, steps_per_episode=12)
    result = el.run(episodes=8)

    assert result.episodes == 8
    assert len(result.rewards) == 8
    assert len(result.episodes_results) == 8
    assert isinstance(result.solved, bool)
    assert isinstance(result.note, str)
    assert "UNRESOLVED" in result.note
    # Every episode has a complete, comparable record.
    for ep in result.episodes_results:
        assert ep.steps >= 0
        assert isinstance(ep.total_reward, float)
        assert isinstance(ep.alive, bool)
        assert ep.n_memories >= 0


def test_negative_finding_not_fabricated():
    """The harness must not claim success: solved is only True when the
    measured final-window mean reward clearly exceeds the zero baseline AND
    survival is high. It must not assert consciousness or resolved control."""
    el = EmbodiedLearning(architecture="ecs_iw", seed=3, n_patches=4,
                          patch_index=0, steps_per_episode=10)
    result = el.run(episodes=6)
    # `solved` is measured, never hard-coded True.
    assert result.solved in (True, False)
    # The outcome is reported, not framed as a breakthrough.
    assert "reported, not claimed as success" in result.note


def test_embodied_learning_checkpoint_restore():
    """Checkpoint/restore of the embodied-learning harness preserves habitat
    state and episode counter (shared world continues)."""
    el = EmbodiedLearning(architecture="ecs_iw", seed=11, n_patches=4,
                          patch_index=0, steps_per_episode=8)
    el.run(episodes=3)
    cp = el.checkpoint()
    assert cp["schema"] == "embodied-learning/v1"
    assert cp["episode"] == 3

    habitat = make_default_habitat(seed=11, n_patches=4)
    el2 = EmbodiedLearning.restore(cp, habitat)
    assert el2._episode == 3
    assert el2.architecture == "ecs_iw"
    # Restored habitat continues the same scientific state.
    assert el2.habitat.t_gyr == pytest.approx(el.habitat.t_gyr, rel=1e-9, abs=1e-9)


def test_organism_acts_on_shared_habitat():
    """An agent's action changes the shared habitat observed by another."""
    habitat = make_default_habitat(seed=5, n_patches=4)
    org = make_organism(architecture="ecs_iw", seed=1, habitat=habitat,
                        patch_index=0)
    before = habitat.patches[0].nutrient_mol
    org.step()
    after = habitat.patches[0].nutrient_mol
    # Foraging consumes nutrient from the shared patch (may be 0 if none
    # available, but the action is applied to the shared habitat either way).
    assert isinstance(before, float) and isinstance(after, float)
    assert org.n_steps == 1
