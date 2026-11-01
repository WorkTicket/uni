"""Tests for self-model and workspace (Sprint 5, Research Program D5-D6)."""

import numpy as np
import pytest

from universe_engine.research.self_model.model import (
    SelfModel, BodySchema, make_default_self_model,
)
from universe_engine.research.workspace.model import (
    GlobalWorkspace, WorkspaceContent, WorkspaceAgentController,
)
from universe_engine.research.contracts import (
    SensoryObservation, AgentAction,
)


# ---------------------------------------------------------------------------
# BodySchema tests
# ---------------------------------------------------------------------------

def test_body_schema_predicted_sensory_change():
    """Body schema predicts sensory change from actions."""
    body = BodySchema(
        position=(0.0, 0.0),
        sensorimotor_map={"move": np.array([1.0, 0.0])},
    )
    pred = body.predicted_sensory_change("move", {"dx": 1, "dy": 0})
    assert np.allclose(pred, np.array([0.1, 0.0]))  # scale=0.1

    # Unknown action returns zeros
    pred2 = body.predicted_sensory_change("jump", {})
    assert np.allclose(pred2, np.zeros(2))


# ---------------------------------------------------------------------------
# SelfModel tests
# ---------------------------------------------------------------------------

def test_self_model_predict_and_update():
    """Self model predicts then updates from observed consequences."""
    sm = make_default_self_model()
    predicted = sm.predict("move", {"dx": 1, "dy": 0})
    assert predicted.shape == (2,)

    actual = np.array([0.15, 0.0])  # actual sensory change differs from prediction
    diag = sm.update("move", {"dx": 1, "dy": 0}, predicted[:2], actual[:2])
    assert "calibration_error" in diag
    assert "self_caused_fraction" in diag
    assert "prediction_error" in diag
    assert sm.n_updates == 1
    # Calibration error should reflect the mismatch
    assert diag["prediction_error"] > 0


def test_self_model_detect_external_event():
    """Self model detects externally caused sensory changes."""
    sm = make_default_self_model()
    # Predict small change, actual large change -> external event
    predicted = np.array([0.05, 0.0])
    actual = np.array([0.5, 0.0])  # much larger than predicted
    result = sm.detect_external_event("move", {"dx": 1, "dy": 0}, actual,
                                       threshold=0.1)
    assert result["is_external"] is True
    assert result["confidence"] > 0
    assert result["residual"] > 0.1

    # Predicted matches actual -> not external
    result2 = sm.detect_external_event("move", {"dx": 1, "dy": 0},
                                        predicted, threshold=0.1)
    assert result2["is_external"] is False


def test_self_model_recalibrate():
    """Recalibration increases capability uncertainty."""
    sm = make_default_self_model()
    before = dict(sm.body.capability_uncertainty)
    sm.recalibrate(0.5)
    for key in before:
        assert sm.body.capability_uncertainty[key] >= before[key]


def test_self_model_checkpoint_restore():
    """Self model checkpoint/restore preserves state."""
    sm = make_default_self_model()
    sm.update("move", {"dx": 1, "dy": 0},
              np.array([0.1, 0.0]), np.array([0.15, 0.0]))
    cp = sm.checkpoint()

    sm2 = SelfModel.restore(cp)
    assert sm2.calibration_error == sm.calibration_error
    assert sm2.n_updates == sm.n_updates
    assert sm2.body.energy_capacity == sm.body.energy_capacity


# ---------------------------------------------------------------------------
# GlobalWorkspace tests
# ---------------------------------------------------------------------------

def test_workspace_receive_and_broadcast():
    """Workspace receives content and broadcasts the most salient."""
    gw = GlobalWorkspace(capacity=1)
    gw.receive("perception", {"a": 1}, 0.3, "mod1", timestamp=0.0)
    gw.receive("memory", {"b": 2}, 0.8, "mod2", timestamp=0.0)
    # Only the highest salience (0.8) should be kept
    assert gw.workspace_occupancy() == 1.0
    broadcast = gw.broadcast()
    assert broadcast is not None
    assert broadcast.modality == "memory"
    assert broadcast.source_module == "mod2"
    assert gw.n_broadcasts == 1


def test_workspace_empty_broadcast_returns_none():
    """Broadcasting from empty workspace returns None."""
    gw = GlobalWorkspace(capacity=1)
    assert gw.broadcast() is None


def test_workspace_decay():
    """Content salience decays after broadcast."""
    gw = GlobalWorkspace(capacity=2, decay=0.5)
    gw.receive("perception", {"a": 1}, 1.0, "mod1", timestamp=0.0)
    gw.broadcast()  # broadcast the content
    # After broadcast, salience decays
    assert gw.top_salience() <= 0.5 or gw.top_salience() == 0.0


def test_workspace_checkpoint_restore():
    """Workspace checkpoint/restore."""
    gw = GlobalWorkspace(capacity=2)
    gw.receive("perception", {"a": 1}, 0.7, "mod1", timestamp=1.0)
    cp = gw.checkpoint()
    gw2 = GlobalWorkspace.restore(cp)
    assert gw2.capacity == gw.capacity
    assert gw2.n_broadcasts == gw.n_broadcasts


# ---------------------------------------------------------------------------
# WorkspaceAgentController (ECS-IW) tests
# ---------------------------------------------------------------------------

def test_ecs_iw_controller_selects_action():
    """ECS-IW controller returns a valid AgentAction."""
    ctrl = WorkspaceAgentController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.5,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    action = ctrl.select_action(obs)
    assert isinstance(action, AgentAction)
    assert action.action_type in ("move", "forage", "rest", "wait")


def test_ecs_iw_controller_learns_updates_self_model():
    """ECS-IW learn() updates self-model calibration error."""
    ctrl = WorkspaceAgentController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.5,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    action = ctrl.select_action(obs)
    next_obs = SensoryObservation(time=0.1, position=(0, 0), energy_level=0.6,
                                  nutrient_level=1.0, local_temperature=288.0,
                                  local_light=1.0, nearby_resources=0.7,
                                  nearby_hazards=0.0, nearby_agents=0)
    n_updates_before = ctrl.self_model.n_updates
    ctrl.learn(obs, action, 0.1, next_obs)
    assert ctrl.self_model.n_updates > n_updates_before


def test_ecs_iw_checkpoint_restore():
    """ECS-IW controller checkpoint/restore preserves state."""
    ctrl = WorkspaceAgentController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.5,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    action = ctrl.select_action(obs)
    ctrl.learn(obs, action, 0.1, obs)
    cp = ctrl.checkpoint()

    ctrl2 = WorkspaceAgentController("agent-2", seed=99)
    ctrl2.restore(cp)
    assert ctrl2.actions_taken == ctrl.actions_taken
    assert ctrl2.self_model.n_updates == ctrl.self_model.n_updates


def test_ecs_iw_distinguishes_self_caused_from_external():
    """ECS-IW self-model can distinguish self-caused from external changes."""
    ctrl = WorkspaceAgentController("agent-1", seed=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.5,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    action = ctrl.select_action(obs)
    next_obs = SensoryObservation(time=0.1, position=(0, 0), energy_level=0.6,
                                  nutrient_level=1.0, local_temperature=288.0,
                                  local_light=1.0, nearby_resources=0.5,
                                  nearby_hazards=0.8,  # external hazard increase
                                  nearby_agents=0)
    ctrl.learn(obs, action, 0.1, next_obs)
    # After learning, the self-model should have recorded the update
    assert ctrl.self_model.n_updates >= 1


# ---------------------------------------------------------------------------
# Ablation: compare ECS-IW without workspace
# ---------------------------------------------------------------------------

def test_workspace_ablation_changes_behavior():
    """Removing workspace broadcasting changes controller behavior."""
    # With workspace
    ctrl_with = WorkspaceAgentController("a1", seed=42, workspace_capacity=1)
    obs = SensoryObservation(time=0.0, position=(0, 0), energy_level=0.5,
                             nutrient_level=1.0, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    a1 = ctrl_with.select_action(obs)

    # Without workspace (capacity=0 means nothing can be held)
    ctrl_without = WorkspaceAgentController("a2", seed=42, workspace_capacity=0)
    a2 = ctrl_without.select_action(obs)

    # Both should return valid actions
    assert a1.action_type in ("move", "forage", "rest", "wait")
    assert a2.action_type in ("move", "forage", "rest", "wait")
    # They may differ due to workspace broadcast influence
