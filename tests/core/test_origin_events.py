"""Tests for model-dependent universe-origin events (Requirement 7 / Test F)."""
import math
import tempfile
import shutil
import numpy as np
import pytest

from universe_engine.multiverse import MultiverseManager
from universe_engine.core.origin_events import (
    ReferenceFLRWOrigin, EkpyroticOrigin, InflationaryBubbleOrigin,
    UniverseOriginRecord, OriginEventManager,
)
from universe_engine.core.universe import OriginEvent


def test_reference_flrw_origin():
    origin = ReferenceFLRWOrigin()
    assert origin.model_id == "flrw_reference"
    assert origin.theory_classification == "reference"
    # Reference FLRW has no transition condition; starts directly.
    assert origin.check_transition(None, 0.0) is None
    state = origin.create_initial_state()
    assert state["era"] == "radiation_dominated"
    assert state["a"] == 1e-13


def test_ekpyrotic_origin_no_transition():
    """Ekpyrotic without valid transition produces no universe."""
    origin = EkpyroticOrigin(transition_valid=False, collision_time=1.0)
    assert origin.theory_classification == "speculative"
    info = origin.check_transition(None, 1.0)
    assert info is not None
    assert info["nonsingular"] is False
    # Initial state should error.
    state = origin.create_initial_state(info)
    assert "error" in state


def test_ekpyrotic_origin_valid_transition():
    """Ekpyrotic with valid transition produces a hot Big Bang state."""
    origin = EkpyroticOrigin(transition_valid=True, collision_time=1.0)
    info = origin.check_transition(None, 1.0)
    assert info["nonsingular"] is True
    state = origin.create_initial_state(info)
    assert "error" not in state
    assert state["era"] == "post_collision_hot_phase"
    assert state["a"] == 1e-13


def test_inflationary_bubble_not_implemented():
    """Inflationary bubble without potential returns no transition."""
    origin = InflationaryBubbleOrigin()
    assert origin.theory_classification == "research"
    assert origin.check_transition(None, 0.0) is None
    state = origin.create_initial_state()
    assert "error" in state


def test_origin_event_record():
    event = OriginEvent(
        event_id="evt-1",
        parent_structure="universe-alpha",
        model_id="ekpyrotic_brane_collision",
        parameters={"lambda": 10.0},
        initial_state={"a": 1e-13, "T_K": 1e15},
        energy_field_info={"rho": 1e10, "w": 1/3},
        creation_time=1.0,
        validity_metadata={"valid": True},
        theory_classification="speculative",
    )
    assert event.event_id == "evt-1"
    assert event.theory_classification == "speculative"
    d = event.__dict__
    assert d["model_id"] == "ekpyrotic_brane_collision"


def test_multiverse_origin_event_creation():
    """Test F creates a new persistent universe from a model transition."""
    root = tempfile.mkdtemp(prefix="origin-test-")
    try:
        m = MultiverseManager(root)
        # Create a parent universe with ekpyrotic origin model enabled
        parent = m.create_universe("Parent", seed=42)
        inst = m.get_instance(parent.uuid)
        # Manually set up an ekpyrotic transition at t=1.0
        # Advance to collision time
        ekpyrotic = EkpyroticOrigin(transition_valid=True, collision_time=1.0)
        m.origin_manager.register_model(ekpyrotic)
        
        # Set parent state to just before collision
        inst._state = inst.flrw.state_at(1e-6, t=0.9)
        inst.clocks.cosmic.t = 0.9
        
        # Check transition
        transitions = m.origin_manager.check_all_transitions(
            parent.uuid, inst.state, 1.0)
        assert len(transitions) == 1
        assert transitions[0]["model_id"] == "ekpyrotic_brane_collision"
        
        # Create universe from transition
        record = m.origin_manager.create_universe_from_transition(
            transitions[0], inst, m)
        assert record is not None
        assert record.new_universe_id != ""
        assert record.new_universe_id != parent.uuid
        
        # Verify the new universe exists in the multiverse
        new_inst = m.get_instance(record.new_universe_id)
        assert new_inst is not None
        assert new_inst.uuid == record.new_universe_id
        
        # List origins
        origins = m.list_origin_events()
        assert len(origins) == 1
        assert origins[0]["new_universe_id"] == record.new_universe_id
    finally:
        shutil.rmtree(root)


def test_ekpyrotic_invalid_transition_creates_no_universe():
    """Critical rule: invalid transition -> no new universe."""
    root = tempfile.mkdtemp(prefix="origin-test-")
    try:
        m = MultiverseManager(root)
        parent = m.create_universe("Parent", seed=42)
        inst = m.get_instance(parent.uuid)
        
        ekpyrotic = EkpyroticOrigin(transition_valid=False, collision_time=1.0)
        m.origin_manager.register_model(ekpyrotic)
        
        transitions = m.origin_manager.check_all_transitions(
            parent.uuid, inst.state, 1.0)
        assert len(transitions) == 1
        assert transitions[0]["transition_info"]["nonsingular"] is False
        
        record = m.origin_manager.create_universe_from_transition(
            transitions[0], inst, m)
        assert record is not None
        assert record.new_universe_id == ""  # no universe created
        
        # No new universe in multiverse
        assert len(m.list_universes()) == 1  # only parent
    finally:
        shutil.rmtree(root)
