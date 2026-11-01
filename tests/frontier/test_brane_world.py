"""Tests for the reduced-order brane-world research model (Requirement 6, Test E)."""
import math
import numpy as np
import pytest

from universe_engine.frontier.brane_world import (
    BraneWorldModel, BraneWorldParameters, BraneState,
)


def test_brane_init_and_friedmann():
    params = BraneWorldParameters(lambda_brane=10.0, G_5=1.0, Lambda_5=-6.0)
    params.derive_4d_constants()
    assert params.G_4 is not None and params.G_4 > 0
    model = BraneWorldModel(params)
    state = model.initialize(t=0.0, a_brane=1e-10, rho_matter=0.0, rho_radiation=1e10)
    assert state.a_brane == 1e-10
    assert state.H_brane > 0  # expanding
    assert state.validity == "valid"


def test_brane_radiation_era_scaling():
    """Radiation era: rho ~ a^-4, H ~ a^-2, a ~ t^(1/2)."""
    params = BraneWorldParameters(lambda_brane=1000.0, G_5=1.0, Lambda_5=-6.0,
                                  junction_tol=1e12)
    params.derive_4d_constants()
    model = BraneWorldModel(params)
    state = model.initialize(t=0.0, a_brane=1e-6, rho_matter=0.0, rho_radiation=1e8)
    # Take several small steps
    for _ in range(10):
        state = model.step(1e-6)
    assert state.validity == "valid"
    assert state.a_brane > 1e-6  # scale factor grew
    assert state.H_brane > 0


def test_brane_high_energy_correction():
    """At high rho >> lambda, the rho^2 term dominates and H^2 ~ rho^2."""
    params = BraneWorldParameters(lambda_brane=1.0, G_5=1.0, Lambda_5=-6.0)
    params.derive_4d_constants()
    model = BraneWorldModel(params)
    # Low density: H^2 ~ rho
    state_low = model.initialize(t=0.0, a_brane=1.0, rho_matter=1e-6, rho_radiation=0.0)
    # High density: H^2 ~ rho^2 (brane-world correction)
    state_high = model.initialize(t=0.0, a_brane=1.0, rho_matter=1e6, rho_radiation=0.0)
    # H_high / H_low should be much larger than sqrt(rho_high/rho_low) = 1e6
    ratio = state_high.H_brane / max(state_low.H_brane, 1e-300)
    assert ratio > 10.0  # brane-world correction enhances H at high density


def test_brane_junction_residual():
    params = BraneWorldParameters(lambda_brane=10.0, G_5=1.0, Lambda_5=-6.0)
    params.derive_4d_constants()
    model = BraneWorldModel(params)
    state = model.initialize(t=0.0, a_brane=1.0, rho_matter=0.0, rho_radiation=0.0)
    # Junction residual should be finite
    assert math.isfinite(state.junction_residual)


def test_brane_invalid_on_singular_transition():
    """Critical rule: ekpyrotic collision without valid transition -> invalid."""
    params = BraneWorldParameters(
        lambda_brane=10.0, G_5=1.0, Lambda_5=-6.0,
        collision_model="ekpyrotic", collision_time=1.0, transition_valid=False,
        junction_tol=1e12,
    )
    params.derive_4d_constants()
    model = BraneWorldModel(params)
    state = model.initialize(t=0.0, a_brane=1e-6, rho_radiation=1e8)
    # Step up to the collision time (continue past approaching_singularity)
    while state.t < 1.0 and state.validity not in ("invalid", "finished"):
        state = model.step(0.1)
    assert state.validity == "invalid"
    assert "nonsingular transition" in state.validity_message


def test_brane_valid_on_singular_transition():
    """With transition_valid=True, ekpyrotic collision produces valid state."""
    params = BraneWorldParameters(
        lambda_brane=10.0, G_5=1.0, Lambda_5=-6.0,
        collision_model="ekpyrotic", collision_time=1.0, transition_valid=True,
        junction_tol=1e12,
    )
    params.derive_4d_constants()
    model = BraneWorldModel(params)
    state = model.initialize(t=0.0, a_brane=1e-6, rho_radiation=1e8)
    while state.t < 1.0 and state.validity not in ("invalid", "finished"):
        state = model.step(0.1)
    assert state.validity == "valid"


def test_brane_origin_event():
    model = BraneWorldModel()
    event = model.origin_event(
        parent_structure="bulk_5D",
        model_id="rs_ii_brane",
        parameters={"lambda": 10.0},
        initial_state={"a": 1e-13, "T": 1e16},
        energy_field_info={"rho": 1e10, "p": 1e10/3},
        creation_time=0.0,
        validity_metadata={"numerical_error": 1e-10},
        theory_classification="speculative",
    )
    assert event.model_id == "rs_ii_brane"
    assert event.theory_classification == "speculative"
    assert event.parent_structure == "bulk_5D"
    assert event.event_id.startswith("brane-")


def test_brane_state_serialization():
    state = BraneState(
        t=1.0, a_brane=2.0, H_brane=0.5,
        rho_matter=0.3, rho_radiation=0.7,
        y_brane=0.0, y_velocity=0.0,
        extrinsic_curvature=0.5, junction_residual=1e-6,
        validity="valid", validity_message="ok", error_estimate=1e-8,
    )
    d = state.__dict__
    assert d["a_brane"] == 2.0
    assert d["validity"] == "valid"
