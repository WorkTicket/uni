"""Tests for scientific discovery laboratory (Sprint 7, Research Program F)."""

import numpy as np
import math
import pytest

from universe_engine.research.discovery import (
    BellApparatus, CorrelationModel, ClassicalLocalModel, QuantumModel,
    ConstantModel, DiscoveryResult, run_discovery_experiment,
)
from universe_engine.research.quantum.chsh import optimal_chsh_settings


# ---------------------------------------------------------------------------
# Bell apparatus tests
# ---------------------------------------------------------------------------

def test_bell_apparatus_measure_returns_counts():
    """Bell apparatus measurement returns outcome counts and correlation."""
    app = BellApparatus(seed=1)
    r = app.measure(0.0, math.pi / 4.0, n_shots=1000)
    assert "counts" in r
    assert "correlation" in r
    assert r["n_shots"] == 1000
    total = sum(r["counts"].values())
    assert total == 1000
    assert -1.0 <= r["correlation"] <= 1.0


def test_bell_apparatus_history():
    """Apparatus records measurement history."""
    app = BellApparatus(seed=1)
    app.measure(0.0, 0.0, n_shots=100)
    app.measure(0.1, 0.2, n_shots=100)
    assert len(app.history()) == 2


def test_bell_apparatus_reset():
    """Apparatus reset clears history."""
    app = BellApparatus(seed=1)
    app.measure(0.0, 0.0, n_shots=100)
    app.reset()
    assert len(app.history()) == 0


def test_bell_apparatus_chsh():
    """CHSH computation from four settings."""
    app = BellApparatus(seed=7)
    angles = optimal_chsh_settings()
    s = app.chsh(angles, n_shots=5000)
    assert abs(s - 2.828) < 0.1  # close to quantum max


# ---------------------------------------------------------------------------
# Correlation model tests
# ---------------------------------------------------------------------------

def test_classical_local_bound():
    """Classical local model's CHSH prediction satisfies |S| <= 2."""
    model = ClassicalLocalModel()
    # Fit on some data
    data = [{"angle_a": 0.0, "angle_b": math.pi / 4, "correlation": 0.5},
            {"angle_a": 0.0, "angle_b": 3 * math.pi / 4, "correlation": -0.3}]
    model.fit(data)
    a, ap, b, bp = optimal_chsh_settings()
    s = (model.predict(a, b) - model.predict(a, bp)
         + model.predict(ap, b) + model.predict(ap, bp))
    assert abs(s) <= 2.0 + 1e-6


def test_quantum_chsh_violation():
    """Quantum model's CHSH prediction violates the classical bound."""
    model = QuantumModel()
    data = [{"angle_a": 0.0, "angle_b": math.pi / 4, "correlation": 0.7},
            {"angle_a": 0.0, "angle_b": 3 * math.pi / 4, "correlation": -0.7},
            {"angle_a": math.pi / 2, "angle_b": math.pi / 4, "correlation": 0.7},
            {"angle_a": math.pi / 2, "angle_b": 3 * math.pi / 4, "correlation": 0.7}]
    model.fit(data)
    a, ap, b, bp = optimal_chsh_settings()
    s = (model.predict(a, b) - model.predict(a, bp)
         + model.predict(ap, b) + model.predict(ap, bp))
    assert abs(s) > 2.0
    assert abs(s - 2.828) < 0.2


def test_constant_model():
    """Constant model predicts the mean correlation."""
    model = ConstantModel()
    data = [{"angle_a": 0.0, "angle_b": 0.1, "correlation": 0.5},
            {"angle_a": 0.2, "angle_b": 0.3, "correlation": 0.9}]
    model.fit(data)
    assert model.c == pytest.approx(0.7)
    assert model.predict(1.0, 2.0) == 0.7


# ---------------------------------------------------------------------------
# Discovery experiment tests
# ---------------------------------------------------------------------------

def test_discovery_experiment_runs():
    """Discovery experiment completes and returns a result."""
    r = run_discovery_experiment(seed=42, n_training=24, n_test=12, n_shots=2000)
    assert isinstance(r, DiscoveryResult)
    assert r.n_training_settings > 0
    assert r.n_test_settings > 0
    assert "quantum" in r.test_error
    assert "classical_local" in r.test_error
    assert "constant" in r.test_error


def test_discovery_selects_quantum():
    """Agent selects quantum model on held-out data for quantum apparatus."""
    r = run_discovery_experiment(seed=42, n_training=30, n_test=15, n_shots=5000)
    assert r.selected_model == "quantum"
    assert r.discovery_successful


def test_discovery_chsh_near_quantum_max():
    """Measured CHSH S is near the quantum maximum."""
    r = run_discovery_experiment(seed=7, n_training=24, n_test=12, n_shots=10000)
    assert abs(r.chsh_quantum - 2.828) < 0.15


def test_discovery_not_supplied_chsh():
    """Notes confirm the agent was not supplied the CHSH formula."""
    r = run_discovery_experiment(seed=42, n_training=24, n_test=12, n_shots=5000)
    assert any("not supplied" in n.lower() for n in r.notes)


def test_discovery_deterministic_with_seed():
    """Same seed produces identical results."""
    r1 = run_discovery_experiment(seed=99, n_training=20, n_test=10, n_shots=3000)
    r2 = run_discovery_experiment(seed=99, n_training=20, n_test=10, n_shots=3000)
    assert r1.selected_model == r2.selected_model
    assert r1.chsh_quantum == r2.chsh_quantum
    assert r1.discovery_successful == r2.discovery_successful


def test_discovery_classical_vs_quantum_prediction():
    """Classical model predicts CHSH<=2, quantum predicts ~2*sqrt(2)."""
    r = run_discovery_experiment(seed=42, n_training=30, n_test=15, n_shots=5000)
    assert r.chsh_classical <= 2.0 + 1e-6
    assert abs(r.chsh_quantum - 2.828) < 0.2
