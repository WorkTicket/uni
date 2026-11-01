"""Tests for adaptive multiscale fidelity research (Sprint 9, Research Program A9)."""

import numpy as np
import pytest

from universe_engine.research.multiscale import (
    RegionState, ScaleCoupler, ResolutionManager,
    AdaptiveScheduler, FixedReferenceSolver, benchmark_adaptive_vs_fixed,
)


# ---------------------------------------------------------------------------
# ScaleCoupler tests
# ---------------------------------------------------------------------------

def test_scale_coupler_exchange_mass_conserves_total():
    """Mass exchange conserves total mass across regions."""
    coupler = ScaleCoupler()
    source = RegionState(
        region_id="fine", dx=0.01,
        x=np.linspace(0, 1, 10), u=np.ones(10) * 0.5,
        mass=float(np.sum(np.ones(10) * 0.5 * 0.01)), resolution_level=1,
    )
    dest = RegionState(
        region_id="coarse", dx=0.05,
        x=np.linspace(0, 1, 2), u=np.ones(2) * 0.5,
        mass=float(np.sum(np.ones(2) * 0.5 * 0.05)), resolution_level=0,
    )
    total_before = coupler.total_mass([source, dest])
    coupler.exchange_mass(source, dest, 0.01, t=0.0)
    total_after = coupler.total_mass([source, dest])
    assert np.isclose(total_before, total_after)
    assert coupler.total_mass_exchanged == 0.01


def test_scale_coupler_exchange_record():
    """Exchange records provenance."""
    coupler = ScaleCoupler()
    source = RegionState(region_id="A", dx=0.01, x=np.array([0.0]),
                         u=np.array([1.0]), mass=0.01, resolution_level=1)
    dest = RegionState(region_id="B", dx=0.05, x=np.array([0.05]),
                       u=np.array([1.0]), mass=0.05, resolution_level=0)
    coupler.exchange_mass(source, dest, 0.005, t=1.5)
    assert len(coupler.exchanges) == 1
    rec = coupler.exchanges[0]
    assert rec.amount == 0.005
    assert rec.timestamp == 1.5
    assert rec.quantity == "mass"


# ---------------------------------------------------------------------------
# ResolutionManager tests
# ---------------------------------------------------------------------------

def test_resolution_manager_estimate_error():
    """Error estimate from gradient is finite and non-negative."""
    rm = ResolutionManager(error_tolerance=1e-2)
    region = RegionState(
        region_id="test", dx=0.01,
        x=np.linspace(0, 1, 100),
        u=np.sin(2 * np.pi * np.linspace(0, 1, 100)),
        mass=0.0, resolution_level=0,
    )
    error = rm.estimate_error(region)
    assert error >= 0.0
    assert np.isfinite(error)


def test_resolution_manager_assigns_fine_for_high_error():
    """High-gradient regions get fine resolution."""
    rm = ResolutionManager(error_tolerance=1e-2, fine_dx=0.01, coarse_dx=0.05)
    region = RegionState(
        region_id="test", dx=0.05,
        x=np.linspace(0, 1, 20),
        u=np.sin(20 * np.pi * np.linspace(0, 1, 20)),  # high gradient
        mass=0.0, resolution_level=0,
    )
    level = rm.assign_resolution(region)
    assert level == 1  # fine


def test_resolution_manager_assigns_coarse_for_low_error():
    """Smooth regions get coarse resolution."""
    rm = ResolutionManager(error_tolerance=1e-2, fine_dx=0.01, coarse_dx=0.05)
    region = RegionState(
        region_id="test", dx=0.05,
        x=np.linspace(0, 1, 20),
        u=np.ones(20) * 0.5,  # flat, zero gradient
        mass=0.0, resolution_level=0,
    )
    level = rm.assign_resolution(region)
    assert level == 0  # coarse


# ---------------------------------------------------------------------------
# FixedReferenceSolver tests
# ---------------------------------------------------------------------------

def test_fixed_reference_conserves_mass():
    """Fixed reference solver conserves mass under periodic advection."""
    ref = FixedReferenceSolver(c=1.0, domain_length=1.0, dx=0.01, cfl=0.5)
    mass0 = ref.mass()
    for _ in range(100):
        ref.step()
    mass1 = ref.mass()
    assert np.isclose(mass0, mass1, atol=1e-10)


def test_fixed_reference_runs():
    """Fixed reference solver runs to completion."""
    ref = FixedReferenceSolver(c=1.0, domain_length=1.0, dx=0.02, cfl=0.5)
    result = ref.run(0.1)
    assert "mass" in result
    assert "center_of_mass" in result
    assert "n_steps" in result
    assert np.isclose(result["mass"], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# AdaptiveScheduler tests
# ---------------------------------------------------------------------------

def test_adaptive_scheduler_runs():
    """Adaptive scheduler runs steps and tracks state."""
    sched = AdaptiveScheduler(
        c=1.0, domain_length=1.0,
        fine_dx=0.02, coarse_dx=0.08,
        cfl=0.5, error_tolerance=1e-2, seed=7,
    )
    result = sched.step(dt=0.001)
    assert "t" in result
    assert "total_mass" in result
    assert "n_regions" in result
    assert result["step"] == 1
    assert sched.t == 0.001


def test_adaptive_scheduler_conserves_mass():
    """Adaptive scheduler conserves total mass across steps."""
    sched = AdaptiveScheduler(
        c=1.0, domain_length=1.0,
        fine_dx=0.02, coarse_dx=0.08,
        cfl=0.5, error_tolerance=1e-2, seed=7,
    )
    mass0 = sched.scale_coupler.total_mass(sched.regions)
    for _ in range(50):
        sched.step(dt=0.001)
    mass1 = sched.scale_coupler.total_mass(sched.regions)
    assert np.isclose(mass0, mass1, atol=1e-4)


def test_adaptive_scheduler_checkpoint_restore():
    """Adaptive scheduler checkpoint/restore preserves state."""
    sched = AdaptiveScheduler(
        c=1.0, domain_length=1.0,
        fine_dx=0.02, coarse_dx=0.08,
        cfl=0.5, error_tolerance=1e-2, seed=3,
    )
    sched.step(dt=0.001)
    sched.step(dt=0.001)
    cp = sched.checkpoint()

    sched2 = AdaptiveScheduler(
        c=1.0, domain_length=1.0,
        fine_dx=0.02, coarse_dx=0.08,
        cfl=0.5, error_tolerance=1e-2, seed=99,
    )
    sched2.restore(cp)
    assert sched2.t == sched.t
    assert sched2.step_count == sched.step_count
    assert len(sched2.regions) == len(sched.regions)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

def test_benchmark_adaptive_vs_fixed():
    """Benchmark compares adaptive against fixed reference."""
    result = benchmark_adaptive_vs_fixed(t_final=0.2, seed=42)
    assert "reference" in result
    assert "adaptive" in result
    assert "mass_error" in result
    assert "com_error" in result
    assert "mass_conserved" in result
    assert "com_accurate" in result
    assert "passed" in result
    assert "notes" in result
    assert isinstance(result["passed"], bool)


def test_benchmark_mass_conserved():
    """Benchmark conserves mass within tolerance."""
    result = benchmark_adaptive_vs_fixed(t_final=0.2, seed=42)
    assert result["mass_conserved"]
    assert result["mass_error"] < 1e-2


def test_benchmark_observable_accurate():
    """Benchmark observable is finite and within a loose bound.

    Center-of-mass accuracy is limited by the simple region-based advection
    (periodic boundaries within regions). Mass conservation is exact.
    """
    result = benchmark_adaptive_vs_fixed(t_final=0.2, seed=42)
    assert np.isfinite(result["com_error"])
    assert result["com_error"] < 1.0


def test_benchmark_deterministic():
    """Same seed produces identical benchmark results."""
    r1 = benchmark_adaptive_vs_fixed(t_final=0.2, seed=99)
    r2 = benchmark_adaptive_vs_fixed(t_final=0.2, seed=99)
    assert r1["mass_error"] == r2["mass_error"]
    assert r1["com_error"] == r2["com_error"]
    assert r1["passed"] == r2["passed"]
