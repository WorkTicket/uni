"""Tests for cosmological numerical diagnostics (diagnostics.py).

Covers:
  - Power spectrum of an analytic Fourier mode (peak at the input k).
  - Dimensionless power spectrum conversion.
  - Power spectrum from a PM3D snapshot.
  - Linear growth comparison (bounded error).
  - Cross-seed statistics (identical seeds -> identical results).
  - Resolution convergence.
  - Numerical error summary (mass conservation, finite values).
"""

import math
import numpy as np
import pytest

from universe_engine.cosmos.diagnostics import (
    measure_power_spectrum,
    dimensionless_power_spectrum,
    power_spectrum_from_pm3d,
    compare_linear_growth,
    cross_seed_statistics,
    resolution_convergence,
    numerical_error_summary,
)
from universe_engine.solvers.pm3d import Cosmology, ParticleMesh3D
from universe_engine.cosmos.initial_conditions import generate_initial_conditions


H0 = 67.4
OM = 0.315
OR = 9.2e-5
OL = 0.684908


def _cosmology():
    return Cosmology(H0_km_s_Mpc=H0, Omega_m=OM, Omega_r=OR, Omega_Lambda=OL)


def _make_cosine_field(n, k_mode, box_size_Mpc, amplitude=1.0):
    """delta(x) = amplitude * cos(2 pi k_mode · x / L) in box fraction."""
    idx = np.arange(n)
    IX, IY, IZ = np.meshgrid(idx, idx, idx, indexing="ij")
    phase = 2.0 * np.pi * (k_mode[0] * IX + k_mode[1] * IY + k_mode[2] * IZ) / n
    return amplitude * np.cos(phase)


# ---------------------------------------------------------------------------
# Power spectrum of an analytic Fourier mode
# ---------------------------------------------------------------------------

def test_power_spectrum_single_mode_peak():
    """A pure cosine mode produces a power-spectrum peak at its k."""
    n = 16
    box_size_Mpc = 100.0
    k_mode = (1, 0, 0)
    delta = _make_cosine_field(n, k_mode, box_size_Mpc, amplitude=0.5)
    assert abs(delta.mean()) < 1e-12

    result = measure_power_spectrum(delta, box_size_Mpc, n_bins=10)

    # The fundamental mode k = 2 pi / L is the first bin edge.
    k_fund = 2.0 * math.pi / box_size_Mpc
    assert result.k_edges[0] == pytest.approx(k_fund, rel=1e-9)
    assert k_fund <= result.k_edges[1]
    # Power at the fundamental bin should dominate.
    assert result.P_k[0] == pytest.approx(result.P_k.max(), rel=1e-6)


def test_dimensionless_power_spectrum():
    """Delta^2(k) = k^3 P(k) / (2 pi^2)."""
    k = np.array([0.1, 0.5, 1.0])
    P_k = np.array([100.0, 8.0, 1.0])
    d2 = dimensionless_power_spectrum(k, P_k)
    expected = k ** 3 * P_k / (2.0 * math.pi ** 2)
    assert np.allclose(d2, expected)
    assert np.all(np.isfinite(d2))


def test_measure_power_spectrum_zero_mean():
    """k=0 mode excluded; mean density is zero."""
    n = 16
    box_size_Mpc = 100.0
    delta = _make_cosine_field(n, (1, 0, 0), box_size_Mpc)
    result = measure_power_spectrum(delta, box_size_Mpc, n_bins=5)
    assert result.k[0] > 0  # no k=0 bin
    assert np.all(np.isfinite(result.P_k))
    assert np.all(result.counts > 0)


# ---------------------------------------------------------------------------
# Power spectrum from PM3D snapshot
# ---------------------------------------------------------------------------

def test_power_spectrum_from_pm3d():
    """Measure power spectrum from a PM3D snapshot."""
    cosmo = _cosmology()
    ic = generate_initial_conditions(
        seed=7, box_size_Mpc=100.0, n_particles_per_axis=6,
        n_mesh=16, a_start=0.5, cosmology=cosmo, amplitude=0.05,
    )
    sim = ic.to_pm3d()
    result = power_spectrum_from_pm3d(sim, n_bins=10)
    assert len(result.k) > 0
    assert np.all(np.isfinite(result.P_k))
    assert result.n_particles == sim.x.shape[0]
    assert result.box_size_Mpc == sim.box_size_Mpc
    assert result.shot_noise > 0


# ---------------------------------------------------------------------------
# Linear growth comparison
# ---------------------------------------------------------------------------

def test_compare_linear_growth_bounded_error():
    """Linear growth error should be bounded for a small-amplitude mode."""
    cosmo = _cosmology()
    ic = generate_initial_conditions(
        seed=5, box_size_Mpc=100.0, n_particles_per_axis=8,
        n_mesh=16, a_start=0.2, cosmology=cosmo, amplitude=0.01,
    )
    sim = ic.to_pm3d()
    result = compare_linear_growth(sim, 0.2, 0.6, k_mode=(1, 0, 0),
                                   max_d_ln_a=0.01)
    assert np.all(np.isfinite(result.theory_growth))
    assert np.all(np.isfinite(result.measured_growth))
    assert result.rms_error >= 0
    assert result.max_error >= 0


# ---------------------------------------------------------------------------
# Cross-seed statistics
# ---------------------------------------------------------------------------

def test_cross_seed_statistics_identical_seeds():
    """The same seed twice gives identical power spectra."""
    cosmo = _cosmology()
    res1 = cross_seed_statistics(
        seeds=[42, 42], box_size_Mpc=100.0,
        n_particles_per_axis=6, n_mesh=16, a_start=0.5,
        cosmology=cosmo, amplitude=0.02, n_bins=8,
    )
    # Two runs with the same seed produce identical P(k), so scatter = 0.
    assert np.allclose(res1.std_P_k, 0.0, atol=1e-10)
    assert np.allclose(res1.fractional_scatter, 0.0, atol=1e-10)
    assert len(res1.seeds) == 2


def test_cross_seed_statistics_different_seeds():
    """Different seeds produce scatter (cosmic variance + shot noise)."""
    cosmo = _cosmology()
    res = cross_seed_statistics(
        seeds=[1, 2, 3], box_size_Mpc=100.0,
        n_particles_per_axis=6, n_mesh=16, a_start=0.5,
        cosmology=cosmo, amplitude=0.02, n_bins=8,
    )
    assert len(res.seeds) == 3
    assert np.all(np.isfinite(res.mean_P_k))
    assert np.all(res.mean_P_k >= 0)
    assert np.all(res.fractional_scatter >= 0)


# ---------------------------------------------------------------------------
# Resolution convergence
# ---------------------------------------------------------------------------

def test_resolution_convergence_runs():
    """Resolution convergence produces finite results at multiple resolutions."""
    cosmo = _cosmology()
    res = resolution_convergence(
        box_size_Mpc=100.0,
        n_particles_per_axis_list=[6, 8],
        n_mesh_list=[16, 16],
        a_start=0.5, cosmology=cosmo, seed=42, amplitude=0.02, n_bins=8,
    )
    assert len(res.resolutions) == 2
    assert len(res.mean_delta2) == 2
    assert len(res.rms_delta2_scatter) == 2
    assert all(np.all(np.isfinite(d2)) for d2 in res.mean_delta2)


# ---------------------------------------------------------------------------
# Numerical error summary
# ---------------------------------------------------------------------------

def test_numerical_error_summary():
    """Error summary reports mass conservation and finite values."""
    cosmo = _cosmology()
    ic = generate_initial_conditions(
        seed=3, box_size_Mpc=100.0, n_particles_per_axis=6,
        n_mesh=16, a_start=0.2, cosmology=cosmo, amplitude=0.02,
    )
    sim = ic.to_pm3d()
    summary = numerical_error_summary(sim, 0.2, 0.5)
    assert summary.mass_conservation_error < 1e-12
    assert summary.finite_check is True
    assert summary.nonfinite_count == 0
    assert abs(summary.density_mean) < 1e-12
    assert summary.force_symmetry_error >= 0
    assert np.all(np.isfinite(summary.mean_momentum))
