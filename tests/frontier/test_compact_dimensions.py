"""Tests for compact extra-dimension research models (Requirement 8 / Test H)."""
import math
import pytest

from universe_engine.frontier.compact_dimensions import (
    CompactDimension, CompactificationSpec,
    CircleCompactification, ToroidalCompactification, SphereCompactification,
    CompactDimensionExplorer,
    verify_kk_circle_spectrum, verify_kk_torus_spectrum, verify_sphere_spectrum,
)


# ---------------------------------------------------------------- circle
def test_circle_kk_mass_analytic():
    """KK mass for circle: m_n = n/R (exact analytic result)."""
    R = 2.5
    circle = CircleCompactification(R)
    for n in range(0, 11):
        expected = abs(n) / R
        assert circle.kk_mass(n) == pytest.approx(expected, rel=1e-12)


def test_circle_verify_spectrum():
    result = verify_kk_circle_spectrum(3.0, max_mode=20)
    assert result["all_exact"]
    assert result["max_error"] < 1e-12


def test_circle_radius_positive():
    with pytest.raises(ValueError):
        CircleCompactification(0.0)
    with pytest.raises(ValueError):
        CircleCompactification(-1.0)


# ---------------------------------------------------------------- torus
def test_torus_kk_mass_analytic():
    """KK mass for torus: m^2 = sum_i (n_i/R_i)^2 (exact analytic result)."""
    radii = [1.5, 2.0, 0.7]
    torus = ToroidalCompactification(radii)
    modes = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, -1, 3]]
    for m_vec in modes:
        expected = math.sqrt(sum((n / R) ** 2 for n, R in zip(m_vec, radii)))
        assert torus.kk_mass(m_vec) == pytest.approx(expected, rel=1e-12)


def test_torus_verify_spectrum():
    radii = [1.0, 2.0]
    modes = [[1, 0], [0, 1], [1, 1], [2, 1]]
    result = verify_kk_torus_spectrum(radii, modes)
    assert result["all_exact"]
    for r in result["results"]:
        assert r["error"] < 1e-12


def test_torus_dimensions():
    radii = [1.0, 2.0, 3.0]
    torus = ToroidalCompactification(radii)
    assert torus.d == 3


def test_torus_radius_positive():
    with pytest.raises(ValueError):
        ToroidalCompactification([1.0, 0.0])


# ---------------------------------------------------------------- sphere
def test_sphere_laplacian_eigenvalue():
    """Laplacian eigenvalue on S^2: lambda_l = l(l+1)/R^2."""
    R = 2.0
    sphere = SphereCompactification(R)
    for l in range(0, 6):
        expected = l * (l + 1) / R**2
        assert sphere.laplacian_eigenvalue(l) == pytest.approx(expected, rel=1e-12)


def test_sphere_degeneracy():
    sphere = SphereCompactification(1.0)
    for l in range(0, 6):
        assert sphere.degeneracy(l) == 2 * l + 1


def test_sphere_verify_spectrum():
    result = verify_sphere_spectrum(1.5, max_l=8)
    assert result["all_exact"]
    assert result["max_error"] < 1e-12


def test_sphere_radius_positive():
    with pytest.raises(ValueError):
        SphereCompactification(0.0)


# ---------------------------------------------------------------- compactification spec
def test_compactification_spec():
    dims = [
        CompactDimension(name="theta", type="circle", radius=1.5,
                         periodic=True, range=(0.0, 2 * math.pi)),
    ]
    spec = CompactificationSpec(base_D=4, compact_dims=dims)
    assert spec.total_D == 5
    spectrum = spec.kk_spectrum(max_mode=5)
    assert "theta" in spectrum
    assert len(spectrum["theta"]) == 6


def test_compactification_total_kk_mass():
    dims = [
        CompactDimension(name="theta1", type="circle", radius=1.0,
                         periodic=True, range=(0.0, 2 * math.pi)),
        CompactDimension(name="theta2", type="circle", radius=2.0,
                         periodic=True, range=(0.0, 2 * math.pi)),
    ]
    spec = CompactificationSpec(base_D=4, compact_dims=dims)
    m = spec.total_kk_mass({"theta1": 1, "theta2": 1})
    expected = math.sqrt((1/1.0)**2 + (1/2.0)**2)
    assert m == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------- explorer
def test_explorer_accessibility():
    R = 1.0  # radius in GeV^-1 => KK mass n/R in GeV
    explorer = CompactDimensionExplorer(R)
    # At E=1 GeV, mode n=1 has mass 1.0 GeV -> accessible
    result = explorer.effective_4d_mass(1, 0.0, 1.0)
    assert result["kk_mass"] == 1.0
    assert result["accessible"] is True
    # At E=0.5 GeV, mode n=1 mass 1.0 GeV -> NOT accessible
    result2 = explorer.effective_4d_mass(1, 0.0, 0.5)
    assert result2["accessible"] is False


def test_explorer_visibility_vs_scale():
    explorer = CompactDimensionExplorer(0.5)  # R=0.5 GeV^-1 => m_1 = 2 GeV
    scales = [0.1, 1.0, 5.0]
    results = explorer.visibility_vs_scale(scales)
    assert len(results) == 3
    # At E=0.1, no KK modes accessible (m_1 = 2 > 0.1)
    assert results[0]["n_accessible"] == 1  # only n=0 (m0=0)
    # At E=5.0, modes n=0..9 accessible (m_10 = 20 > 5)
    assert results[2]["n_accessible"] >= 2


# ---------------------------------------------------------------- geometry spec from circle
def test_circle_geometry_spec():
    circle = CircleCompactification(2.0)
    spec = circle.geometry_spec()
    assert spec.chart.D == 5
    assert spec.chart.is_periodic[4] is True  # theta is periodic
    assert spec.compactification["type"] == "circle"
    assert spec.compactification["radius"] == 2.0
    # Flat: zero curvature
    curv = spec.metric.curvature()
    assert curv.ricci_scalar == 0.0
