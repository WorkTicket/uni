"""Tests for the higher-dimensional geometry core (Requirement 5 / Test D)."""
import math
import numpy as np
import pytest

from universe_engine.geometry.geometry_core import (
    CoordinateChart, MetricSignature, MetricTensor, ChristoffelSymbols,
    Curvature, GeometrySpec, Field, BoundaryCondition,
    compute_christoffel, ConstantMetric, FLRWMetric, Bulk5DMetric,
    flat_minkowski_4d, flat_minkowski_5d, circle_compactification,
)


# ---------------------------------------------------------------- charts
def test_coordinate_chart():
    chart = CoordinateChart(
        name="test", coords=["t", "x", "y", "z"],
        ranges=[(None, None)] * 4, is_periodic=[False] * 4,
    )
    assert chart.D == 4
    assert chart.coords == ["t", "x", "y", "z"]


# ---------------------------------------------------------------- signatures
def test_metric_signature():
    sig = MetricSignature.lorentzian_4d()
    assert sig.D == 4
    assert sig.n_negative == 1 and sig.n_positive == 3 and sig.n_zero == 0
    assert sig.signature_string == "(1,0,3)"

    sig5 = MetricSignature.lorentzian_5d()
    assert sig5.D == 5 and sig5.n_positive == 4


# ---------------------------------------------------------------- constant metric
def test_constant_metric_is_flat():
    chart = CoordinateChart(name="m4", coords=["t", "x", "y", "z"])
    sig = MetricSignature.lorentzian_4d()
    metric = ConstantMetric(sig, chart)
    assert metric.is_diagonal
    assert not metric.depends_on_coords
    curv = metric.curvature()
    assert np.allclose(curv.ricci, 0)
    assert curv.ricci_scalar == 0.0
    assert np.allclose(curv.riemann, 0)


def test_constant_metric_5d_flat():
    chart = CoordinateChart(name="m5", coords=["t", "x", "y", "z", "w"])
    sig = MetricSignature.lorentzian_5d()
    metric = ConstantMetric(sig, chart)
    assert metric.D == 5
    curv = metric.curvature()
    assert np.allclose(curv.ricci, 0)
    assert curv.ricci_scalar == 0.0


# ---------------------------------------------------------------- flat geometry specs
def test_flat_minkowski_4d():
    spec = flat_minkowski_4d()
    assert spec.chart.D == 4
    assert spec.metric.D == 4
    assert spec.metric.is_lorentzian()
    curv = spec.metric.curvature()
    assert np.allclose(curv.ricci, 0)
    assert curv.ricci_scalar == 0.0


def test_flat_minkowski_5d():
    spec = flat_minkowski_5d()
    assert spec.chart.D == 5
    assert spec.metric.D == 5
    assert spec.metric.is_lorentzian()
    curv = spec.metric.curvature()
    assert np.allclose(curv.ricci, 0)


# ---------------------------------------------------------------- circle compactification
def test_circle_compactification():
    R = 2.5
    spec = circle_compactification(R)
    assert spec.chart.D == 5
    assert spec.chart.is_periodic[4] is True  # theta is periodic
    assert spec.compactification["type"] == "circle"
    assert spec.compactification["radius"] == R
    # Metric: g_theta_theta = R^2
    assert spec.metric.components[4, 4] == pytest.approx(R**2)
    # Flat: zero curvature
    curv = spec.metric.curvature()
    assert np.allclose(curv.ricci, 0)
    assert curv.ricci_scalar == 0.0


# ---------------------------------------------------------------- FLRW metric
def test_flrw_metric_components():
    chart = CoordinateChart(name="flrw", coords=["t", "x", "y", "z"])
    a_func = lambda t: 1.0 + 0.1 * t
    adot_func = lambda t: 0.1
    metric = FLRWMetric(chart, a_func, adot_func)
    comps = metric.components_at(0.0)
    assert comps[0, 0] == -1.0
    assert comps[1, 1] == pytest.approx(1.0)
    assert comps[2, 2] == pytest.approx(1.0)
    assert comps[3, 3] == pytest.approx(1.0)

    # At t=10: a = 2.0, so g_ii = 4.0
    comps10 = metric.components_at(10.0)
    assert comps10[1, 1] == pytest.approx(4.0)


def test_flrw_christoffel():
    chart = CoordinateChart(name="flrw", coords=["t", "x", "y", "z"])
    a_func = lambda t: 1.0 + 0.1 * t
    adot_func = lambda t: 0.1
    metric = FLRWMetric(chart, a_func, adot_func)
    Gamma = metric.christoffel_at(0.0)
    # Gamma^0_ii = a*adot = 1*0.1 = 0.1
    assert Gamma.components[0, 1, 1] == pytest.approx(0.1)
    assert Gamma.components[0, 2, 2] == pytest.approx(0.1)
    assert Gamma.components[0, 3, 3] == pytest.approx(0.1)
    # Gamma^i_0i = adot/a = 0.1/1 = 0.1
    assert Gamma.components[1, 0, 1] == pytest.approx(0.1)
    assert Gamma.components[1, 1, 0] == pytest.approx(0.1)


def test_flrw_ricci_scalar():
    chart = CoordinateChart(name="flrw", coords=["t", "x", "y", "z"])
    # a(t) = t (Milne-like, but flat slicing); a'=1, a''=0
    # R_ij = 2 adot^2 delta_ij = 2 delta_ij, R_00 = 0
    # R = (1/a^2) * 3 * 2 = 6 / t^2
    a_func = lambda t: max(t, 1e-6)
    adot_func = lambda t: 1.0
    metric = FLRWMetric(chart, a_func, adot_func)
    R_scalar = metric.ricci_scalar_at(1.0)
    assert R_scalar == pytest.approx(6.0, abs=1e-6)


# ---------------------------------------------------------------- bulk 5D metric
def test_bulk5d_metric():
    chart = CoordinateChart(name="bulk5d", coords=["t", "x", "y", "z", "w"])
    n_func = lambda t, w: 1.0
    a_func = lambda t, w: 1.0 + 0.05 * w
    b_func = lambda t, w: 1.0
    metric = Bulk5DMetric(chart, n_func, a_func, b_func)
    assert metric.D == 5
    comps = metric.components_at(0.0, 0.0)
    assert comps[0, 0] == -1.0
    assert comps[4, 4] == 1.0
    comps_w = metric.components_at(0.0, 10.0)
    assert comps_w[1, 1] == pytest.approx(1.5**2)


# ---------------------------------------------------------------- fields
def test_scalar_field():
    chart = CoordinateChart(name="test", coords=["t", "x"])
    field = Field(name="phi", value=np.array([1.0, 2.0, 3.0]), chart=chart)
    assert field.name == "phi"
    assert field.value.shape == (3,)


# ---------------------------------------------------------------- boundary conditions
def test_boundary_condition():
    bc = BoundaryCondition(
        name="wall", coordinate="w", value=0.0,
        type="dirichlet", applies_to=["g_tt", "g_xx"],
    )
    assert bc.type == "dirichlet"
    assert bc.coordinate == "w"


# ---------------------------------------------------------------- geometry spec serialization
def test_geometry_spec_to_dict():
    spec = flat_minkowski_4d()
    d = spec.to_dict()
    assert d["chart"]["name"] == "minkowski_4d"
    assert d["metric"]["signature"]["n_negative"] == 1
    assert d["symmetries"] == ["homogeneous", "isotropic", "static"]


# ---------------------------------------------------------------- compute_christoffel
def test_compute_christoffel_flat():
    chart = CoordinateChart(name="flat", coords=["t", "x", "y", "z"])
    sig = MetricSignature.lorentzian_4d()
    metric = ConstantMetric(sig, chart)
    D = 4
    dg = np.zeros((D, D, D))  # all derivatives zero
    Gamma = compute_christoffel(metric, dg)
    assert np.allclose(Gamma.components, 0)
