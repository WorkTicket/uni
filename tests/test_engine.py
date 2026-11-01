import numpy as np
import pytest
from universe_engine.physics import hubble, solve, validate_cosmology
from universe_engine.field import gaussian_field
C={'H0_km_s_Mpc':67.4,'Omega_m':0.315,'Omega_r':0.000092,'Omega_Lambda':0.684908,'a_start':0.02,'a_end':1.0,'steps':500}

def test_H_today(): assert hubble(1.0,C)==pytest.approx(67.4)
def test_positive_hubble(): assert all(hubble(a,C)>0 for a in (0.02,0.1,0.5,1))
def test_expansion_trend(): assert hubble(0.1,C)>hubble(0.5,C)>hubble(1,C)
def test_growth():
    s=solve(C)
    assert s['D'][-1]==pytest.approx(1)
    assert np.all(np.diff(s['D'])>0)
    assert np.all(np.diff(s['elapsed_Gyr'])>0)
def test_reproducibility(): assert np.array_equal(gaussian_field(16,42,0.001,0.965),gaussian_field(16,42,0.001,0.965))
def test_field_mean(): assert abs(gaussian_field(16,42,0.001,0.965).mean())<1e-12
def test_field_rms(): assert gaussian_field(16,42,0.001,0.965).std()==pytest.approx(0.001)
def test_reject_curvature():
    bad={**C,'Omega_Lambda':0.6}
    with pytest.raises(ValueError): validate_cosmology(bad)
def test_reject_bad_grid():
    with pytest.raises(ValueError): gaussian_field(15,42,0.001,0.965)
def test_rk4_convergence():
    x=solve(C)['elapsed_Gyr'][-1]
    c={**C,'steps':1000}
    y=solve(c)['elapsed_Gyr'][-1]
    assert abs(x-y)<0.001

def test_future_cosmology_a_gt_1():
    c={**C,'a_end':2.0,'steps':1000}
    s=solve(c)
    assert s['a'][-1]==pytest.approx(2.0)
    assert np.all(np.diff(s['elapsed_Gyr'])>0)
    assert np.all(np.isfinite(s['H']))
    assert s['elapsed_Gyr'][-1] > solve(C)['elapsed_Gyr'][-1]

def test_future_cosmology_monotone_D():
    c={**C,'a_end':5.0,'steps':2000}
    s=solve(c)
    assert np.all(np.diff(s['D'])>0)

def test_validate_allows_future_a_end_beyond_present():
    c={**C,'a_end':100.0}
    validate_cosmology(c)  # must not raise: a_end is a model-specific validity bound

def test_validate_allows_future_a_end():
    c={**C,'a_end':3.0}
    validate_cosmology(c)  # must not raise
