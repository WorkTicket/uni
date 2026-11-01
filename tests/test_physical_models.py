import numpy as np
import pytest
from universe_engine.physical_models import gas_evolve,gas_step,GasState,stellar_track,disk_accretion,prebiotic_network

def test_gas_conserves_total_mass():
    s,d=gas_evolve(n=32,steps=50)
    assert d['mass_error']<1e-10
    assert np.isfinite(s.energy).all()
def test_cfl_rejects():
    n=16;s=GasState(np.ones(n),np.zeros(n),np.ones(n),np.zeros(n),np.zeros(n))
    with pytest.raises(ValueError):gas_step(s,1/n,1)
def test_stellar_solar():
    t=stellar_track(1)
    assert t['luminosity_solar_approx']==1
    assert abs(t['temperature_K_approx']-5772)<1e-8
def test_stellar_mass_invalid():
    with pytest.raises(ValueError): stellar_track(0)
def test_disk_mass():
    d=disk_accretion()
    assert abs(d['total_solids_earth_masses']-100)<1e-9
    assert (d['planetesimals_earth_masses']>=0).all()
def test_chemistry_nonnegative():
    c=prebiotic_network(steps=200)
    assert all(x>=0 for x in c['last_concentrations'].values())
def test_chemistry_evolves():
    c=prebiotic_network()
    assert c['last_concentrations']['P']>0.001
