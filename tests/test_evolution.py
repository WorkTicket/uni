import numpy as np
import pytest
from universe_engine.evolution import cic_density,poisson_gradient,pm_evolve,halo_candidates,galaxy_population,stellar_and_planetary_catalog,evolutionary_ecology

def test_cic_mass():
    rng=np.random.default_rng(4);d=cic_density(rng.random((87,3)),8)
    assert d.mean()==pytest.approx(1,abs=1e-12)
    assert d.min()>=0

def test_poisson_zero(): assert np.allclose(poisson_gradient(np.zeros((8,8,8))),0)
def test_non_linear_run():
    p,v,d,h=pm_evolve(n=8,steps=40,amplitude=.1)
    assert p.shape==(512,3) and d.shape==(8,8,8)
    assert np.all((p>=0)&(p<1)) and np.isfinite(v).all()
    assert d.mean()==pytest.approx(1)
    assert len(h)>1 and all(q['rms']>=0 for q in h)
def test_repeatable():
    _,_,d,_=pm_evolve(n=8,steps=12,seed=33)
    _,_,other,_=pm_evolve(n=8,steps=12,seed=33)
    assert np.array_equal(d,other)
def test_invalid():
    with pytest.raises(ValueError):pm_evolve(n=5)
def test_halos_and_catalogs():
    d=np.ones((8,8,8));d[1,2,3]=7
    h=halo_candidates(d)
    assert len(h)==1
    g=galaxy_population(h)
    assert len(g)==1 and g[0]['stellar_mass_proxy_solar']>0
    s=stellar_and_planetary_catalog(g)
    assert all(x['star_mass_solar']>0 for x in s)
def test_evolutionary_replay_and_bounds():
    a=evolutionary_ecology(seed=22,generations=35)
    assert a==evolutionary_ecology(seed=22,generations=35)
    assert all(0<=r['mean_trait']<=1 and r['population']>=2 for r in a)
