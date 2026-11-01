import numpy as np
import pytest
from universe_engine.planet import Planet, PlanetSpec, StarSpec
from universe_engine.planet import constants as K
from universe_engine.planet.model import annual_mean_insolation_shape, water_boiling_k, carbonate_ph


def earth(**kw):
    return Planet(PlanetSpec(**kw), t_gyr=4.57)


def test_earth_like_calibration_and_energy_closure():
    p = earth(); e = p.environment()
    assert 284 < e["T_global_K"] < 291 and p.converged
    assert abs(e["energy_imbalance_wm2"]) < 1e-3          # global energy closure at equilibrium
    assert e["liquid_water_fraction"] > 0.7 and e["flags"] == []
    assert p.T[len(p.T)//2] > p.T[0]                      # equator warmer than pole


@pytest.mark.parametrize("a", [0.9, 1.0, 1.3, 1.6, 2.0])
def test_energy_closure_across_orbits(a):
    p = earth(a_au=a)
    assert abs(p.energy_imbalance_wm2()) < 1e-2


def test_solar_luminosity_history():
    s = StarSpec(1.0)
    assert abs(s.luminosity(4.57) / K.L_SUN - 1) < 1e-12
    assert 0.69 < s.luminosity(0.0) / K.L_SUN < 0.72      # faint young Sun ~70%
    assert s.luminosity(9.0) > s.luminosity(4.57)


def test_insolation_shape_properties():
    x = -1 + (np.arange(36) + 0.5) / 18
    s = annual_mean_insolation_shape(x, 23.44)
    assert abs(s.mean() - 1) < 1e-12 and s[18] > s[0]
    s90 = annual_mean_insolation_shape(x, 90.0)
    assert s90[0] > s90[18]                               # at 90 deg obliquity the poles get more annual sunlight than the equator


def test_boiling_point_anchor():
    assert abs(water_boiling_k(101325.0) - 373.15) < 3.0
    assert np.isnan(water_boiling_k(100.0))               # below the triple-point pressure: no liquid


def test_ice_albedo_hysteresis():
    p = earth(); down = []
    for a in np.linspace(1.0, 1.25, 26):
        p.spec.a_au = a; p._equilibrate(); down.append(p.global_mean_T())
    up = []
    for a in np.linspace(1.25, 1.0, 26):
        p.spec.a_au = a; p._equilibrate(); up.append(p.global_mean_T())
    # at some intermediate distance both a warm and a snowball state exist
    gaps = [abs(d - u) for d, u in zip(down, up[::-1])]
    assert max(gaps) > 20.0


def test_sterile_cold_world_has_no_liquid_water():
    p = earth(a_au=2.5, pco2_bar=1e-4)
    e = p.environment()
    assert e["liquid_water_fraction"] == 0.0 and np.isnan(e["pH_ocean"])


def test_dry_world_has_no_ocean_chemistry():
    p = earth(has_surface_water=False)
    assert p.liquid_water_fraction() == 0.0


def test_runaway_regime_flagged_not_hidden():
    p = earth(a_au=0.7)
    assert any("runaway" in f for f in p.environment()["flags"])


def test_ph_decreases_with_pco2():
    assert carbonate_ph(1e-3, 2.3e-3) < carbonate_ph(1e-4, 2.3e-3) < 14


def test_carbon_silicate_thermostat_vs_stagnant_lid():
    a = earth(); a.evolve(5.57, dt_gyr=0.05)
    b = earth(tectonics="stagnant"); b.evolve(5.57, dt_gyr=0.05)
    T0 = earth().global_mean_T()
    assert a.pco2 < 2.8e-4 * 0.5                          # weathering draws CO2 down as the star brightens
    assert b.pco2 == 2.8e-4
    assert abs(a.global_mean_T() - T0) < 6.0
    assert b.global_mean_T() - T0 > 6.0                   # without the feedback the planet warms


def test_snowball_escape_by_co2_accumulation():
    p = Planet(PlanetSpec(pco2_bar=0.1), t_gyr=0.5)
    h = p.evolve(4.57, dt_gyr=0.05)
    assert min(e["liquid_water_fraction"] for e in h) == 0.0 and h[-1]["liquid_water_fraction"] > 0.5


def test_deterministic():
    r = [earth().evolve(5.0, dt_gyr=0.05)[-1]["T_global_K"] for _ in range(2)]
    assert r[0] == r[1]
