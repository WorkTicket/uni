import numpy as np
import pytest
from universe_engine.chemistry import Species, Reaction, Network, env_from_planet
from universe_engine.chemistry.network import R_GAS, N_A
from universe_engine.planet import Planet, PlanetSpec

ENV = {"T": 300.0, "pH": 7.0, "liquid_water": True}
A_, B_ = Species("A", {"C": 1, "H": 2}), Species("B", {"C": 1, "H": 2})


def test_unbalanced_reaction_is_rejected():
    X = Species("X", {"C": 1}); Y = Species("Y", {"C": 2})
    with pytest.raises(ValueError):
        Network([X, Y], [Reaction({"X": 1}, {"Y": 1}, 1.0, 0.0)])
    P = Species("P", {"H": 1}, +1); N = Species("N", {"H": 1}, 0)
    with pytest.raises(ValueError):
        Network([P, N], [Reaction({"P": 1}, {"N": 1}, 1.0, 0.0)])          # charge not conserved


def test_first_order_decay_matches_analytic():
    net = Network([A_, B_], [Reaction({"A": 1}, {"B": 1}, 1e6, 40e3)])
    k = 1e6 * np.exp(-40e3 / (R_GAS * 300.0))
    t, y = net.integrate([1.0, 0.0], ENV, 5.0 / k)
    assert np.max(np.abs(y[:, 0] - np.exp(-k * t))) < 1e-7
    assert max(net.conservation_error(y).values()) < 1e-9


def test_arrhenius_temperature_dependence():
    r = Reaction({"A": 1}, {"B": 1}, 1e6, 50e3)
    ratio = r.k_forward(310.0) / r.k_forward(300.0)
    assert abs(ratio - np.exp(50e3 / R_GAS * (1 / 300 - 1 / 310))) < 1e-12


def test_equilibrium_equals_thermodynamic_constant_and_ignores_catalysis():
    for kH in (0.0, 1e9):
        r = Reaction({"A": 1}, {"B": 1}, 1e8, 30e3, dH=-5e3, dS=10.0, reversible=True, kH=kH)
        net = Network([A_, B_], [r])
        env = {"T": 290.0, "pH": 3.0, "liquid_water": True}
        t, y = net.integrate([1.0, 0.0], env, 1e5)
        assert abs(y[-1, 1] / y[-1, 0] - r.k_equilibrium(290.0)) / r.k_equilibrium(290.0) < 1e-6   # catalyst changes speed, not K


def test_acid_catalysis_speeds_reaction():
    r = Reaction({"A": 1}, {"B": 1}, 1e3, 30e3, kH=1e6)
    assert r.k_forward(300, 1e-3) > 100 * r.k_forward(300, 1e-7)


def test_no_liquid_water_means_zero_rate():
    net = Network([A_, B_], [Reaction({"A": 1}, {"B": 1}, 1e9, 0.0)])
    t, y = net.integrate([1.0, 0.0], {"T": 300, "pH": 7, "liquid_water": False}, 10.0)
    assert np.all(y[:, 0] == 1.0)


def test_second_order_dimerisation_analytic():
    M = Species("M", {"C": 2}); D = Species("D", {"C": 4})
    net = Network([M, D], [Reaction({"M": 2}, {"D": 1}, 1e4, 0.0)])
    k = 1e4; r = 2 * k                                  # d[M]/dt = -2 k [M]^2 for 2M->D with rate k[M]^2
    t, y = net.integrate([1.0, 0.0], ENV, 5.0)
    assert np.max(np.abs(y[:, 0] - 1.0 / (1.0 + r * t))) < 1e-7
    assert max(net.conservation_error(y).values()) < 1e-9


def test_gillespie_mean_matches_ode_and_conserves_exactly():
    r = Reaction({"A": 1}, {"B": 1}, 1.0, 0.0, reversible=True, dS=0.0)     # K = 1, k=1
    net = Network([A_, B_], [r])
    vol = 200.0 / (N_A * 1.0)                           # volume holding 200 molecules at 1 M
    fin = []
    for seed in range(60):
        t, n, ev = net.gillespie([200, 0], ENV, vol, 1.0, seed=seed)
        assert n[-1].sum() == 200                       # exact particle (element) conservation
        fin.append(n[-1, 0])
    # ODE: A(t) = (1+exp(-2t))/2 * 200
    expect = 200 * (1 + np.exp(-2.0)) / 2
    assert abs(np.mean(fin) - expect) < 4 * np.std(fin) / np.sqrt(len(fin)) + 1.0


def test_gillespie_reproducible():
    net = Network([A_, B_], [Reaction({"A": 1}, {"B": 1}, 1.0, 0.0)])
    a = net.gillespie([50, 0], ENV, 50 / N_A, 2.0, seed=3)[1]; b = net.gillespie([50, 0], ENV, 50 / N_A, 2.0, seed=3)[1]
    assert np.array_equal(a, b)


def test_planetary_conditions_change_chemical_outcome():
    """Same network, two planets: the environment alone changes the outcome (and a frozen sterile world does nothing)."""
    M = Species("M", {"C": 2, "H": 4, "O": 2}); W = Species("W", {"H": 2, "O": 1}); D = Species("D", {"C": 4, "H": 6, "O": 3})
    cond = Reaction({"M": 2}, {"D": 1, "W": 1}, 1e9, 55e3, dH=+20e3, dS=-30.0, reversible=True)
    net = Network([M, W, D], [cond])
    out = {}
    for name, a in (("temperate", 1.0), ("cold", 1.9)):
        p = Planet(PlanetSpec(a_au=a), t_gyr=4.57); env = env_from_planet(p.environment())
        t, y = net.integrate([0.5, 55.5, 0.0], env, 1e4)
        out[name] = (env, y[-1, 2], max(net.conservation_error(y).values()))
    assert out["temperate"][0]["liquid_water"] and not out["cold"][0]["liquid_water"]
    assert out["cold"][1] == 0.0 and out["temperate"][1] > 0.0
    assert out["temperate"][2] < 1e-8
