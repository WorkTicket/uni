import numpy as np
import pytest
from universe_engine.prebiotic import polymer, autocatalysis as ac
from universe_engine.prebiotic.compartment import World, Protocell, A_SPH, DIV
from universe_engine.prebiotic.provenance import SOURCED

WET = {"T": 298.15, "pH": 7.0, "liquid_water": True, "a_w": 1.0}
DRY = {"T": 298.15, "pH": 7.0, "liquid_water": True, "a_w": 0.05}


def test_bond_free_energy_is_the_sourced_value():
    assert abs(SOURCED["peptide_bond_dG"]["value_J_mol"] - 14644.0) < 1.0
    net = polymer.build(6)
    assert abs(net.reactions[0].dH - 14644.0) < 1.0 and net.reactions[0].dS == 0.0


def test_chain_distribution_is_geometric_at_equilibrium_and_units_conserved():
    net = polymer.build(10, boost=1e7)
    t, y = net.integrate(np.r_[1.0, np.zeros(10)], WET, 1e3, n_out=3, rtol=1e-11, atol=1e-18)
    c = y[-1]; K = net.reactions[0].k_equilibrium(298.15, 1.0)
    for n in range(1, 9):
        assert abs(c[n] / (c[n - 1] * c[0]) - K) / K < 1e-6                       # detailed balance: c_{n+1}/(c_n c_1) = K
    assert abs(polymer.monomer_units(y[-1]) - 1.0) < 1e-9 and abs(polymer.monomer_units(y[0]) - 1.0) < 1e-12


def test_water_activity_shifts_equilibrium_toward_chains():
    net = polymer.build(10, boost=1e7)
    Kw, Kd = net.reactions[0].k_equilibrium(298.15, 1.0), net.reactions[0].k_equilibrium(298.15, 0.05)
    assert abs(Kd / Kw - 20.0) < 1e-9


def test_dilute_aqueous_condensation_makes_almost_no_oligomers():
    net = polymer.build(10, boost=1e7)
    _, y = net.integrate(np.r_[0.1, np.zeros(10)], WET, 1e3, n_out=2)
    assert y[-1][0] > 0.099                                              # >99% monomer: peptides do not form spontaneously in water


def test_wet_dry_cycling_beats_constant_wet_under_same_assumed_kinetics():
    """Cold wet phase (slow hydrolysis) alternating with a hot, concentrated, low-water-activity dry phase. Chain retention after rewetting
    emerges from thermodynamics (K_eff = K/a_w, concentration) plus Arrhenius kinetics; the schedule and rate prefactors are assumptions."""
    net = polymer.build(12, kf_298=5e-6, Ea=100e3)
    wet = {"T": 278.15, "pH": 7.0, "liquid_water": True, "a_w": 1.0}
    dry = {"T": 363.15, "pH": 7.0, "liquid_water": True, "a_w": 0.05}
    const = polymer.run_cycles(net, 0.1, wet, wet, 1.0, 1e3, 1e3, 3)
    cyc = polymer.run_cycles(net, 0.1, wet, dry, 100.0, 1e3, 1e3, 3)
    assert polymer.mean_chain_length(const) < 1.0005
    assert polymer.mean_chain_length(cyc) > polymer.mean_chain_length(const) + 0.05
    assert abs(polymer.monomer_units(cyc) - 0.1) < 1e-8                  # monomer units conserved through concentrate/dilute


def test_uncatalysed_condensation_is_negligible_over_a_year_with_placeholder_kinetics():
    net = polymer.build(8, boost=1.0, kf_298=1e-12)                      # slow kinetics: thermodynamics alone does not make chains quickly
    _, y = net.integrate(np.r_[0.1, np.zeros(8)], WET, 3.15e7, n_out=2)
    assert y[-1][0] > 0.0999


# ---- RAF ----
def test_raf_toy_examples():
    food = {"a", "b"}
    R = [(("a", "b"), ("c",)), (("a", "c"), ("d",)), (("c", "d"), ("e",))]
    assert ac.max_raf(food, R, [{"d"}, {"c"}, {"zzz"}]) == [0, 1]       # r3's catalyst is not producible -> removed
    assert ac.max_raf(food, R, [set(), set(), set()]) == []             # no catalysis -> no RAF
    assert ac.max_raf(food, R, [{"e"}, {"c"}, {"c"}]) == [0, 1, 2]


def test_raf_requires_food_generation():
    R = [(("x", "y"), ("z",))]
    assert ac.max_raf({"a"}, R, [{"z"}]) == []


def test_random_polymer_raf_probability_increases_with_catalysis():
    frac = {}
    for p in (0.0, 0.002, 0.02):
        hits = 0
        for seed in range(6):
            mols, food, R, cat = ac.binary_polymer_model(5, 2, p, seed)
            hits += len(ac.max_raf(food, R, cat)) > 0
        frac[p] = hits / 6
    assert frac[0.0] == 0.0 and frac[0.02] == 1.0 and frac[0.002] <= frac[0.02]


def test_raf_kinetics_grows_nonfood_mass_only_with_catalysis():
    mols, food, R, cat = ac.binary_polymer_model(4, 2, 0.05, 1)
    raf = ac.max_raf(food, R, cat)
    assert raf
    t, with_cat = ac.simulate(food, R, cat, raf, mols, 50.0, k0=1e-6, kc=1.0)
    t, no_cat = ac.simulate(food, R, cat, raf, mols, 50.0, k0=1e-6, kc=0.0)
    assert with_cat[-1] > 5 * no_cat[-1]


# ---- protocells ----
def test_geometry_constants():
    V = 1.0; assert abs(2 * A_SPH(V / 2) / A_SPH(V) - DIV) < 1e-12


def test_protocell_with_solute_grows_swells_and_divides_conserving_amphiphile():
    cell = Protocell(A=A_SPH(0.5), V=0.5, N=5e6)
    w = World([cell], pool_molecules=4e9, seed=1, solute_rate=0.0, c_out_M=2e-3)
    for _ in range(60000):
        w.step(0.05)
        assert abs(w.amphiphile_total() - w.total0) / w.total0 < 1e-12
        if w.events["divisions"] >= 1: break
    assert w.events["divisions"] >= 1 and all(c.gen >= 1 for c in w.cells)
    assert sum(c.N for c in w.cells) == 5e6                              # solute count conserved through division


def test_osmotic_swelling_without_membrane_growth_causes_lysis():
    cell = Protocell(A=A_SPH(0.5), V=0.5, N=5e7)
    w = World([cell], pool_molecules=1.0, k_uptake=0.0, seed=0, c_out_M=0.0)    # no amphiphile available
    for _ in range(40000):
        w.step(0.05)
        if w.events["lyses"]: break
    assert w.events["lyses"] == 1 and not w.cells


def test_division_is_stochastic_in_contents_but_reproducible_by_seed():
    def run(seed):
        w = World([Protocell(A_SPH(0.5), 0.5, 5e6)], 4e9, seed=seed, c_out_M=2e-3)
        for _ in range(60000):
            w.step(0.05)
            if w.events["divisions"]: break
        return sorted(c.N for c in w.cells)
    assert run(1) == run(1) and run(1) != run(2)
