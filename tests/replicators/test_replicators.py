import numpy as np
import pytest
from universe_engine.replicators import Population, duplex_mechanism, mutate, ALPHABET


def hamming(a, b): return sum(x != y for x, y in zip(a, b))


def fixed(k, d=0.0):
    return lambda g, n, M: (k, d)


def test_copy_operator_error_statistics():
    rng = np.random.default_rng(0); g = "ACGU" * 5; mu = 0.02; L = len(g)
    muts = [mutate(g, mu, rng)[1] for _ in range(20000)]
    assert abs(np.mean(muts) - L * mu) < 0.02                                     # mean = L mu
    assert abs(np.mean(np.array(muts) == 0) - (1 - mu) ** L) < 0.01              # P(exact copy) = (1-mu)^L
    c, m = mutate(g, 0.3, rng); assert hamming(g, c) == m                          # reported count equals actual substitutions
    assert mutate(g, 0.0, rng) == (g, 0)


def test_heredity_without_mutation_and_exact_unit_conservation():
    p = Population({"ACGUACGU": 5}, pool=10_000, rate_fn=fixed(0.1, 0.01), mu=0.0, seed=1)
    p.gillespie(30.0)
    assert set(p.counts) == {"ACGUACGU"} and p.events["replications"] > 5
    assert p.ledger_error() == 0 and p.units() == p.initial_units                 # closed system: integer-exact


def test_resource_limit_stops_replication():
    p = Population({"ACGU": 3}, pool=40, rate_fn=fixed(1.0, 0.0), mu=0.0, seed=2)
    p.gillespie(200.0)
    assert p.pool < 4 and p.total() * 4 + p.pool == p.initial_units
    assert p.total() == 3 + (40 // 4)                                             # every monomer ends up in a copy


def test_lineage_tracks_ancestry_and_mutation_distance():
    p = Population({"A" * 12: 20}, pool=100_000, rate_fn=fixed(0.5, 0.0), mu=0.05, seed=3)
    p.gillespie(6.0)
    assert len(p.counts) > 5
    for g, (parent, t0, m) in p.lineage.items():
        if parent is None: assert g == "A" * 12; continue
        assert hamming(g, parent) == m and m >= 1 and t0 > 0
    leaf = max(p.counts, key=lambda g: hamming(g, "A" * 12)); path = p.path_to_root(leaf)
    assert path[-1] == "A" * 12 and path[0] == leaf


def test_selection_coefficient_matches_theory_without_mutation():
    # two genotypes, same decay, rates 0.6 and 0.5; ln(n1/n2) should grow at ~0.1 per unit time in a non-limiting pool
    p = Population({"AAAA": 300, "CCCC": 300}, pool=10**9, rate_fn=lambda g, n, M: ((0.6 if g == "AAAA" else 0.5), 0.2), mu=0.0, seed=4)
    p.tau_leap(10.0, 0.01)
    slope = np.log(p.counts["AAAA"] / p.counts["CCCC"]) / 10.0
    assert abs(slope - 0.1) < 0.03


def test_neutral_drift_fixes_either_type_at_roughly_equal_rate():
    wins = 0; runs = 60
    for s in range(runs):
        p = Population({"AAAA": 6, "CCCC": 6}, pool=10**9, rate_fn=lambda g, n, M: (0.5, 0.5), mu=0.0, seed=100 + s)
        for _ in range(4000):
            p.gillespie(p.t + 1.0)
            if len(p.counts) < 2 or p.total() == 0: break
            if p.total() > 60: p.transfer(0.5, 10**9)                                  # keep it small so drift is visible
        if len(p.counts) == 1: wins += list(p.counts)[0] == "AAAA"
    assert 0.3 < wins / runs < 0.7


def test_error_threshold_matches_quasispecies_theory():
    L, sigma, master = 10, 10.0, "A" * 10
    def quasi(mu, seed):
        rate = lambda g, n, M: (sigma if g == master else 1.0, 0.0)
        p = Population({master: 3000}, pool=10**12, rate_fn=rate, mu=mu, seed=seed)
        fr = []
        for step in range(45):
            p.tau_leap(p.t + 0.1, 0.02)
            if p.total() > 6000: p.transfer(0.5, 10**12)
            if step >= 25: fr.append(p.counts.get(master, 0) / max(p.total(), 1))
        return float(np.mean(fr)), p
    x_low, p = quasi(0.02, 5)
    Q = (1 - 0.02) ** L; theory = (Q * sigma - 1) / (sigma - 1)
    assert abs(x_low - theory) < 0.08                                                  # single-peak quasispecies equilibrium
    x_high, _ = quasi(0.35, 6)
    assert x_high < 0.02                                                              # beyond the error threshold the master sequence is lost
    assert p.ledger_error() == 0


def _growth(g, T, c=1e-7):
    kr, kd = duplex_mechanism(T, 1.0)(g, c * 6.02214076e23, 10**9)
    return kr - kd


def _t_growth(g):
    lo, hi = 250.0, 420.0
    for _ in range(60):
        m = (lo + hi) / 2
        if _growth(g, m) > 0: hi = m
        else: lo = m
    return hi


def test_minimum_growth_temperature_rises_with_gc_content():
    """Consequence of the supplied duplex mechanism (strand-separation problem): GC-rich strands need a hotter environment to replicate."""
    tg = [_t_growth(g) for g in ("A" * 10, "G" * 5 + "A" * 5, "G" * 10)]
    assert tg[0] < tg[1] < tg[2] and tg[2] - tg[0] > 20.0


def test_mechanism_selection_in_simulation_depends_on_temperature():
    at, mixed = "A" * 10, "G" * 5 + "A" * 5
    def run(T, seed):
        p = Population({at: 200, mixed: 200}, pool=10**9, rate_fn=duplex_mechanism(T, 1e-13), mu=0.0, seed=seed)
        for _ in range(30):
            p.tau_leap(p.t + 20.0, 2.0)
            if p.total() > 2000: p.transfer(0.5, 10**9)
        return p.counts.get(at, 0), p.counts.get(mixed, 0), p.total()
    a, m, tot = run(298.0, 1)                       # between the two growth thresholds (288 K and 307 K)
    assert a > 4 * max(m, 1)                      # started 1:1; the AT-rich strand is strongly enriched
    a2, m2, tot2 = run(280.0, 1)                    # colder than both thresholds: everything declines
    assert tot2 < 400
    a3, m3, tot3 = run(330.0, 1)                    # hotter than both: both grow
    assert a3 > 100 and m3 > 100
