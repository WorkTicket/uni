"""RAF (reflexively autocatalytic and F-generated) sets, plus the random binary-polymer catalytic model.
Which molecule catalyses which reaction is ASSUMED (random, probability p); RAF emergence is a property of that assumption."""
import itertools
import numpy as np


def closure(food, reactions):
    """Closure of the food set under reactions (each reaction: (reactants, products)); catalysis ignored."""
    have = set(food); changed = True
    while changed:
        changed = False
        for re_, pr in reactions:
            if set(re_) <= have and not set(pr) <= have:
                have |= set(pr); changed = True
    return have


def max_raf(food, reactions, catalysis):
    """reactions: list of (reactants, products); catalysis: list of sets of catalyst molecules per reaction (same index).
    Returns indices of the maximal RAF (empty if none): iteratively keep reactions that are F-generated and catalysed by a
    molecule in the closure of the food set under the *current* reaction subset."""
    R = list(range(len(reactions)))
    while True:
        sub = [reactions[i] for i in R]
        cl = closure(food, sub)
        keep = [i for i in R if set(reactions[i][0]) <= cl and (catalysis[i] & (cl | set(food)))]
        if keep == R: return R
        R = keep


def binary_polymer_model(max_len, food_len, p, seed):
    rng = np.random.default_rng(seed)
    mols = ["".join(b) for L in range(1, max_len + 1) for b in itertools.product("01", repeat=L)]
    food = [m for m in mols if len(m) <= food_len]
    reactions = []
    for m in mols:
        for k in range(1, len(m)):
            reactions.append(((m[:k], m[k:]), (m,)))       # ligation a + b -> m
            reactions.append(((m,), (m[:k], m[k:])))       # cleavage m -> a + b
    cat = [set() for _ in reactions]
    # each molecule catalyses each reaction with probability p
    hits = rng.random((len(mols), len(reactions))) < p
    for mi, ri in zip(*np.nonzero(hits)): cat[ri].add(mols[mi])
    return mols, food, reactions, cat


def simulate(food, reactions, catalysis, raf, mols, t_end, k0=1e-6, kc=1.0, food_clamp=1.0, n_out=50):
    """Mass-action kinetics on the given reactions: rate = (k0 + kc * sum catalyst conc) * prod(reactants). Food held at fixed
    concentration (open system). Conserved quantity for the closed non-food part is not claimed; returns total non-food mass."""
    from scipy.integrate import solve_ivp
    idx = {m: i for i, m in enumerate(mols)}; foodset = set(food)
    sel = list(raf)
    def f(t, c):
        d = np.zeros_like(c)
        for i in sel:
            re_, pr = reactions[i]
            r = np.prod([c[idx[m]] for m in re_]) * (k0 + kc * sum(c[idx[m]] for m in catalysis[i]))
            for m in re_: d[idx[m]] -= r
            for m in pr: d[idx[m]] += r
        for m in foodset: d[idx[m]] = 0.0
        return d
    c0 = np.full(len(mols), 1e-9)
    for m in food: c0[idx[m]] = food_clamp
    sol = solve_ivp(f, (0, t_end), c0, method="LSODA", rtol=1e-8, atol=1e-14, t_eval=np.linspace(0, t_end, n_out))
    nonfood = [i for m, i in idx.items() if m not in foodset]
    return sol.t, sol.y.T[:, nonfood].sum(1)
