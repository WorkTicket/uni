"""Replicating-sequence populations.

SUPPLIED (assumed) mechanisms, not emergent:
- template-directed copying exists, consumes one monomer unit per site, and makes substitution errors with probability mu per site
  (uniform over the other 3 letters; no indels, no site-dependent error);
- a molecule is copy-competent only while single-stranded; a template can sit in a duplex with a complementary-like partner with
  equilibrium constant K = exp(-dG_duplex / RT), dG_duplex = sum of per-base contributions (GC stronger than AT; PLACEHOLDER values);
- duplex form is protected from decay (d_ds << d_ss; PLACEHOLDER values).
Replication and survival rates of a sequence follow from these, and so depend on temperature and strand concentration. No arbitrary fitness number is assigned
in the default mechanism; test landscapes may inject rate functions explicitly.
Units: counts are molecules; monomer pool in monomer units; concentrations mol/L via volume_l."""
from dataclasses import dataclass, field
import numpy as np

ALPHABET = "ACGU"
N_A = 6.02214076e23
R_GAS = 8.314462618


def mutate(g, mu, rng):
    """Copy g with independent per-site substitution probability mu. Returns (offspring, number_of_mutations)."""
    L = len(g); m = int(rng.binomial(L, mu))
    if m == 0: return g, 0
    pos = rng.choice(L, size=m, replace=False); s = list(g)
    for p in pos:
        s[p] = rng.choice([a for a in ALPHABET if a != s[p]])
    return "".join(s), m


def duplex_mechanism(T, volume_l, k_copy=1e-2, K_M=1e3, d_ss=1e-3, d_ds=1e-5,
                     dH_AT=-30e3, dS_AT=-85.0, dH_GC=-37e3, dS_GC=-95.0):
    """Returns rate_fn(g, n_total, M) -> (replication rate per molecule, decay rate per molecule).
    Duplex free energy per base dG = dH - T dS with PLACEHOLDER nearest-neighbour-order values (not fitted to any dataset)."""
    def rate(g, n_total, M):
        L = len(g); gc = sum(1 for a in g if a in "GC"); at = L - gc
        dG = gc * (dH_GC - T * dS_GC) + at * (dH_AT - T * dS_AT)
        K = np.exp(-dG / (R_GAS * T))                          # M^-1
        c = n_total / N_A / volume_l                           # strand concentration, M
        s = 1.0 / (1.0 + K * c)                                # single-stranded (copy-competent) fraction
        return k_copy * s * M / (K_M + M), s * d_ss + (1 - s) * d_ds
    return rate


@dataclass
class Population:
    counts: dict                       # genotype -> molecules
    pool: int                          # free monomer units
    rate_fn: callable                  # (g, n_total, pool) -> (k_rep, k_decay) per molecule
    mu: float = 0.0
    seed: int = 0
    t: float = 0.0
    supplied: int = 0                  # monomer units added from outside (refills)
    removed: int = 0                   # monomer units removed (thinning / transfer)
    lineage: dict = field(default_factory=dict)   # genotype -> (parent genotype or None, birth time, n_mut)
    events: dict = field(default_factory=lambda: {"replications": 0, "decays": 0, "mutants": 0})

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.initial_units = self.units()
        for g in self.counts: self.lineage.setdefault(g, (None, 0.0, 0))

    # ---- accounting ---------------------------------------------------------------------------------
    def total(self): return sum(self.counts.values())
    def units(self): return int(self.pool + sum(len(g) * n for g, n in self.counts.items()))
    def ledger_error(self):
        """Exact integer conservation: units now = initial + supplied - removed (must be 0)."""
        return self.units() - (self.initial_units + self.supplied - self.removed)

    def _born(self, g_parent, g_child, nmut):
        self.counts[g_child] = self.counts.get(g_child, 0) + 1
        if g_child not in self.lineage: self.lineage[g_child] = (g_parent, self.t, nmut)
        if nmut: self.events["mutants"] += 1

    # ---- exact stochastic simulation (Gillespie direct method) ----------------------------------------------
    def gillespie(self, t_end, max_events=10**7):
        ev = 0
        while self.t < t_end and ev < max_events:
            if not self.counts: self.t = t_end; break
            nt = self.total(); gs = list(self.counts); n = np.array([self.counts[g] for g in gs], float)
            rr = np.array([self.rate_fn(g, nt, self.pool) for g in gs])
            can = np.array([self.pool >= len(g) for g in gs])
            a_rep = n * rr[:, 0] * can; a_dec = n * rr[:, 1]
            a0 = a_rep.sum() + a_dec.sum()
            if a0 <= 0: self.t = t_end; break
            dt = self.rng.exponential(1.0 / a0)
            if self.t + dt > t_end: self.t = t_end; break
            self.t += dt
            i = self.rng.choice(2 * len(gs), p=np.r_[a_rep, a_dec] / a0)
            if i < len(gs):
                g = gs[i]; child, m = mutate(g, self.mu, self.rng)
                self.pool -= len(g); self._born(g, child, m); self.events["replications"] += 1
            else:
                g = gs[i - len(gs)]; self.counts[g] -= 1; self.pool += len(g); self.events["decays"] += 1
                if self.counts[g] == 0: del self.counts[g]
            ev += 1
        return ev

    # ---- tau-leaping for large populations (approximate; same rates) ----------------------------------------------
    def tau_leap(self, t_end, dt):
        while self.t < t_end - 1e-12:
            h = min(dt, t_end - self.t); nt = self.total(); new = {}
            reps = {}; decs = {}
            for g, n in list(self.counts.items()):
                kr, kd = self.rate_fn(g, nt, self.pool)
                reps[g] = int(self.rng.poisson(n * kr * h)); decs[g] = min(int(self.rng.poisson(n * kd * h)), n)
            demand = sum(len(g) * r for g, r in reps.items())
            if demand > self.pool:                                      # resource-limited: scale replications down
                f = self.pool / demand if demand else 0.0
                reps = {g: int(self.rng.binomial(r, f)) for g, r in reps.items()}
            for g, r in reps.items():
                if r == 0: continue
                self.pool -= len(g) * r; self.events["replications"] += r
                if self.mu <= 0:
                    new[g] = new.get(g, 0) + r; continue
                ms = self.rng.binomial(len(g), self.mu, size=r)
                new[g] = new.get(g, 0) + int((ms == 0).sum())
                for m in ms[ms > 0]:
                    pos = self.rng.choice(len(g), size=int(m), replace=False); s = list(g)
                    for p in pos: s[p] = self.rng.choice([a for a in ALPHABET if a != s[p]])
                    child = "".join(s); new[child] = new.get(child, 0) + 1; self.events["mutants"] += 1
                    self.lineage.setdefault(child, (g, self.t, int(m)))
            for g, d in decs.items():
                if d: self.counts[g] -= d; self.pool += len(g) * d; self.events["decays"] += d
            for g, c in new.items(): self.counts[g] = self.counts.get(g, 0) + c
            self.counts = {g: n for g, n in self.counts.items() if n > 0}
            self.t += h

    # ---- bounded population: serial transfer (random thinning + refill to a target pool) ------------------------------
    def transfer(self, keep_fraction, refill_pool):
        for g in list(self.counts):
            k = int(self.rng.binomial(self.counts[g], keep_fraction))
            self.removed += len(g) * (self.counts[g] - k)
            if k: self.counts[g] = k
            else: del self.counts[g]
        out = int(self.pool * (1 - keep_fraction)); self.pool -= out; self.removed += out
        add = max(refill_pool - self.pool, 0); self.pool += add; self.supplied += add

    # ---- diagnostics ----------------------------------------------------------------------------------------------------
    def path_to_root(self, g):
        out = [g]
        while self.lineage[out[-1]][0] is not None: out.append(self.lineage[out[-1]][0])
        return out

    def fractions(self):
        nt = self.total() or 1
        return {g: n / nt for g, n in self.counts.items()}

    def mean_gc(self):
        nt = self.total() or 1
        return sum(sum(a in "GC" for a in g) / len(g) * n for g, n in self.counts.items()) / nt
