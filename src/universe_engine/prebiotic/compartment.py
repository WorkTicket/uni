"""Vesicle protocell: membrane area A, enclosed volume V, impermeant solute count N.
- water crosses by osmosis (flux ~ Pf * A * Vw * (c_in - c_out)); solutes and amphiphile pool are conserved separately;
- r = A / A_sphere(V): r < 1 means the membrane is stretched (tense), r > 1 means excess area (flaccid / deformable);
- lysis if stretched beyond the strain limit; division when r >= 2^(1/3) (two equal spheres of the same total area and volume: geometric);
- membrane tension raises amphiphile uptake (qualitative mechanism, strength ASSUMED).
Parameters are placeholders (provenance.ASSUMED). Units: area um^2, volume um^3, solute molecules, concentrations mol/L."""
from dataclasses import dataclass
import numpy as np

N_A = 6.02214076e23
A_SPH = lambda V: (36 * np.pi) ** (1 / 3) * V ** (2 / 3)
DIV = 2 ** (1 / 3)
V_W = 1.8e-5           # molar volume of water, m3/mol


@dataclass
class Protocell:
    A: float; V: float; N: float; lineage: int = 0; gen: int = 0; alive: bool = True; fate: str = ""
    @property
    def ratio(self): return self.A / A_SPH(self.V)


class World:
    """Closed amphiphile accounting: pool + sigma * sum(A over live cells) is conserved (lysis returns membrane to the pool)."""
    def __init__(self, cells, pool_molecules, sigma=2.0e6, k_uptake=1e-3, tension_gain=300.0, Pf=1e-5, c_out_M=0.0, lyse_strain=0.04,
                 solute_rate=0.0, seed=0, max_cells=4000, cvc_fraction=0.02):
        self.cells = list(cells); self.pool = float(pool_molecules); self.sigma = sigma; self.k_uptake = k_uptake
        self.gain = tension_gain; self.Pf = Pf; self.c_out = c_out_M; self.lyse = lyse_strain; self.solute_rate = solute_rate
        self.rng = np.random.default_rng(seed); self.t = 0.0; self.max_cells = max_cells
        self.pool0 = float(pool_molecules); self.cvc = cvc_fraction * float(pool_molecules)   # uptake stops when pool falls to this (placeholder CVC)
        self.events = {"divisions": 0, "lyses": 0}
        self.total0 = self.amphiphile_total()

    def amphiphile_total(self):
        return self.pool + self.sigma * sum(c.A for c in self.cells if c.alive)

    def step(self, dt):
        born = []
        for c in self.cells:
            if not c.alive: continue
            c.N += self.solute_rate * dt                                      # internal production of impermeant solute (INPUT rate)
            c_in = c.N / N_A / (c.V * 1e-18) if c.V > 0 else 0.0                # mol/m3
            dV = self.Pf * (c.A * 1e-12) * V_W * (c_in - self.c_out * 1e3) / 1e-18 * dt
            c.V = max(c.V + dV, 1e-6)
            strain = max(A_SPH(c.V) / c.A - 1.0, 0.0)
            avail = max(self.pool - self.cvc, 0.0)
            dA = min(self.k_uptake * c.A * (1 + self.gain * strain) * (avail / self.pool0) * dt, avail / self.sigma)
            c.A += dA; self.pool -= dA * self.sigma
            if A_SPH(c.V) / c.A - 1.0 > self.lyse:
                c.alive = False; c.fate = "lysed"; self.events["lyses"] += 1; self.pool += self.sigma * c.A
            elif c.ratio >= DIV and len(self.cells) + len(born) < self.max_cells:
                n1 = int(self.rng.binomial(int(c.N), 0.5))
                born += [Protocell(c.A / 2, c.V / 2, n1, c.lineage, c.gen + 1), Protocell(c.A / 2, c.V / 2, int(c.N) - n1, c.lineage, c.gen + 1)]
                c.alive = False; c.fate = "divided"; self.events["divisions"] += 1
        self.cells = [c for c in self.cells if c.alive] + born
        self.t += dt
