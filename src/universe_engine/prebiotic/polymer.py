"""Chain growth by monomer addition vs hydrolysis, on top of the conservation-checked reaction engine.
P_n + M <-> P_{n+1} + W (water at fixed activity). Thermodynamics from a sourced bond free energy; kinetics are inputs."""
import numpy as np
from ..chemistry import Species, Reaction, Network
from .provenance import SOURCED

MONO = {"C": 2, "H": 5, "N": 1, "O": 2}             # glycine-like composition (bookkeeping only)


def chain_species(n):
    return Species(f"P{n}", {e: n * c - (n - 1) * (2 if e == "H" else 1 if e == "O" else 0) for e, c in MONO.items()})


def build(n_max=12, dG=SOURCED["peptide_bond_dG"]["value_J_mol"], kf_298=1e-4, Ea=60e3, boost=1.0):
    """dS=0 assumption: dH=dG. kf_298 (M^-1 s^-1) and boost are ASSUMED inputs; K is thermodynamic."""
    sp = [chain_species(n) for n in range(1, n_max + 1)] + [Species("W", {"H": 2, "O": 1})]
    A = kf_298 * boost * np.exp(Ea / (8.314462618 * 298.15))
    rx = [Reaction({f"P{n}": 1, "P1": 1} if n > 1 else {"P1": 2}, {f"P{n+1}": 1, "W": 1}, A, Ea, dH=dG, dS=0.0, reversible=True,
                   solvent=("W",), provenance="dG sourced (secondary); kinetics assumed", label=f"add to P{n}") for n in range(1, n_max)]
    return Network(sp, rx)


def monomer_units(c):
    n = np.arange(1, c.shape[-1])                      # last column is W
    return c[..., :-1] @ n


def run_cycles(net, c0_total_monomer, wet, dry, conc_factor, t_wet, t_dry, n_cycles):
    """Wet phase -> evaporate (concentrate by conc_factor, dry env) -> rewet (dilute back). Returns monomer-unit totals & chain distribution."""
    n = len(net.species); c = np.zeros(n); c[0] = c0_total_monomer
    for _ in range(n_cycles):
        _, y = net.integrate(c, wet, t_wet, n_out=2); c = y[-1].copy()
        c[:-1] *= conc_factor
        _, y = net.integrate(c, dry, t_dry, n_out=2); c = y[-1].copy()
        c[:-1] /= conc_factor
    _, y = net.integrate(c, wet, t_wet, n_out=2)
    return y[-1]


def mean_chain_length(c):
    n = np.arange(1, len(c)); x = c[:-1]
    return float((x @ n) / x.sum())
