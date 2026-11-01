"""Reaction networks. Concentrations are mol/L (M); standard state 1 M for equilibrium constants.

Rule (from the project brief): rate parameters are NEVER presented as measurements unless they carry a citation.
Every Reaction has `provenance`; the default 'illustrative' means 'chosen to exercise the model, not measured'."""
from dataclasses import dataclass, field
from math import comb, factorial
import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314462618
N_A = 6.02214076e23


@dataclass(frozen=True)
class Species:
    name: str
    composition: dict            # element -> atoms, e.g. {"C": 2, "H": 6, "O": 1}
    charge: int = 0


@dataclass
class Reaction:
    reactants: dict              # species name -> stoichiometric coefficient
    products: dict
    A: float                     # pre-exponential factor of the forward rate constant, in M^(1-order) s^-1
    Ea: float                    # J/mol
    dH: float = 0.0              # reaction enthalpy J/mol (forward), used only if reversible
    dS: float = 0.0              # reaction entropy J/mol/K
    reversible: bool = False
    kH: float = 0.0              # extra specific acid-catalysed term: k = k0 + kH*[H+] + kOH*[OH-] (same Arrhenius scaling)
    kOH: float = 0.0
    requires_liquid_water: bool = True
    solvent: tuple = ()          # species at fixed unit activity (e.g. water): kept in the balance, excluded from mass action
    provenance: str = "illustrative"
    label: str = ""

    def k_forward(self, T, h_plus=1e-7):
        arr = self.A * np.exp(-self.Ea / (R_GAS * T))
        base = 1.0 + self.kH * h_plus + self.kOH * (1e-14 / h_plus)
        return arr * base

    def k_equilibrium(self, T, a_w=1.0):
        """K for the written reaction (1 M standard state, solvent at unit activity). If water is a product (solvent),
        a lower water activity a_w raises the effective K by a_w^-nu_w (drying drives condensation)."""
        K = np.exp(-(self.dH - T * self.dS) / (R_GAS * T))
        nu_w = self.products.get("W", 0) - self.reactants.get("W", 0) if "W" in self.solvent else 0
        return K * a_w ** (-nu_w)


def env_from_planet(planet_env):
    """Convert a planet.environment() dict into a chemistry environment (liquid-water bodies only)."""
    lw = planet_env["liquid_water_fraction"] > 0
    return {"T": planet_env["T_liquid_mean_K"] if lw else planet_env["T_global_K"],
            "pH": planet_env["pH_ocean"] if lw else 7.0, "liquid_water": lw}


class Network:
    def __init__(self, species, reactions):
        self.species = list(species)
        self.index = {s.name: i for i, s in enumerate(self.species)}
        self.reactions = list(reactions)
        self.elements = sorted({e for s in self.species for e in s.composition})
        self.E = np.array([[s.composition.get(e, 0) for s in self.species] for e in self.elements], float)
        self.q = np.array([s.charge for s in self.species], float)
        nS, nR = len(self.species), len(self.reactions)
        self.nu_r = np.zeros((nR, nS)); self.nu_p = np.zeros((nR, nS))
        for j, r in enumerate(self.reactions):
            for n, c in r.reactants.items(): self.nu_r[j, self.index[n]] = c
            for n, c in r.products.items(): self.nu_p[j, self.index[n]] = c
            d = self.nu_p[j] - self.nu_r[j]
            if np.any(self.E @ d != 0) or (self.q @ d) != 0:
                raise ValueError(f"reaction {j} ({r.label}) violates element or charge balance")
        self.S = (self.nu_p - self.nu_r).T                 # stoichiometric matrix (species x reactions)
        self.ma_r = self.nu_r.copy(); self.ma_p = self.nu_p.copy()          # mass-action exponents (solvent columns removed)
        for j, r in enumerate(self.reactions):
            for n in r.solvent: self.ma_r[j, self.index[n]] = 0; self.ma_p[j, self.index[n]] = 0

    # ---- rate constants under an environment ---------------------------------------------------
    def _ks(self, env):
        T = env["T"]; h = 10 ** -env.get("pH", 7.0)
        kf = np.zeros(len(self.reactions)); kr = np.zeros(len(self.reactions))
        for j, r in enumerate(self.reactions):
            if r.requires_liquid_water and not env.get("liquid_water", True):
                continue                                    # no solvent -> no aqueous reaction (rate exactly 0)
            kf[j] = r.k_forward(T, h)
            if r.reversible: kr[j] = kf[j] / r.k_equilibrium(T, env.get("a_w", 1.0))
        return kf, kr

    def rates(self, c, env, ks=None):
        kf, kr = ks or self._ks(env)
        cc = np.maximum(c, 0.0)
        fwd = kf * np.prod(np.where(self.ma_r > 0, cc[None, :] ** self.ma_r, 1.0), axis=1)
        rev = kr * np.prod(np.where(self.ma_p > 0, cc[None, :] ** self.ma_p, 1.0), axis=1)
        return fwd - rev

    def jacobian(self, c, env, ks=None):
        """Analytic d(dc/dt)/dc (avoids scipy's finite-difference loop on species with zero sensitivity, e.g. solvents)."""
        kf, kr = ks or self._ks(env)
        cc = np.maximum(c, 0.0); nS = len(cc); nR = len(self.reactions)
        dr = np.zeros((nR, nS))
        for j in range(nR):
            for ma, k, sign in ((self.ma_r[j], kf[j], 1.0), (self.ma_p[j], kr[j], -1.0)):
                if k == 0.0: continue
                for i in np.nonzero(ma)[0]:
                    e = ma.copy(); e[i] -= 1
                    dr[j, i] += sign * k * ma[i] * np.prod(np.where(e > 0, cc ** e, 1.0))
        return self.S @ dr

    # ---- deterministic ---------------------------------------------------------------------------
    def integrate(self, c0, env, t_end, n_out=200, rtol=1e-9, atol=1e-14, max_nfev=2_000_000):
        """Stiff integration (BDF with analytic Jacobian; LSODA fallback). Raises if neither converges within the budget."""
        c0 = np.asarray(c0, float); ks = self._ks(env); last = None
        for method in ("BDF", "LSODA"):
            n = [0]
            def rhs(t, c):
                n[0] += 1
                if n[0] > max_nfev: raise RuntimeError("integration budget exceeded")
                return self.S @ self.rates(c, env, ks)
            try:
                sol = solve_ivp(rhs, (0, t_end), c0, method=method, jac=lambda t, c: self.jacobian(c, env, ks),
                                t_eval=np.linspace(0, t_end, n_out), rtol=rtol, atol=atol)
            except RuntimeError as e:
                last = str(e); continue
            if sol.success: return sol.t, sol.y.T
            last = sol.message
        raise RuntimeError(last)

    def conservation_error(self, c_traj):
        """max relative drift of every element total and of net charge along a trajectory (concentration units)."""
        out = {}
        for k, e in enumerate(self.elements):
            tot = c_traj @ self.E[k]; out[e] = float(np.max(np.abs(tot - tot[0])) / max(abs(tot[0]), 1e-300))
        qt = c_traj @ self.q
        out["charge"] = float(np.max(np.abs(qt - qt[0])) / max(np.max(np.abs(c_traj @ np.abs(self.q))) , 1e-300))
        return out

    # ---- stochastic (Gillespie direct method) ----------------------------------------------------------
    def gillespie(self, counts0, env, volume_l, t_end, seed=0, max_events=5_000_000, record_dt=None):
        rng = np.random.default_rng(seed)
        n = np.asarray(counts0, np.int64).copy(); kf, kr = self._ks(env)
        nR = len(self.reactions); orders_f = self.ma_r.sum(1); orders_r = self.ma_p.sum(1)
        fac_f = np.array([np.prod([factorial(int(v)) for v in row]) for row in self.ma_r])
        fac_r = np.array([np.prod([factorial(int(v)) for v in row]) for row in self.ma_p])
        cf = kf * fac_f / (N_A * volume_l) ** (orders_f - 1); cr = kr * fac_r / (N_A * volume_l) ** (orders_r - 1)
        t = 0.0; times = [0.0]; traj = [n.copy()]; next_rec = record_dt or np.inf; events = 0
        d = self.S.T.astype(np.int64)
        while t < t_end and events < max_events:
            af = cf * np.array([np.prod([comb(int(n[i]), int(self.ma_r[j, i])) for i in np.nonzero(self.ma_r[j])[0]]) for j in range(nR)], float)
            ar = cr * np.array([np.prod([comb(int(n[i]), int(self.ma_p[j, i])) for i in np.nonzero(self.ma_p[j])[0]]) for j in range(nR)], float)
            a = np.concatenate([af, ar]); a0 = a.sum()
            if a0 <= 0: t = t_end; break
            t += rng.exponential(1.0 / a0)
            if t > t_end: break
            m = rng.choice(2 * nR, p=a / a0)
            n += d[m] if m < nR else -d[m - nR]
            events += 1
            if record_dt:
                while next_rec <= t: times.append(next_rec); traj.append(n.copy()); next_rec += record_dt
        times.append(t_end); traj.append(n.copy())
        return np.array(times), np.array(traj), events
