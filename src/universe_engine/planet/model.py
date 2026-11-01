"""Coupled planet model: stellar evolution -> insolation -> 1D latitudinal energy balance -> surface state;
carbon-silicate cycle (outgassing vs temperature-dependent weathering) -> pCO2 -> carbonate pH; water phase; energy sources.

All outputs are model results under the labelled approximations in constants.PROVENANCE. Nothing here creates life."""
from dataclasses import dataclass, field, asdict
import numpy as np
from . import constants as K


@dataclass
class StarSpec:
    mass_msun: float = 1.0
    age_gyr_at_start: float = 0.0

    @property
    def luminosity_ms_lsun(self):
        return self.mass_msun ** 3.5

    @property
    def lifetime_gyr(self):
        return 10.0 * self.mass_msun / self.luminosity_ms_lsun

    def luminosity(self, t_gyr):
        """W. Gough-type brightening, with time scaled so the Sun's 4.57 Gyr corresponds to the L_sun fraction."""
        tref = 4.57 * self.lifetime_gyr / 10.0
        # normalise so that for the Sun L(4.57)=L_sun
        return K.L_SUN * self.luminosity_ms_lsun * 1.0 / (1 + 0.4 * (1 - t_gyr / tref))


@dataclass
class PlanetSpec:
    mass_mearth: float = 1.0
    a_au: float = 1.0
    ecc: float = 0.0167
    obliquity_deg: float = 23.44
    ocean_fraction: float = 0.71          # surface covered by liquid-capable ocean/land fraction used for heat capacity
    albedo_cloud_open: float = 0.32       # planetary albedo of ice-free surface incl. clouds (Earth-tuned)
    pco2_bar: float = 2.8e-4              # initial CO2 partial pressure
    pn2_bar: float = 1.0                  # background gas, no radiative role in this model
    alkalinity_eq_per_l: float = 2.3e-3   # fixed ocean alkalinity (diagnostic only)
    tectonics: str = "mobile"             # "mobile" | "stagnant"  (INPUT, not predicted)
    has_surface_water: bool = True
    star: StarSpec = field(default_factory=StarSpec)
    seed: int = 0

    @property
    def radius_m(self):
        return K.R_EARTH * self.mass_mearth ** 0.27

    @property
    def gravity(self):
        return K.G * self.mass_mearth * K.M_EARTH / self.radius_m ** 2


def annual_mean_insolation_shape(x, obliquity_deg, n_season=720):
    """Annual-mean daily-mean insolation at latitude sin(phi)=x relative to S/4 (circular orbit), integrated numerically."""
    eps = np.radians(obliquity_deg)
    lam = (np.arange(n_season) + 0.5) / n_season * 2 * np.pi
    sd = np.sin(eps) * np.sin(lam); delta = np.arcsin(sd)
    phi = np.arcsin(np.clip(x, -1, 1))[:, None]
    tt = -np.tan(phi) * np.tan(delta)[None, :]
    h0 = np.where(tt >= 1, 0.0, np.where(tt <= -1, np.pi, np.arccos(np.clip(tt, -1, 1))))
    q = (h0 * np.sin(phi) * np.sin(delta)[None, :] + np.cos(phi) * np.cos(delta)[None, :] * np.sin(h0)) / np.pi
    s = q.mean(axis=1)
    w = 1.0 / len(x)                       # x-uniform grid => area weights equal
    return s / (s.sum() * w)               # normalised so global (area) mean = 1


def water_boiling_k(p_pa):
    """Boiling temperature from a Clausius-Clapeyron fit anchored at the triple point and at 373.15 K / 1 atm (approximate between/beyond the anchors)."""
    if p_pa < K.P_TRIPLE:
        return np.nan
    # effective latent heat fixed by the two measured anchors (triple point; 373.15 K at 101325 Pa) -> exact at both
    Rv = 461.5
    L = Rv * np.log(101325.0 / K.P_TRIPLE) / (1.0 / K.T_TRIPLE - 1.0 / 373.15)
    return 1.0 / (1.0 / K.T_TRIPLE - Rv / L * np.log(p_pa / K.P_TRIPLE))


def carbonate_ph(pco2_bar, alk, T_k=288.0):
    """Solve charge balance Alk = [HCO3-]+2[CO3--]+[OH-]-[H+] for pH at fixed alkalinity (diagnostic)."""
    KH = 3.4e-2 * np.exp(2400.0 * (1 / T_k - 1 / 298.15)); K1, K2, Kw = 4.45e-7, 4.69e-11, 1e-14
    co2 = KH * pco2_bar
    lo, hi = 0.0, 14.0
    for _ in range(80):
        ph = 0.5 * (lo + hi); h = 10 ** -ph
        f = K1 * co2 / h + 2 * K1 * K2 * co2 / h ** 2 + Kw / h - h - alk
        if f > 0: hi = ph      # f increases with pH; f>0 means pH too high
        else: lo = ph
    return 0.5 * (lo + hi)


class Planet:
    N = 36
    def __init__(self, spec: PlanetSpec, t_gyr=4.5):
        self.spec = spec
        self.t_gyr = t_gyr
        self.x = -1 + (np.arange(self.N) + 0.5) * 2 / self.N
        self.xe = -1 + np.arange(1, self.N) * 2 / self.N
        self.shape = annual_mean_insolation_shape(self.x, spec.obliquity_deg)
        self.T = np.full(self.N, 288.0)
        self.pco2 = spec.pco2_bar
        self.history = []
        self._C = (spec.ocean_fraction * 2.9e8 + (1 - spec.ocean_fraction) * 1.0e7) * np.ones(self.N)
        self._D = 0.55
        self._equilibrate(spec_only=True)

    # ---- radiation -------------------------------------------------------------------------
    def stellar_flux(self):
        L = self.spec.star.luminosity(self.t_gyr)
        a = self.spec.a_au * K.AU
        return L / (4 * np.pi * a ** 2) / np.sqrt(1 - self.spec.ecc ** 2)      # annual-mean at orbit

    def albedo(self, T):
        sp = self.spec
        ice = 0.5 * (1 - np.tanh((T - 263.15) / 2.0))
        if not sp.has_surface_water:                    # dry world: no ice-albedo feedback in this model
            ice = ice * 0.0
        return sp.albedo_cloud_open + (0.62 - sp.albedo_cloud_open) * ice

    def olr(self, T):
        Tc = T - 273.15
        return 203.3 + 2.09 * Tc - 5.35 * np.log(max(self.pco2, 1e-9) / 2.8e-4)

    def _equilibrate(self, spec_only=False, tol=1e-5, max_iter=4000, dt_yr=0.5):
        N = self.N; dt = dt_yr * K.YEAR
        S = self.stellar_flux()
        Q = S / 4.0 * self.shape
        # conservative diffusion operator on x grid: D d/dx[(1-x^2) dT/dx]
        h = 2.0 / N
        Lp = np.zeros((N, N))
        for i in range(N - 1):
            c = self._D * (1 - self.xe[i] ** 2) / h ** 2
            Lp[i, i] -= c; Lp[i, i + 1] += c; Lp[i + 1, i + 1] -= c; Lp[i + 1, i] += c
        B = 2.09
        C = np.diag(self._C / dt)
        M = C - Lp + B * np.eye(N)
        T = self.T.copy()
        for it in range(max_iter):
            absorbed = Q * (1 - self.albedo(T))
            rhs = self._C / dt * T + absorbed - (203.3 - 2.09 * 273.15 - 5.35 * np.log(max(self.pco2, 1e-9) / 2.8e-4))
            Tn = np.linalg.solve(M, rhs)
            d = np.max(np.abs(Tn - T)); T = Tn
            if d < tol: break
        self.T = T
        self.converged = d < 1e-3

    # ---- diagnostics -----------------------------------------------------------------------
    def global_mean_T(self):
        return float(self.T.mean())              # x-uniform grid: area weighted

    def energy_imbalance_wm2(self):
        Q = self.stellar_flux() / 4.0 * self.shape
        return float(np.mean(Q * (1 - self.albedo(self.T)) - np.array([self.olr(t) for t in self.T])))

    def absorbed_global_wm2(self):
        return float(np.mean(self.stellar_flux() / 4.0 * self.shape * (1 - self.albedo(self.T))))

    def surface_pressure_pa(self):
        return (self.pco2 + self.spec.pn2_bar) * 1e5

    def liquid_water_fraction(self):
        """Fraction of surface area (area-weighted bands) with T in the liquid range at the surface pressure."""
        if not self.spec.has_surface_water: return 0.0
        tb = water_boiling_k(self.surface_pressure_pa())
        if np.isnan(tb): return 0.0
        return float(np.mean((self.T > K.T_TRIPLE) & (self.T < tb)))

    def regime_flags(self):
        flags = []
        Tg = self.global_mean_T()
        if self.absorbed_global_wm2() > 280.0: flags.append("runaway-greenhouse regime: model invalid (treated sterile)")
        if Tg < 220 or Tg > 330: flags.append("outside linear OLR validity (-50..+57 C global mean)")
        if not (2.8e-5 <= self.pco2 <= 2.8e-2): flags.append("pCO2 outside the 0.1-100x present range where the logarithmic CO2 forcing law is valid")
        return flags

    def geothermal_heat_rel(self, t_gyr=None):
        t = self.t_gyr if t_gyr is None else t_gyr
        w = {"U238": (0.39, 4.468), "U235": (0.02, 0.704), "Th232": (0.42, 14.05), "K40": (0.17, 1.248)}
        now = 4.57
        return sum(f * 2 ** ((now - t) / hl) for f, hl in w.values())    # relative to present-day Earth radiogenic heat

    def environment(self):
        sp = self.spec
        lw = self.liquid_water_fraction()
        Tg = self.global_mean_T()
        Tliq = float(self.T[(self.T > K.T_TRIPLE)].mean()) if lw > 0 else float("nan")
        return {
            "t_gyr": self.t_gyr, "T_global_K": Tg, "T_liquid_mean_K": Tliq, "liquid_water_fraction": lw,
            "pCO2_bar": self.pco2, "surface_pressure_pa": self.surface_pressure_pa(),
            "pH_ocean": carbonate_ph(self.pco2, sp.alkalinity_eq_per_l, Tg) if lw > 0 else float("nan"),
            "absorbed_wm2": self.absorbed_global_wm2(), "energy_imbalance_wm2": self.energy_imbalance_wm2(),
            "geothermal_rel": self.geothermal_heat_rel(), "stellar_flux_wm2": self.stellar_flux(), "flags": self.regime_flags(),
            "tectonics": sp.tectonics, "gravity": sp.gravity}

    # ---- time evolution --------------------------------------------------------------------
    def evolve(self, t_end_gyr, dt_gyr=0.01, record_every=10):
        """Advance the carbon-silicate cycle with the climate in quasi-equilibrium (climate relaxes in ~yr, carbon in ~Myr)."""
        sp = self.spec
        V0 = 1.0   # outgassing normalised so that present-Earth weathering balances it at (T0, p0)
        T0, p0 = 288.0, 2.8e-4
        n = 0
        while self.t_gyr < t_end_gyr - 1e-12:
            dt = min(dt_gyr, t_end_gyr - self.t_gyr)
            self._equilibrate()
            Tg = self.global_mean_T()
            lw = self.liquid_water_fraction()
            if sp.tectonics == "mobile" and sp.has_surface_water:
                out = V0 * self.geothermal_heat_rel() / self.geothermal_heat_rel(4.57)
                wea = lw / 0.889 * (self.pco2 / p0) ** 0.3 * np.exp(0.09 * (Tg - T0)) if lw > 0 else 0.0
                # d ln p / dt in units of 1/Gyr: atmosphere+ocean carbon turnover time ~ 0.5 Myr*... lumped: rate constant 2000/Gyr
                dlnp = 2000.0 * (out - wea) * dt
                self.pco2 = float(np.clip(self.pco2 * np.exp(np.clip(dlnp, -0.5, 0.5)), 1e-6, 5.0))
            self.t_gyr += dt
            n += 1
            if n % record_every == 0:
                self.history.append(self.environment())
        self._equilibrate()
        self.history.append(self.environment())
        return self.history
