"""Physical constants (SI) and the provenance of every empirical parameter used by the planet model.

PROVENANCE honesty rule: values marked 'recalled' were written from memory of standard references and have NOT been
re-checked against the primary source in this environment. They are parameters of approximate models, not measurements."""
SIGMA = 5.670374419e-8      # W m-2 K-4, Stefan-Boltzmann (CODATA 2018, exact in SI 2019)
G = 6.67430e-11             # m3 kg-1 s-2 (CODATA 2018)
AU = 1.495978707e11         # m (IAU 2012, exact)
L_SUN = 3.828e26            # W (IAU 2015 nominal)
M_EARTH = 5.9722e24         # kg
R_EARTH = 6.371e6           # m (mean)
M_SUN = 1.98841e30          # kg
YEAR = 3.15576e7            # s (Julian)
GYR = 1e9 * YEAR
R_GAS = 8.314462618         # J mol-1 K-1
T_TRIPLE = 273.16           # K
P_TRIPLE = 611.657          # Pa

PROVENANCE = {
    "budyko_olr": "OLR = A + B*T[degC], A=203.3 W/m2, B=2.09 W/m2/K (Budyko 1969; North 1975). recalled; linear, valid ~ -50..+40 C only.",
    "co2_forcing": "dF = 5.35*ln(C/C0) W/m2 (Myhre et al. 1998). recalled; logarithmic law, valid for ~0.1-10x present CO2.",
    "diffusion": "meridional heat diffusion D = 0.55 W/m2/K (North 1975 order of magnitude). recalled; tuned, not predicted.",
    "ice_albedo": "alpha_ice=0.62, alpha_open=0.32, switch at -10 C (North 1975 style). recalled; Earth-tuned.",
    "mixed_layer_c": "ocean mixed layer 70 m: C = rho*cp*h = 2.9e8 J/m2/K. computed from water properties.",
    "weathering": "W = W0 (pCO2/p0)^0.3 exp(0.09 (T-T0)) (Walker, Hays & Kasting 1981). recalled; Earth-calibrated, only for mobile-lid + liquid water.",
    "outgassing": "volcanic CO2 outgassing scales with radiogenic+secular heat flux (assumed proportional). assumption, not derived.",
    "radiogenic_heat": "present-day heat share U238 0.39, U235 0.02, Th232 0.42, K40 0.17; half-lives 4.468, 0.704, 14.05, 1.248 Gyr. recalled.",
    "gough_luminosity": "L(t)=L_ms/(1+0.4(1-t/t_ref)) (Gough 1981) with t_ref=4.57 Gyr scaled by main-sequence lifetime. approximate for the Sun, extrapolated to other stars.",
    "mass_luminosity": "L/Lsun = (M/Msun)^3.5, t_ms = 10 Gyr * M/L. textbook approximation for ~0.4-2 Msun.",
    "mass_radius": "R/R_earth = (M/M_earth)^0.27 for rocky planets of Earth-like composition (~0.5-5 M_earth). approximate fit.",
    "runaway_threshold": "absorbed flux > ~280 W/m2 treated as moist/runaway greenhouse regime (Goldblatt et al. 2013 order of magnitude). recalled; model invalid beyond it.",
    "carbonate_chemistry": "Henry KH(298)=3.4e-2 mol/L/atm (dH/R=2400 K), K1(298)=4.45e-7, K2=4.69e-11, Kw=1e-14 (temperature dependence of K1,K2,Kw neglected). recalled textbook values.",
    "carbon_timescale": "lumped atmosphere-ocean carbon response rate 2000/Gyr (0.5 Myr) is an ASSUMPTION; only the equilibrium (out = weathering) is meaningful, not the transient speed.",
    "tectonics": "mobile vs stagnant lid is an INPUT, not predicted.",
}
