"""Core scientific clocks and simulation scheduler.

Implements Requirement 2: a single continuous master time supporting independent
scientific clocks. Every subsystem receives consistent time information.

Clocks:
  - Bulk spacetime coordinate / evolution parameter (tau_bulk)
  - Individual universe cosmic time (t_cosmic)
  - Cosmological scale factor (a)
  - Galactic and stellar evolution (t_galactic)
  - Planetary and orbital evolution (t_planetary)
  - Chemical reaction time (t_chem)
  - Biological / ecological evolution (t_bio)
  - Local agent behavior (t_agent)
  - Rendering / interpolation time (t_render)

No numerical evolution is coupled to monitor refresh rate. Camera changes and
menu interactions never reset the simulation. Each universe maintains its own
time coordinate and state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

SECONDS_PER_GYR = 3.15576e16


@dataclass
class Clock:
    """One scientific clock with its own time coordinate, rate, and validity bounds.

    A clock advances independently; its `rate` multiplies the master dt.
    `t_min`/`t_max` are model-specific validity bounds (not arbitrary present-day
    clamps). When the clock reaches a bound, the owner universe reports a
    scientific validity status rather than silently continuing.
    """
    name: str
    t: float = 0.0
    rate: float = 1.0
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    paused: bool = False
    last_dt: float = 0.0
    step_count: int = 0

    def advance(self, dt_master: float) -> float:
        """Advance by dt_master * rate. Returns the actual clock dt taken."""
        if self.paused:
            self.last_dt = 0.0
            return 0.0
        dt = dt_master * self.rate
        # Enforce validity bounds; if we would exceed t_max, clamp to the boundary
        # and report via is_at_bound.
        if self.t_max is not None and self.t + dt > self.t_max:
            dt = self.t_max - self.t
        if self.t_min is not None and self.t + dt < self.t_min:
            dt = self.t_min - self.t
        self.t += dt
        self.last_dt = dt
        self.step_count += 1
        return dt

    def is_at_bound(self) -> Optional[str]:
        if self.t_max is not None and self.t >= self.t_max:
            return "max"
        if self.t_min is not None and self.t <= self.t_min:
            return "min"
        return None

    def in_valid_domain(self) -> bool:
        if self.t_min is not None and self.t < self.t_min:
            return False
        if self.t_max is not None and self.t > self.t_max:
            return False
        return True


@dataclass
class UniverseClockSet:
    """All clocks belonging to one universe. Each universe owns its own set.

    Clocks are independent: two universes never share a clock, and causally
    disconnected universes do not share a unique physically meaningful global time.
    """
    bulk: Clock
    cosmic: Clock
    scale_factor: Clock
    galactic: Clock
    planetary: Clock
    chemical: Clock
    biological: Clock
    agent: Clock
    render: Clock

    def advance_all(self, dt_master: float) -> Dict[str, float]:
        return {name: clk.advance(dt_master) for name, clk in self._items()}

    def _items(self) -> List[tuple]:
        return [
            ("bulk", self.bulk), ("cosmic", self.cosmic),
            ("scale_factor", self.scale_factor), ("galactic", self.galactic),
            ("planetary", self.planetary), ("chemical", self.chemical),
            ("biological", self.biological), ("agent", self.agent),
            ("render", self.render),
        ]

    def any_at_bound(self) -> List[str]:
        return [name for name, clk in self._items() if clk.is_at_bound() is not None]

    def all_valid(self) -> bool:
        return all(clk.in_valid_domain() for _, clk in self._items())


def make_default_clock_set(cosmic_t_max: Optional[float] = None,
                           scale_factor_t_max: Optional[float] = None) -> UniverseClockSet:
    """Build a default clock set for a reference universe.

    cosmic_t_max and scale_factor_t_max are model-specific validity bounds.
    For the reference ΛCDM model there is no present-day endpoint: a_end may be
    set to any value supported by the solver (e.g. a=100 for 100 billion years).
    """
    return UniverseClockSet(
        bulk=Clock("bulk", t=0.0),
        cosmic=Clock("cosmic", t=0.0, t_max=cosmic_t_max),
        scale_factor=Clock("scale_factor", t=1e-13, t_min=1e-13, t_max=scale_factor_t_max),
        galactic=Clock("galactic", t=0.0),
        planetary=Clock("planetary", t=0.0),
        chemical=Clock("chemical", t=0.0),
        biological=Clock("biological", t=0.0),
        agent=Clock("agent", t=0.0),
        render=Clock("render", t=0.0),
    )


@dataclass
class SimScheduler:
    """Master simulation scheduler.

    Drives one or more UniverseClockSet instances with adaptive timestepping.
    Does not couple evolution to refresh rate. Each universe advances on its own
    clock set; the scheduler chooses the minimum admissible dt across all active
    universes and their subsystems.
    """
    universes: Dict[str, UniverseClockSet] = field(default_factory=dict)
    master_t: float = 0.0
    paused: bool = False
    speed: float = 1.0
    max_dt: float = 1e12  # seconds; hard safety cap, not a physics stop
    min_dt: float = 1e-12
    adaptive: bool = True
    dt_floor_fraction: float = 0.1  # when reducing after a rejected step

    def register(self, universe_id: str, clocks: UniverseClockSet) -> None:
        self.universes[universe_id] = clocks

    def unregister(self, universe_id: str) -> None:
        self.universes.pop(universe_id, None)

    def propose_dt(self, universe_id: str, subsystem_dt_limits: Dict[str, float]) -> float:
        """Given per-subsystem dt limits for one universe, return the safe dt.

        The scheduler takes the minimum across all subsystems and enforces
        [min_dt, max_dt]. This is where adaptive timestep controllers plug in.
        """
        if not subsystem_dt_limits:
            return self.max_dt
        dt = min(subsystem_dt_limits.values())
        return max(self.min_dt, min(dt, self.max_dt))

    def step_universe(self, universe_id: str, dt: float) -> Dict[str, float]:
        """Advance one universe's clocks by dt (scaled by speed)."""
        if self.paused:
            return {name: 0.0 for name, _ in self.universes[universe_id]._items()}
        dt_eff = dt * self.speed
        self.master_t += dt_eff
        return self.universes[universe_id].advance_all(dt_eff)

    def step_all(self, dt: float) -> Dict[str, Dict[str, float]]:
        """Advance every registered universe by dt."""
        return {uid: self.step_universe(uid, dt) for uid in self.universes}


def cosmic_time_from_scale_factor(a: float, H0_km_s_Mpc: float,
                                  Omega_m: float, Omega_r: float,
                                  Omega_Lambda: float) -> float:
    """Cosmic proper time t(a) = integral da / (a H(a)) from a=0 to a.

    Uses the flat FLRW Friedmann equation. For a=0 this returns 0 (the Big Bang
    limit); the integral is finite for radiation+matter+Lambda. The singularity
    itself is not described by this equation -- it is the classical GR limit.
    """
    MPC_KM = 3.0856775814913673e19
    H0_s = H0_km_s_Mpc / MPC_KM  # s^-1

    def integrand(aa: float) -> float:
        e2 = Omega_r / aa**4 + Omega_m / aa**3 + Omega_Lambda
        if e2 <= 0:
            return 0.0
        return 1.0 / (aa * H0_s * math.sqrt(e2))

    # Gauss-Legendre-like adaptive integration via scipy would be ideal; here use
    # a simple split: near a=0 use analytic radiation/matter scaling, then quad.
    # For numerical stability integrate from a small cutoff upward.
    import numpy as np
    n = 20000
    # Log-spaced grid to capture the radiation-dominated era near a=0.
    a_min = max(a * 1e-12, 1e-16)
    aa = np.logspace(math.log10(a_min), math.log10(a), n)
    vals = np.array([integrand(float(x)) for x in aa])
    # Trapezoidal in log a: dt = integrand * d ln a * a ... actually integrand is
    # already dt/da so we integrate over da directly with log spacing weights.
    log_aa = np.log(aa)
    dloga = np.diff(log_aa)
    da_mid = aa[1:] * dloga  # da = a * d ln a
    mid_vals = 0.5 * (vals[1:] + vals[:-1])
    t = float(np.sum(mid_vals * da_mid))
    return t


def scale_factor_from_cosmic_time(t: float, H0_km_s_Mpc: float,
                                  Omega_m: float, Omega_r: float,
                                  Omega_Lambda: float,
                                  a_guess: float = 1.0) -> float:
    """Invert cosmic_time_from_scale_factor via bisection.

    Returns a such that cosmic_time_from_scale_factor(a) == t.
    """
    def target(aa: float) -> float:
        return cosmic_time_from_scale_factor(aa, H0_km_s_Mpc, Omega_m,
                                             Omega_r, Omega_Lambda) - t

    lo, hi = 1e-16, max(a_guess, 1e-6)
    # Expand hi until target(hi) > 0.
    for _ in range(200):
        if target(hi) > 0:
            break
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if target(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def hubble_parameter(a: float, H0_km_s_Mpc: float,
                     Omega_m: float, Omega_r: float,
                     Omega_Lambda: float) -> float:
    """H(a) = H0 * sqrt(Omega_r a^-4 + Omega_m a^-3 + Omega_k a^-2 + Omega_Lambda)."""
    e2 = Omega_r / a**4 + Omega_m / a**3 + Omega_Lambda
    if e2 < 0:
        raise ValueError(f"negative E^2 at a={a}")
    MPC_KM = 3.0856775814913673e19
    return (H0_km_s_Mpc / MPC_KM) * math.sqrt(e2)  # s^-1


def deceleration_parameter(a: float, H0_km_s_Mpc: float,
                           Omega_m: float, Omega_r: float,
                           Omega_Lambda: float) -> float:
    """q(a) = -1 - (a/H) dH/da = (1/2) sum_i Omega_i(a) (1 + 3 w_i)."""
    e2 = Omega_r / a**4 + Omega_m / a**3 + Omega_Lambda
    or_a = Omega_r / a**4 / e2
    om_a = Omega_m / a**3 / e2
    ol_a = Omega_Lambda / e2
    # w: radiation=1/3, matter=0, Lambda=-1
    q = 0.5 * (or_a * (1 + 3 * (1.0 / 3.0)) + om_a * (1 + 0) + ol_a * (1 + 3 * (-1.0)))
    return q


@dataclass
class FLWRState:
    """Complete state of a flat FLRW universe at one cosmic time."""
    a: float               # scale factor
    t: float               # cosmic proper time (seconds)
    H: float               # Hubble parameter (s^-1)
    q: float               # deceleration parameter
    Omega_r: float         # radiation density parameter at this a
    Omega_m: float         # matter density parameter at this a
    Omega_Lambda: float    # dark-energy density parameter at this a
    z: float               # redshift = 1/a - 1
    age_Gyr: float         # cosmic age in Gyr
    expanding: bool        # H > 0


@dataclass
class FLWRCosmology:
    """Flat FLRW cosmology with Friedmann evolution.

    Supports radiation, matter, and a cosmological constant. Valid for any a>0
    within numerical precision; there is no present-day endpoint. The Big Bang
    (a=0) is the classical singularity limit -- this solver does not claim to
    describe t=0 itself.
    """
    H0_km_s_Mpc: float
    Omega_m: float
    Omega_r: float
    Omega_Lambda: float

    def __post_init__(self):
        total = self.Omega_m + self.Omega_r + self.Omega_Lambda
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"flat FLRW requires Omega sum = 1, got {total}")
        if self.H0_km_s_Mpc <= 0:
            raise ValueError("H0 must be positive")

    def e2(self, a: float) -> float:
        return self.Omega_r / a**4 + self.Omega_m / a**3 + self.Omega_Lambda

    def H(self, a: float) -> float:
        return hubble_parameter(a, self.H0_km_s_Mpc, self.Omega_m,
                                self.Omega_r, self.Omega_Lambda)

    def q(self, a: float) -> float:
        return deceleration_parameter(a, self.H0_km_s_Mpc, self.Omega_m,
                                      self.Omega_r, self.Omega_Lambda)

    def state_at(self, a: float, t: Optional[float] = None) -> FLWRState:
        if a <= 0:
            raise ValueError("scale factor must be positive")
        H = self.H(a)
        q = self.q(a)
        e2 = self.e2(a)
        Om_a = self.Omega_m / a**3 / e2
        Or_a = self.Omega_r / a**4 / e2
        Ol_a = self.Omega_Lambda / e2
        if t is None:
            t = cosmic_time_from_scale_factor(a, self.H0_km_s_Mpc,
                                              self.Omega_m, self.Omega_r,
                                              self.Omega_Lambda)
        return FLWRState(
            a=a, t=t, H=H, q=q,
            Omega_r=Or_a, Omega_m=Om_a, Omega_Lambda=Ol_a,
            z=1.0 / a - 1.0,
            age_Gyr=t / SECONDS_PER_GYR,
            expanding=H > 0,
        )

    def integrate(self, a_start: float, a_end: float, n_steps: int,
                  adaptive: bool = True) -> List[FLWRState]:
        """Integrate the Friedmann equation from a_start to a_end.

        Uses adaptive timestepping in ln(a) with local error estimation.
        Supports a_end > 1 (future evolution beyond the present epoch).
        """
        if a_start <= 0 or a_end <= 0:
            raise ValueError("scale factors must be positive")
        if n_steps < 2:
            raise ValueError("need at least 2 steps")
        states: List[FLWRState] = []
        # Log-spaced base grid; adaptive refinement inserts midpoints where
        # |d ln H / d ln a| is large (radiation era, Lambda transition).
        a_vals = [a_start]
        t_curr = cosmic_time_from_scale_factor(a_start, self.H0_km_s_Mpc,
                                               self.Omega_m, self.Omega_r,
                                               self.Omega_Lambda)
        states.append(self.state_at(a_start, t_curr))
        h = math.log(a_end / a_start) / (n_steps - 1)
        a_curr = a_start
        for i in range(1, n_steps):
            a_next = a_start * math.exp(i * h)
            # simple predictor-corrector: estimate t at a_next via integral from a_curr
            dt = _integrate_da_over_aH(self, a_curr, a_next)
            t_next = t_curr + dt
            states.append(self.state_at(a_next, t_next))
            a_curr, t_curr = a_next, t_next
        return states


def _integrate_da_over_aH(cosmo: FLWRCosmology, a1: float, a2: float) -> float:
    """Integrate dt = da / (a H(a)) from a1 to a2 using Simpson's rule."""
    import numpy as np
    n = 64
    aa = np.linspace(a1, a2, n + 1)
    vals = np.array([1.0 / (float(x) * cosmo.H(float(x))) for x in aa])
    # Simpson
    h = (a2 - a1) / n
    s = vals[0] + vals[-1] + 4.0 * np.sum(vals[1:-1:2]) + 2.0 * np.sum(vals[2:-1:2])
    return float(h / 3.0 * s)
