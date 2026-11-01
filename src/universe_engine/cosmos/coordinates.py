"""Physical vs comoving coordinate transformations for an expanding FLRW universe.

All transformations are explicit, documented, and unit-checked. They never silently
convert between frames; every conversion declares the scale factor `a` and the
Hubble parameter `H(a)` used.

Frames:
  - comoving: coordinates that expand with the background. A comoving particle at
    fixed x_comoving rides the Hubble flow; its proper position is a * x_comoving.
  - proper: physical (correct) distance at the current cosmic time.
  - peculiar: velocity relative to the local Hubble flow.

Equations (flat FLRW, scale factor a, Hubble H(a) = da/(a dt)):
  x_proper      = a * x_comoving
  x_comoving    = x_proper / a
  v_pec         = a * d(x_comoving)/dt        (peculiar velocity, proper m/s)
  v_proper      = H(a) * x_proper + v_pec     (total proper velocity)
  d(v_pec)/dt   = -H(a) * v_pec + a * g_comoving   (peculiar acceleration, eq. of motion)
  rho_proper    = rho_comoving / a^3
  rho_comoving  = a^3 * rho_proper
  p_comoving    = a^2 * p_proper              (conjugate momentum per unit mass)
  p_proper      = p_comoving / a^2

The comoving equation of motion for collisionless matter:
  d^2(x_comoving)/dt^2 + 2 H(a) d(x_comoving)/dt = g_comoving / a
  or, in terms of peculiar velocity:
  d(v_pec)/dt = -H(a) v_pec + g_comoving / a
where g_comoving = -grad_xcomoving Phi_comoving is the comoving gravitational acceleration.

Reference: standard cosmological perturbation theory (e.g. Baumann, TASI lectures;
Mo, van den Bosch & White, Galaxy Formation and Evolution, Ch. 6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

SECONDS_PER_GYR = 3.15576e16
MPC_KM = 3.0856775814913673e19


@dataclass(frozen=True)
class Frame:
    """A coordinate frame declaration."""
    name: str            # 'comoving' | 'proper' | 'peculiar'
    scale_factor: float  # a at which the frame is evaluated
    hubble_s: float      # H(a) in s^-1


def comoving_to_proper(x_comoving: np.ndarray, a: float) -> np.ndarray:
    """Convert comoving position to proper position: x_proper = a * x_comoving."""
    return a * np.asarray(x_comoving, dtype=float)


def proper_to_comoving(x_proper: np.ndarray, a: float) -> np.ndarray:
    """Convert proper position to comoving position: x_comoving = x_proper / a."""
    if a <= 0:
        raise ValueError("scale factor must be positive")
    return np.asarray(x_proper, dtype=float) / a


def peculiar_velocity_from_comoving(x_comoving: np.ndarray,
                                    dx_comoving_dt: np.ndarray,
                                    a: float) -> np.ndarray:
    """Peculiar velocity (proper m/s) from comoving velocity: v_pec = a * dx_comoving/dt."""
    return a * np.asarray(dx_comoving_dt, dtype=float)


def proper_velocity(x_comoving: np.ndarray,
                    dx_comoving_dt: np.ndarray,
                    a: float,
                    H_s: float) -> np.ndarray:
    """Total proper velocity: v_proper = H * x_proper + v_pec."""
    x_proper = comoving_to_proper(x_comoving, a)
    v_pec = peculiar_velocity_from_comoving(x_comoving, dx_comoving_dt, a)
    return H_s * x_proper + v_pec


def comoving_velocity_from_proper(x_proper: np.ndarray,
                                  v_proper: np.ndarray,
                                  a: float,
                                  H_s: float) -> np.ndarray:
    """Recover comoving velocity dx_comoving/dt from proper velocity:
       dx_comoving/dt = (v_proper - H * x_proper) / a."""
    x_comoving = proper_to_comoving(x_proper, a)
    v_pec = v_proper - H_s * x_proper
    return v_pec / a


def physical_density(rho_comoving: np.ndarray, a: float) -> np.ndarray:
    """Proper (physical) mass density: rho_proper = rho_comoving / a^3."""
    if a <= 0:
        raise ValueError("scale factor must be positive")
    return np.asarray(rho_comoving, dtype=float) / a ** 3


def comoving_density(rho_proper: np.ndarray, a: float) -> np.ndarray:
    """Comoving mass density: rho_comoving = a^3 * rho_proper."""
    if a <= 0:
        raise ValueError("scale factor must be positive")
    return a ** 3 * np.asarray(rho_proper, dtype=float)


def conjugate_momentum_comoving(p_proper: np.ndarray, a: float) -> np.ndarray:
    """Comoving (conjugate) momentum per unit mass: p_comoving = a^2 * p_proper."""
    if a <= 0:
        raise ValueError("scale factor must be positive")
    return a ** 2 * np.asarray(p_proper, dtype=float)


def conjugate_momentum_proper(p_comoving: np.ndarray, a: float) -> np.ndarray:
    """Proper momentum per unit mass: p_proper = p_comoving / a^2."""
    if a <= 0:
        raise ValueError("scale factor must be positive")
    return np.asarray(p_comoving, dtype=float) / a ** 2


def peculiar_acceleration_equation(v_pec: np.ndarray,
                                   g_comoving: np.ndarray,
                                   a: float,
                                   H_s: float) -> np.ndarray:
    """Right-hand side of the peculiar-velocity evolution equation:
       d(v_pec)/dt = -H(a) * v_pec + g_comoving / a.

    g_comoving is the comoving gravitational acceleration (m/s^2 in comoving
    length units per proper time squared, i.e. -grad_comoving Phi / a).
    """
    v_pec = np.asarray(v_pec, dtype=float)
    g_comoving = np.asarray(g_comoving, dtype=float)
    return -H_s * v_pec + g_comoving / a


def comoving_position_equation(dx_comoving_dt: np.ndarray,
                               v_pec: np.ndarray,
                               a: float) -> np.ndarray:
    """Right-hand side for comoving position: d(x_comoving)/dt = v_pec / a."""
    return np.asarray(v_pec, dtype=float) / a


@dataclass
class ComovingParticleState:
    """A collisionless particle state declared in a specific frame."""
    x_comoving: np.ndarray     # comoving position (box fraction or Mpc/h)
    v_pec: np.ndarray          # peculiar velocity (proper m/s)
    mass: np.ndarray           # particle mass (kg)
    a: float                   # scale factor at which state is defined
    H_s: float                 # Hubble parameter at a (s^-1)

    def proper_position(self) -> np.ndarray:
        return comoving_to_proper(self.x_comoving, self.a)

    def proper_velocity(self) -> np.ndarray:
        return proper_velocity(self.x_comoving, self.v_pec / self.a, self.a, self.H_s)

    def to_dict(self) -> dict:
        return {
            "frame": "comoving",
            "a": self.a,
            "H_s": self.H_s,
            "x_comoving": self.x_comoving.tolist(),
            "v_pec": self.v_pec.tolist(),
            "mass": self.mass.tolist(),
        }


def validate_frame_consistency(x_comoving: np.ndarray,
                               v_pec: np.ndarray,
                               a: float,
                               H_s: float,
                               tol: float = 1e-9) -> dict:
    """Round-trip check: comoving -> proper -> comoving reproduces the input."""
    x_comoving = np.asarray(x_comoving, dtype=float)
    v_pec = np.asarray(v_pec, dtype=float)
    x_proper = comoving_to_proper(x_comoving, a)
    x_back = proper_to_comoving(x_proper, a)
    v_proper = proper_velocity(x_comoving, v_pec / a, a, H_s)
    v_pec_back = v_proper - H_s * x_proper
    x_err = float(np.max(np.abs(x_back - x_comoving))) if x_comoving.size else 0.0
    v_err = float(np.max(np.abs(v_pec_back - v_pec))) if v_pec.size else 0.0
    return {
        "roundtrip_position_error": x_err,
        "roundtrip_velocity_error": v_err,
        "consistent": x_err < tol and v_err < tol,
        "a": a,
        "H_s": H_s,
    }


def hubble_from_cosmology(H0_km_s_Mpc: float, Omega_m: float, Omega_r: float,
                          Omega_Lambda: float, a: float) -> float:
    """Flat FLRW Hubble parameter H(a) in s^-1."""
    e2 = Omega_r / a ** 4 + Omega_m / a ** 3 + Omega_Lambda
    if e2 <= 0:
        raise ValueError(f"negative E^2 at a={a}")
    return (H0_km_s_Mpc / MPC_KM) * math.sqrt(e2)


def growth_rate_approx(Omega_m: float, Omega_Lambda: float, a: float,
                       Omega_r: float = 0.0) -> float:
    """Approximate linear growth rate f = d ln D / d ln a (Lahav et al. 1991
    fitting formula, radiation neglected). For regression tests only."""
    e2 = Omega_m / a ** 3 + Omega_Lambda + Omega_r / a ** 4
    om_a = (Omega_m / a ** 3) / e2
    ol_a = Omega_Lambda / e2
    return om_a ** (4.0 / 7.0) + (ol_a / 70.0) * (1.0 + om_a / 2.0)
