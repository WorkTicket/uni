"""3D periodic particle-mesh (PM) gravity in an expanding FLRW background.

Evolves collisionless dark-matter particles in comoving coordinates using the
cosmological equations of motion in `s = ln a`:

    d^2 x / ds^2 + (2 + d ln H / d ln a) dx/ds = -(3/2) Omega_m(a) * grad_x Delta^{-1} delta

where x is the comoving position (box fraction), Delta^{-1} is the periodic inverse
Laplacian (FFT), delta = rho/rho_bar - 1 is the density contrast, and the RHS is
the comoving gravitational acceleration per unit ln-a time squared.

Independent variables:
  - s = ln a (integration variable for particle motion)
  - t = cosmic proper time, obtained by integrating dt = da / (a H(a))
  - a = exp(s) (scale factor)

Physical outputs (proper frame), all requiring a physical box size:
  - x_comoving_Mpc   = box_size_Mpc * x_comoving              [comoving Mpc]
  - x_proper_Mpc     = a * box_size_Mpc * x_comoving          [proper Mpc]
  - x_proper_meters  = a * box_size_Mpc * x_comoving * MPC_METERS [proper m]
  - v_pec_km_s       = a * H(a) * dx/ds * box_size_Mpc * MPC_KM [peculiar km/s]
  - rho_comoving     = mean_comoving_density * (1 + delta)    [M_sun / Mpc^3 comoving]
  - rho_proper       = rho_comoving / a^3                     [M_sun / Mpc^3 proper]
  - delta            = rho/rho_bar - 1                        [dimensionless]

Integration: kick-drift-kick (KDK, 2nd order) in s, with adaptive substeps so
|d ln a| per step stays small. Proper time is integrated alongside using the
FLRW background dt/da = 1/(a H(a)).

This is a minimal reference problem solver: periodic box, single species
(cold dark matter), no gas, no radiation transfer, no baryons. It validates the
comoving equations and force convergence against linear-theory growth.

NOT a research-grade production cosmology solver: no transfer function, no
2LPT initial conditions, no halo finder, no adaptive mesh refinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..cosmos.coordinates import (
    hubble_from_cosmology,
    growth_rate_approx,
    comoving_to_proper,
    physical_density,
    MPC_KM,
    SECONDS_PER_GYR,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MPC_METERS = MPC_KM * 1000.0                      # meters per Mpc
G_SI = 6.67430e-11                                 # m^3 kg^-1 s^-2
MSUN_KG = 1.98847e30                              # kg per solar mass


def critical_density_Msun_Mpc3(H0_km_s_Mpc: float) -> float:
    """Critical density rho_crit = 3 H0^2 / (8 pi G) in M_sun / Mpc^3."""
    H0_s = H0_km_s_Mpc / MPC_KM                     # s^-1
    rho_crit_kg_m3 = 3.0 * H0_s ** 2 / (8.0 * math.pi * G_SI)
    return rho_crit_kg_m3 * MPC_METERS ** 3 / MSUN_KG


def _poisson_gradient_3d(delta: np.ndarray, L: float) -> np.ndarray:
    """Return grad_x Delta^{-1} delta on a periodic 3D grid.

    delta is density contrast (mean subtracted). Returns a (n,n,n,3) array of
    gradients in units of 1/L (derivative w.r.t. box fraction).
    """
    n = delta.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)  # angular freq per box fraction
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    k2 = KX * KX + KY * KY + KZ * KZ
    inv = np.zeros_like(k2)
    nz = k2 > 0
    inv[nz] = 1.0 / k2[nz]
    d_hat = np.fft.fftn(delta - delta.mean())
    grad = np.empty(delta.shape + (3,), dtype=float)
    for i, ki in enumerate((KX, KY, KZ)):
        grad[..., i] = np.fft.ifftn(-1j * ki * inv * d_hat).real
    return grad  # per box fraction


def _cic_deposit_3d(pos: np.ndarray, n: int) -> np.ndarray:
    """Deposit unit-mass particles onto an n^3 grid via CIC, normalized to mean 1."""
    q = (pos % 1.0) * n
    low = np.floor(q).astype(np.int64)
    f = q - low
    rho = np.zeros((n, n, n), dtype=float)
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                w = ((f[:, 0] if ix else 1.0 - f[:, 0])
                     * (f[:, 1] if iy else 1.0 - f[:, 1])
                     * (f[:, 2] if iz else 1.0 - f[:, 2]))
                idx = (low + np.array([ix, iy, iz])) % n
                np.add.at(rho, (idx[:, 0], idx[:, 1], idx[:, 2]), w)
    rho *= n ** 3 / pos.shape[0]
    return rho


def _cic_deposit_mass_3d(pos: np.ndarray, mass: np.ndarray, n: int) -> np.ndarray:
    """Deposit particle masses onto an n^3 grid via CIC, mean = total mass / n^3."""
    q = (pos % 1.0) * n
    low = np.floor(q).astype(np.int64)
    f = q - low
    rho = np.zeros((n, n, n), dtype=float)
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                w = ((f[:, 0] if ix else 1.0 - f[:, 0])
                     * (f[:, 1] if iy else 1.0 - f[:, 1])
                     * (f[:, 2] if iz else 1.0 - f[:, 2]))
                w = w * mass
                idx = (low + np.array([ix, iy, iz])) % n
                np.add.at(rho, (idx[:, 0], idx[:, 1], idx[:, 2]), w)
    rho *= n ** 3 / np.sum(mass)
    return rho  # mean = total_mass / n^3 in code units per cell


def _sample_vector(grid: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """CIC-interpolate a (n,n,n,3) vector field at particle positions."""
    n = grid.shape[0]
    q = (pos % 1.0) * n
    low = np.floor(q).astype(np.int64)
    f = q - low
    out = np.zeros((pos.shape[0], 3), dtype=float)
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                w = ((f[:, 0] if ix else 1.0 - f[:, 0])
                     * (f[:, 1] if iy else 1.0 - f[:, 1])
                     * (f[:, 2] if iz else 1.0 - f[:, 2]))
                idx = (low + np.array([ix, iy, iz])) % n
                out += w[:, None] * grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    return out


@dataclass
class Cosmology:
    """Flat FLRW background parameters."""
    H0_km_s_Mpc: float
    Omega_m: float
    Omega_r: float
    Omega_Lambda: float

    def e2(self, a: float) -> float:
        return self.Omega_r / a ** 4 + self.Omega_m / a ** 3 + self.Omega_Lambda

    def H(self, a: float) -> float:
        """Hubble parameter in s^-1."""
        return hubble_from_cosmology(self.H0_km_s_Mpc, self.Omega_m,
                                     self.Omega_r, self.Omega_Lambda, a)

    def dlnH_dlnA(self, a: float) -> float:
        """d ln H / d ln a."""
        e2 = self.e2(a)
        return (-4.0 * self.Omega_r / a ** 4 - 3.0 * self.Omega_m / a ** 3) / (2.0 * e2)

    def Omega_m_a(self, a: float) -> float:
        return (self.Omega_m / a ** 3) / self.e2(a)


@dataclass
class ParticleMesh3D:
    """3D PM gravity integrator in comoving coordinates.

    Attributes
    ----------
    n : int
        Mesh size (n^3 grid).
    x : (N,3) float
        Comoving positions in box fraction [0,1).
    u : (N,3) float
        Comoving velocity dx/ds where s = ln a.
    mass : (N,) float
        Particle mass. Code units (mean density = 1) by default; call
        set_physical_masses() to set M_sun per particle for a physical box.
    cosmology : Cosmology
        Background FLRW parameters.
    box_size_Mpc : float
        Comoving box length in Mpc. Default 1.0 preserves dimensionless
        code-unit behavior for existing tests.
    a : float
        Current scale factor.
    t : float
        Current cosmic proper time (seconds).
    box_fraction : bool
        If True, positions are in box fraction; proper length = a * L_box * x.
    """

    n: int
    x: np.ndarray
    u: np.ndarray
    mass: np.ndarray
    cosmology: Cosmology
    box_size_Mpc: float = 1.0
    a: float = 1.0
    t: float = 0.0
    box_fraction: bool = True
    _step_count: int = 0
    _substeps: int = 0
    _mass_is_physical: bool = False
    integrator: str = "kdk_analytic_drag"

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).copy()
        self.u = np.asarray(self.u, dtype=float).copy()
        self.mass = np.asarray(self.mass, dtype=float).copy()
        if self.x.ndim != 2 or self.x.shape[1] != 3:
            raise ValueError("x must have shape (N, 3)")
        if self.u.shape != self.x.shape:
            raise ValueError("u must have the same shape as x")
        if self.mass.shape != (self.x.shape[0],):
            raise ValueError("mass must have shape (N,)")
        if np.any(self.mass <= 0):
            raise ValueError("particle masses must be positive")
        if not (0 < self.a):
            raise ValueError("scale factor must be positive")
        if not (0 < self.box_size_Mpc):
            raise ValueError("box_size_Mpc must be positive")
        # Cosmic time at the current scale factor (integrate from a=0).
        from ..core.clocks import cosmic_time_from_scale_factor
        self.t = cosmic_time_from_scale_factor(
            self.a, self.cosmology.H0_km_s_Mpc,
            self.cosmology.Omega_m, self.cosmology.Omega_r,
            self.cosmology.Omega_Lambda,
        )

    # ---------------------------------------------------------------- fields
    def density_contrast(self) -> np.ndarray:
        """CIC density contrast delta = rho/rho_bar - 1 on the mesh.

        Dimensionless, independent of box size. Mean is zero (Jeans swindle /
        k=0 mode dropped). Uses unit-mass deposition for backward compatibility
        with the code-unit integrator.
        """
        rho = _cic_deposit_3d(self.x, self.n)
        return rho - 1.0

    def density_contrast_mass_weighted(self) -> np.ndarray:
        """Mass-weighted CIC density contrast delta = rho/rho_bar - 1.

        Deposits actual particle masses, so the result is correct when particle
        masses are unequal or in physical units.
        """
        rho = _cic_deposit_mass_3d(self.x, self.mass, self.n)
        return rho - 1.0

    def comoving_mean_density_Msun_Mpc3(self) -> float:
        """Comoving mean mass density in M_sun / Mpc^3.

        If physical masses have been set (set_physical_masses), computes from
        total mass / box volume. Otherwise uses Omega_m * rho_crit from the
        cosmology (code-unit normalization, mean density = 1).
        """
        if self._mass_is_physical:
            total_mass = float(np.sum(self.mass))  # M_sun
            volume = self.box_size_Mpc ** 3         # Mpc^3
            return total_mass / volume
        return self.cosmology.Omega_m * critical_density_Msun_Mpc3(
            self.cosmology.H0_km_s_Mpc)

    def proper_mean_density_Msun_Mpc3(self) -> float:
        """Proper (physical) mean mass density in M_sun / Mpc^3."""
        return self.comoving_mean_density_Msun_Mpc3() / self.a ** 3

    def comoving_density_field_Msun_Mpc3(self) -> np.ndarray:
        """Comoving mass density field in M_sun / Mpc^3 on the mesh.

        rho_comoving = rho_bar_comoving * (1 + delta).
        """
        rho_bar = self.comoving_mean_density_Msun_Mpc3()
        delta = self.density_contrast()
        return rho_bar * (1.0 + delta)

    def proper_density_field_Msun_Mpc3(self) -> np.ndarray:
        """Proper (physical) mass density field in M_sun / Mpc^3 on the mesh."""
        return self.comoving_density_field_Msun_Mpc3() / self.a ** 3

    def set_physical_masses(self) -> None:
        """Set particle masses from the cosmology and box size (M_sun each).

        Computes M_sun per particle = Omega_m * rho_crit * box_size_Mpc^3 / N,
        so the total mass in the box matches the cosmological mean density.
        After calling, _mass_is_physical is True and density conversions use
        the actual particle masses.
        """
        rho_crit = critical_density_Msun_Mpc3(self.cosmology.H0_km_s_Mpc)
        rho_bar = self.cosmology.Omega_m * rho_crit  # M_sun / Mpc^3 (comoving)
        total_mass = rho_bar * self.box_size_Mpc ** 3  # M_sun
        self.mass = np.full(self.x.shape[0], total_mass / self.x.shape[0])
        self._mass_is_physical = True

    # ---------------------------------------------------------------- physical positions
    def comoving_position_Mpc(self) -> np.ndarray:
        """Comoving position in Mpc: r_comoving = box_size_Mpc * x_comoving."""
        return self.box_size_Mpc * self.x

    def proper_position_Mpc(self) -> np.ndarray:
        """Proper (physical) distance in Mpc: r_proper = a * box_size_Mpc * x_comoving."""
        return self.a * self.box_size_Mpc * self.x

    def proper_position_meters(self) -> np.ndarray:
        """Proper (physical) distance in meters."""
        return self.proper_position_Mpc() * MPC_METERS

    # ---------------------------------------------------------------- physical velocity
    def peculiar_velocity_km_s(self) -> np.ndarray:
        """Peculiar velocity in km/s: v_pec = a * H(a) * dx/ds * box_size_Mpc * MPC_KM."""
        H = self.cosmology.H(self.a)  # s^-1
        return self.a * H * self.u * self.box_size_Mpc * MPC_KM

    # ---------------------------------------------------------------- existing API (backward compatible)
    def drag(self) -> float:
        """Cosmological drag coefficient 2 + d ln H / d ln a."""
        return 2.0 + self.cosmology.dlnH_dlnA(self.a)

    def comoving_acceleration(self, delta: Optional[np.ndarray] = None) -> np.ndarray:
        """Comoving gravitational acceleration per unit s^2:
        g = -(3/2) Omega_m(a) * grad_x Delta^{-1} delta.

        Returns code units (per box fraction per s^2). Independent of box_size_Mpc.
        """
        if delta is None:
            delta = self.density_contrast()
        grad = _poisson_gradient_3d(delta, 1.0)  # per box fraction
        om = self.cosmology.Omega_m_a(self.a)
        return -1.5 * om * _sample_vector(grad, self.x)

    def proper_position(self) -> np.ndarray:
        """Proper position = a * x_comoving (box fraction * a)."""
        return comoving_to_proper(self.x, self.a)

    def peculiar_velocity(self) -> np.ndarray:
        """Peculiar velocity v_pec = a * H(a) * dx/ds (code units, box fraction / s)."""
        H = self.cosmology.H(self.a)
        return self.a * H * self.u

    # ---------------------------------------------------------------- stepping
    def _kick(self, u: np.ndarray, half_ds: float, acc: np.ndarray) -> np.ndarray:
        """Kick: u += half_ds * (acc - drag * u). Explicit drag update."""
        return u + half_ds * (acc - self.drag() * u)

    def _kick_analytic_drag(self, u: np.ndarray, half_ds: float,
                            acc: np.ndarray) -> np.ndarray:
        """Kick with analytic drag: u -> u*exp(-drag*dt) + (acc/drag)*(1-exp(-drag*dt)).

        Treats the velocity-dependent drag term exactly over the half-step
        instead of an explicit Euler update, giving second-order accuracy in s
        even when drag is non-negligible.
        """
        drag = self.drag()
        dt = half_ds
        if abs(drag) < 1e-14:
            return u + dt * acc
        E = math.exp(-drag * dt)
        return u * E + (acc / drag) * (1.0 - E)

    def _drift(self, x: np.ndarray, u: np.ndarray, ds: float) -> np.ndarray:
        """Drift: x += ds * u (mod 1). Constant velocity."""
        return (x + ds * u) % 1.0

    def _drift_analytic_drag(self, x: np.ndarray, u: np.ndarray,
                             ds: float) -> tuple:
        """Drift with analytic drag.

        During the drift the velocity decays as v(s) = u*exp(-drag*sigma), so
        the displacement is x + u*(1-exp(-drag*ds))/drag. Returns
        (x_new, u_end) where u_end = u*exp(-drag*ds) is the velocity at the
        end of the drift.
        """
        drag = self.drag()
        if abs(drag) < 1e-14:
            x_new = (x + ds * u) % 1.0
            return x_new, u
        E = math.exp(-drag * ds)
        x_new = (x + u * (1.0 - E) / drag) % 1.0
        return x_new, u * E

    def step(self, ds: Optional[float] = None,
             max_d_ln_a: float = 0.01) -> float:
        """Advance by one KDK step in s = ln a.

        If ds is None, chooses ds = max_d_ln_a (capped). Returns the ds taken.

        Cosmic proper time t(a) is recomputed from the scale factor via the
        FLRW background integral at each step (exact up to the background
        integral's accuracy), rather than accumulated from ds/H, so it does not
        drift over many steps.

        Integrator schemes:
        - "kdk_drag": explicit drag update inside the kick (original scheme).
        - "kdk_analytic_drag": analytic drag treatment (default, second-order
          accurate in s even with non-negligible drag).
        """
        if ds is None:
            ds = max_d_ln_a
        if ds <= 0:
            raise ValueError("ds must be positive")
        a0 = self.a
        a1 = a0 * math.exp(ds)
        # Force at the beginning of the step.
        delta0 = self.density_contrast()
        acc0 = self.comoving_acceleration(delta0)
        # Kick 1 (half step).
        if self.integrator == "kdk_analytic_drag":
            u_half = self._kick_analytic_drag(self.u, 0.5 * ds, acc0)
        else:
            u_half = self._kick(self.u, 0.5 * ds, acc0)
        # Drift (full step).
        if self.integrator == "kdk_analytic_drag":
            x_new, u_end_drift = self._drift_analytic_drag(self.x, u_half, ds)
        else:
            x_new = self._drift(self.x, u_half, ds)
            u_end_drift = u_half
        # Force at the end.
        self.x = x_new
        self.a = a1
        delta1 = self.density_contrast()
        acc1 = self.comoving_acceleration(delta1)
        # Kick 2 (half step), applied to the velocity at end of drift.
        if self.integrator == "kdk_analytic_drag":
            u_new = self._kick_analytic_drag(u_end_drift, 0.5 * ds, acc1)
        else:
            u_new = self._kick(u_half, 0.5 * ds, acc1)
        self.u = u_new
        # Cosmic proper time from the background integral at the new scale factor.
        from ..core.clocks import cosmic_time_from_scale_factor
        self.t = cosmic_time_from_scale_factor(
            self.a, self.cosmology.H0_km_s_Mpc,
            self.cosmology.Omega_m, self.cosmology.Omega_r,
            self.cosmology.Omega_Lambda,
        )
        self._step_count += 1
        self._substeps += 1
        return ds

    def run_to_a(self, a_end: float, max_d_ln_a: float = 0.01,
                 callback=None) -> dict:
        """Evolve from current a to a_end (must be > current a)."""
        if a_end <= self.a:
            raise ValueError("a_end must be greater than current scale factor")
        s0 = math.log(self.a)
        s1 = math.log(a_end)
        ds_total = s1 - s0
        n_sub = max(1, math.ceil(ds_total / max_d_ln_a))
        ds = ds_total / n_sub
        history = []
        for i in range(n_sub):
            self.step(ds, max_d_ln_a=max_d_ln_a)
            if callback is not None:
                callback(self)
            if i % max(1, n_sub // 10) == 0 or i == n_sub - 1:
                history.append({
                    "a": float(self.a),
                    "t": float(self.t),
                    "rms_delta": float(np.std(self.density_contrast())),
                    "max_delta": float(np.max(self.density_contrast())),
                })
        return {"a": float(self.a), "t": float(self.t), "history": history}

    # ---------------------------------------------------------------- diagnostics
    def snapshot(self) -> dict:
        """Immutable snapshot of the current state with physical units."""
        delta = self.density_contrast()
        return {
            "schema": "pm3d/v1",
            "a": self.a,
            "t": self.t,
            "t_Gyr": self.t / SECONDS_PER_GYR,
            "H_s": self.cosmology.H(self.a),
            "z": 1.0 / self.a - 1.0,
            "box_size_Mpc": self.box_size_Mpc,
            "x_comoving": self.x.copy(),
            "u_dx_ds": self.u.copy(),
            "x_comoving_Mpc": self.comoving_position_Mpc().copy(),
            "x_proper": self.proper_position().copy(),
            "x_proper_Mpc": self.proper_position_Mpc().copy(),
            "x_proper_meters": self.proper_position_meters().copy(),
            "v_pec_code": self.peculiar_velocity().copy(),
            "v_pec": self.peculiar_velocity().copy(),
            "v_pec_km_s": self.peculiar_velocity_km_s().copy(),
            "density_contrast": delta.copy(),
            "density_contrast_mass_weighted": self.density_contrast_mass_weighted().copy(),
            "comoving_density_Msun_Mpc3": self.comoving_density_field_Msun_Mpc3().copy(),
            "proper_density_Msun_Mpc3": self.proper_density_field_Msun_Mpc3().copy(),
            "mass": self.mass.copy(),
            "mass_is_physical": self._mass_is_physical,
            "n_mesh": self.n,
            "n_particles": int(self.x.shape[0]),
            "comoving_mean_density_Msun_Mpc3": self.comoving_mean_density_Msun_Mpc3(),
            "proper_mean_density_Msun_Mpc3": self.proper_mean_density_Msun_Mpc3(),
            "units": {
                "x_comoving": "box fraction",
                "x_proper": "box fraction * a",
                "x_comoving_Mpc": "comoving Mpc",
                "x_proper_Mpc": "proper Mpc",
                "x_proper_meters": "proper meters",
                "v_pec_code": "a * H(a) * dx/ds (box fraction / s)",
                "v_pec": "a * H(a) * dx/ds (code length / s)",
                "v_pec_km_s": "peculiar velocity km/s",
                "density_contrast": "dimensionless",
                "comoving_density_Msun_Mpc3": "M_sun / Mpc^3 (comoving)",
                "proper_density_Msun_Mpc3": "M_sun / Mpc^3 (proper)",
                "mass": "M_sun if set_physical_masses() called, else code units",
                "rho_bar_comoving": "M_sun / Mpc^3 (comoving mean)",
                "rho_bar_proper": "M_sun / Mpc^3 (proper mean)",
            },
        }

    def checkpoint(self) -> dict:
        return {
            "schema": "pm3d-checkpoint/v1",
            "a": self.a,
            "t": self.t,
            "box_size_Mpc": self.box_size_Mpc,
            "x": self.x.copy(),
            "u": self.u.copy(),
            "mass": self.mass.copy(),
            "mass_is_physical": self._mass_is_physical,
            "step_count": self._step_count,
            "substeps": self._substeps,
        }

    def restore(self, cp: dict) -> "ParticleMesh3D":
        self.a = float(cp["a"])
        self.t = float(cp["t"])
        self.box_size_Mpc = float(cp.get("box_size_Mpc", 1.0))
        self.x = cp["x"].copy()
        self.u = cp["u"].copy()
        self.mass = cp["mass"].copy()
        self._mass_is_physical = bool(cp.get("mass_is_physical", False))
        self._step_count = int(cp.get("step_count", 0))
        self._substeps = int(cp.get("substeps", 0))
        return self
