"""1D collisionless particles (sheets) + optional gas, sharing ONE periodic Poisson solve.

SCOPE: 1D, periodic, Newtonian, non-expanding. NOT done: 3D, comoving/cosmological terms.

Particles: positions x, velocities v, masses m (mass per unit area of a sheet). Cloud-in-cell
(CIC) deposition and CIC force interpolation use IDENTICAL weights, so total momentum is
conserved by gravity exactly (to roundoff): the particles' momentum gain equals the gas's loss.
Total density rho = rho_gas + rho_particles(CIC); phi, g from poisson_1d_periodic (spectral,
Jeans swindle). Time integration: kick(dt/2) - drift/hydro(dt) - kick(dt/2) (KDK, 2nd order).
Total energy: E_gas + 0.5*sum(m v^2) + W, W = 0.5*sum((rho-mean)*phi)*dx.
Step limits: hydro CFL, particle drift 0.4*dx/max|v|, gravity kick_cfl/sqrt(4 pi G rho_max).
"""
import numpy as np

from .gravity import poisson_1d_periodic
from .hydro import Euler1D


def cic_weights(x, x0, dx, n):
    s = (np.asarray(x) - x0) / dx - 0.5            # position in units of cell spacing from first centre
    i0 = np.floor(s).astype(np.int64)
    w1 = s - i0
    return i0 % n, (i0 + 1) % n, 1.0 - w1, w1


def cic_deposit(x, m, x0, dx, n):
    """Mass density (mass per unit length) on cell centres."""
    i0, i1, w0, w1 = cic_weights(x, x0, dx, n)
    m = np.broadcast_to(np.asarray(m, float), np.shape(x))
    return (np.bincount(i0, m * w0, n) + np.bincount(i1, m * w1, n)) / dx


def cic_interpolate(field, x, x0, dx):
    n = field.size
    i0, i1, w0, w1 = cic_weights(x, x0, dx, n)
    return field[i0] * w0 + field[i1] * w1


class GravitySystem1D:
    def __init__(self, hydro=None, x=None, v=None, mass=None, box=(0.0, 1.0), n=None, G=1.0,
                 kick_cfl=0.1, drift_cfl=0.4):
        if hydro is None and x is None:
            raise ValueError("need gas and/or particles")
        if hydro is not None:
            if hydro.bc != ("periodic", "periodic"):
                raise ValueError("requires periodic boundaries")
            self.x0, self.x1, self.n, self.dx = hydro.x0, hydro.x1, hydro.n, hydro.dx
        else:
            if n is None or n < 4:
                raise ValueError("grid size n required for particle-only systems")
            self.x0, self.x1, self.n = float(box[0]), float(box[1]), int(n)
            self.dx = (self.x1 - self.x0) / self.n
        if G < 0 or kick_cfl <= 0 or drift_cfl <= 0:
            raise ValueError("invalid G or CFL parameters")
        self.hydro, self.G, self.kick_cfl, self.drift_cfl = hydro, float(G), kick_cfl, drift_cfl
        self.L = self.x1 - self.x0
        if x is not None:
            self.x = np.asarray(x, float).copy()
            self.v = np.zeros_like(self.x) if v is None else np.asarray(v, float).copy()
            self.m = np.broadcast_to(np.asarray(mass, float), self.x.shape).copy()
            if self.x.shape != self.v.shape or np.any(self.m <= 0):
                raise ValueError("particle arrays inconsistent or non-positive mass")
            self.x = self.x0 + (self.x - self.x0) % self.L
        else:
            self.x = self.v = self.m = None
        self._t = 0.0
        self.steps = 0
        self._E0 = self.total_energy()

    # ------------------------------------------------------------ fields
    @property
    def t(self):
        return self.hydro.t if self.hydro is not None else self._t

    def density(self):
        rho = np.zeros(self.n) if self.hydro is None else self.hydro.U[0].copy()
        if self.x is not None:
            rho = rho + cic_deposit(self.x, self.m, self.x0, self.dx, self.n)
        return rho

    def fields(self):
        return poisson_1d_periodic(self.density(), self.dx, self.G)

    # ------------------------------------------------------------ diagnostics
    def potential_energy(self):
        rho = self.density()
        phi, _ = poisson_1d_periodic(rho, self.dx, self.G)
        return 0.5 * float(np.sum((rho - rho.mean()) * phi)) * self.dx

    def kinetic_particles(self):
        return 0.0 if self.x is None else 0.5 * float(np.sum(self.m * self.v ** 2))

    def total_energy(self):
        gas = 0.0 if self.hydro is None else float(np.sum(self.hydro.U[2])) * self.dx
        return gas + self.kinetic_particles() + self.potential_energy()

    def momentum(self):
        """(gas, particles) total momentum."""
        pg = 0.0 if self.hydro is None else float(np.sum(self.hydro.U[1])) * self.dx
        pp = 0.0 if self.x is None else float(np.sum(self.m * self.v))
        return pg, pp

    def energy_drift(self):
        e_int = kin = 0.0
        if self.hydro is not None:
            rho, u, p = self.hydro.primitives()
            kin += float(np.sum(0.5 * rho * u * u)) * self.dx
            e_int = float(np.sum(p / (self.hydro.gamma - 1))) * self.dx
        scale = kin + e_int + self.kinetic_particles() + abs(self.potential_energy())
        return abs(self.total_energy() - self._E0) / max(scale, 1e-300)

    # ------------------------------------------------------------ stepping
    def stable_dt(self):
        dts = []
        if self.hydro is not None:
            dts.append(self.hydro.stable_dt())
        if self.x is not None:
            vmax = float(np.max(np.abs(self.v)))
            dts.append(np.inf if vmax == 0 else self.drift_cfl * self.dx / vmax)
        if self.G > 0:
            dts.append(self.kick_cfl / np.sqrt(4.0 * np.pi * self.G * float(self.density().max())))
        return min(dts)

    def _kick(self, dt):
        _, g = self.fields()
        if self.hydro is not None:
            U = self.hydro.U
            m_old = U[1].copy()
            U[1] = m_old + dt * U[0] * g
            U[2] = U[2] + dt * g * 0.5 * (m_old + U[1])
        if self.x is not None:
            self.v = self.v + dt * cic_interpolate(g, self.x, self.x0, self.dx)

    def step(self, dt=None):
        dt = self.stable_dt() if dt is None else float(dt)
        if dt <= 0 or dt > self.stable_dt() * (0.5 / 0.4) * (1 + 1e-12):
            raise ValueError("dt exceeds the combined stable time step")
        self._kick(0.5 * dt)
        if self.x is not None:
            self.x = self.x0 + (self.x + self.v * dt - self.x0) % self.L
        if self.hydro is not None:
            self.hydro.step(dt)
        else:
            self._t += dt
        self._kick(0.5 * dt)
        self.steps += 1
        return dt

    def run_to(self, t_end, callback=None):
        while self.t < t_end * (1 - 1e-14):
            self.step(min(self.stable_dt(), t_end - self.t))
            if callback is not None:
                callback(self)
        return self

    def checkpoint(self):
        return {"hydro": None if self.hydro is None else self.hydro.checkpoint(), "t": self._t, "steps": self.steps,
                "x": None if self.x is None else self.x.copy(), "v": None if self.v is None else self.v.copy(),
                "E0": self._E0}

    def restore(self, cp):
        if self.hydro is not None:
            self.hydro.restore(cp["hydro"])
        self._t, self.steps, self._E0 = cp["t"], cp["steps"], cp["E0"]
        if self.x is not None:
            self.x, self.v = cp["x"].copy(), cp["v"].copy()
        return self

    def snapshot(self):
        """Observer snapshot (copies). Schema nbody1d/v1, dimensionless code units."""
        snap = {"schema": "nbody1d/v1", "t": self.t, "units": "dimensionless code units", "G": self.G,
                "box": [self.x0, self.x1], "n": self.n, "density_total": self.density(),
                "g": self.fields()[1]}
        if self.x is not None:
            snap.update(particle_x=self.x.copy(), particle_v=self.v.copy(), particle_m=self.m.copy())
        if self.hydro is not None:
            snap["gas"] = self.hydro.snapshot()
        return snap
