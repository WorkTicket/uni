"""1D periodic Newtonian self-gravity coupled to the verified Euler solver.

SCOPE: a 1D laboratory slice of R4 (gas only). NOT done: collisionless particles, CIC mass
deposition, 3D, comoving/cosmological scale-factor terms.

Poisson equation with the Jeans swindle (uniform background removed):
    d^2(phi)/dx^2 = 4 pi G (rho - rho_mean),   g = -d(phi)/dx
solved spectrally (FFT); the k=0 mode is dropped, the Nyquist mode of g is zeroed so g is real.
Because the discrete FFT derivative is antisymmetric in k, sum(rho*g) = 0 to roundoff, so
total momentum is conserved by the gravity kick exactly (to roundoff).

Coupling (Strang / kick-drift-kick):   kick(dt/2) -> hydro step(dt) -> kick(dt/2)
Kick:  m += dt*rho*g;   E += dt*g*(m_old + m_new)/2   (changes kinetic energy only; the
internal energy is untouched by the kick).
Conserved total energy:  E_total = sum(E)*dx + W,   W = 0.5*sum((rho-rho_mean)*phi)*dx.
Time-step: dt <= min(hydro CFL dt, kick_cfl / sqrt(4 pi G rho_max))   (free-fall/Jeans time).
Units: dimensionless code units; G is a free parameter.
"""
import numpy as np

from .hydro import Euler1D


def poisson_1d_periodic(rho, dx, G=1.0):
    """Return (phi, g) on cell centres for periodic density rho."""
    rho = np.asarray(rho, float)
    n = rho.size
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    rk = np.fft.fft(rho - rho.mean())
    phik = np.zeros_like(rk)
    nz = k != 0.0
    phik[nz] = -4.0 * np.pi * G * rk[nz] / k[nz] ** 2
    gk = -1j * k * phik
    if n % 2 == 0:
        gk[n // 2] = 0.0                       # Nyquist derivative is not representable as real
    return np.real(np.fft.ifft(phik)), np.real(np.fft.ifft(gk))


class SelfGravityEuler1D:
    def __init__(self, hydro: Euler1D, G=1.0, kick_cfl=0.1):
        if hydro.bc != ("periodic", "periodic"):
            raise ValueError("1D self-gravity slice requires periodic boundaries")
        if G < 0 or kick_cfl <= 0:
            raise ValueError("G must be >= 0 and kick_cfl > 0")
        self.hydro, self.G, self.kick_cfl = hydro, float(G), float(kick_cfl)
        self._E0 = self.total_energy()
        self.kicks = 0

    # ------------------------------------------------------------ diagnostics
    def potential_energy(self):
        rho = self.hydro.U[0]
        phi, _ = poisson_1d_periodic(rho, self.hydro.dx, self.G)
        return 0.5 * float(np.sum((rho - rho.mean()) * phi)) * self.hydro.dx

    def total_energy(self):
        return float(np.sum(self.hydro.U[2])) * self.hydro.dx + self.potential_energy()

    def energy_drift(self):
        """|E_total - E_total(0)| / (|E_kin|+|E_int|+|W| at t) — scale-aware, not roundoff-level."""
        h = self.hydro
        rho, u, p = h.primitives()
        scale = float(np.sum(0.5 * rho * u * u + p / (h.gamma - 1)) * h.dx) + abs(self.potential_energy())
        return abs(self.total_energy() - self._E0) / max(scale, 1e-300)

    @property
    def t(self):
        return self.hydro.t

    # ------------------------------------------------------------ stepping
    def stable_dt(self):
        rho_max = float(self.hydro.U[0].max())
        dt_grav = np.inf if self.G == 0 else self.kick_cfl / np.sqrt(4.0 * np.pi * self.G * rho_max)
        return min(self.hydro.stable_dt(), dt_grav)

    def _kick(self, dt):
        U = self.hydro.U
        _, g = poisson_1d_periodic(U[0], self.hydro.dx, self.G)
        m_old = U[1].copy()
        U[1] = m_old + dt * U[0] * g
        U[2] = U[2] + dt * g * 0.5 * (m_old + U[1])
        self.kicks += 1

    def step(self, dt=None):
        dt = self.stable_dt() if dt is None else float(dt)
        if dt <= 0 or dt > self.stable_dt() * (1 + 1e-12):
            raise ValueError("dt exceeds the combined hydro/gravity stable time step")
        self._kick(0.5 * dt)
        self.hydro.step(dt)
        self._kick(0.5 * dt)
        return dt

    def run_to(self, t_end, callback=None):
        while self.hydro.t < t_end * (1 - 1e-14):
            self.step(min(self.stable_dt(), t_end - self.hydro.t))
            if callback is not None:
                callback(self)
        return self

    def checkpoint(self):
        return {"hydro": self.hydro.checkpoint(), "E0": self._E0, "kicks": self.kicks}

    def restore(self, cp):
        self.hydro.restore(cp["hydro"]); self._E0, self.kicks = cp["E0"], cp["kicks"]
        return self
