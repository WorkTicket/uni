"""1D ideal-gas Euler solver: conservative finite volume, HLL/HLLC fluxes, MUSCL
reconstruction (minmod / MC limiters), SSP-RK2 (Heun) time stepping.

Governing equations (dimensionless code units; no physical units attached here):
    U = [rho, rho*u, E],  E = p/(gamma-1) + 0.5*rho*u^2
    F = [rho*u, rho*u^2 + p, u*(E + p)]
    dU/dt + dF/dx = 0
Conserved variables are the evolved state; primitives are derived.

Numerical order: 2nd order in smooth monotone regions (MUSCL + RK2), degrading to 1st
order at extrema/discontinuities (limiter). order=1 gives the Godunov-type baseline.
Time step: dt = CFL * dx / max(|u| + c), CFL <= 0.4 by default.

Positivity strategy (no hidden clipping of conserved variables):
  1. If a reconstructed interface state has rho <= 0 or p <= 0, that interface falls back
     to first-order (piecewise constant) states; the count is logged.
  2. If a completed step yields non-finite, rho <= 0 or p <= 0 cells, the step is REJECTED
     and retried as two half steps (up to `max_retries` levels). Retries are logged.
  3. If still invalid, SolverFailure is raised. Nothing is silently clamped.

Conservation budget: for every boundary type the update is a flux difference, so
    sum(U)*dx - sum(U0)*dx + integral(F_right - F_left) dt = 0
to roundoff. `budget_residual()` reports it (scale-aware).
"""
import numpy as np


class SolverFailure(RuntimeError):
    """Raised when positivity cannot be maintained even after step rejection."""


# ---------------------------------------------------------------- variables
def prim_to_cons(rho, u, p, gamma):
    return np.array([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u * u], dtype=float)


def cons_to_prim(U, gamma):
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u * u)
    return rho, u, p


def physical_flux(U, gamma):
    rho, u, p = cons_to_prim(U, gamma)
    return np.array([rho * u, rho * u * u + p, u * (U[2] + p)])


# ---------------------------------------------------------------- limiters
def minmod(a, b):
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def mc_limiter(a, b):
    c = 0.5 * (a + b)
    m = np.minimum(np.minimum(np.abs(c), 2.0 * np.abs(a)), 2.0 * np.abs(b))
    return np.where(a * b > 0.0, np.sign(c) * m, 0.0)


LIMITERS = {"minmod": minmod, "mc": mc_limiter}


# ---------------------------------------------------------------- Riemann fluxes
def hll_flux(WL, WR, gamma):
    rL, uL, pL = WL
    rR, uR, pR = WR
    cL, cR = np.sqrt(gamma * pL / rL), np.sqrt(gamma * pR / rR)
    SL = np.minimum(uL - cL, uR - cR)  # Davis wave-speed estimates
    SR = np.maximum(uL + cL, uR + cR)
    UL = prim_to_cons(rL, uL, pL, gamma)
    UR = prim_to_cons(rR, uR, pR, gamma)
    FL, FR = physical_flux(UL, gamma), physical_flux(UR, gamma)
    Fhll = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
    return np.where(SL >= 0.0, FL, np.where(SR <= 0.0, FR, Fhll))


def hllc_flux(WL, WR, gamma):
    rL, uL, pL = WL
    rR, uR, pR = WR
    cL, cR = np.sqrt(gamma * pL / rL), np.sqrt(gamma * pR / rR)
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))
    UL = prim_to_cons(rL, uL, pL, gamma)
    UR = prim_to_cons(rR, uR, pR, gamma)
    FL, FR = physical_flux(UL, gamma), physical_flux(UR, gamma)

    def star(r, u, p, S, U):
        f = r * (S - u) / (S - Ss)
        return np.array([f, f * Ss, f * (U[2] / r + (Ss - u) * (Ss + p / (r * (S - u))))])

    UsL, UsR = star(rL, uL, pL, SL, UL), star(rR, uR, pR, SR, UR)
    FsL, FsR = FL + SL * (UsL - UL), FR + SR * (UsR - UR)
    return np.where(SL >= 0.0, FL, np.where(Ss >= 0.0, FsL, np.where(SR > 0.0, FsR, FR)))


FLUXES = {"hll": hll_flux, "hllc": hllc_flux}
NG = 2  # ghost cells per side (needed by 2nd-order reconstruction)


class Euler1D:
    def __init__(self, rho, u, p, x0=0.0, x1=1.0, gamma=1.4, bc=("periodic", "periodic"),
                 flux="hll", limiter="mc", order=2, cfl=0.4, max_retries=6):
        rho, u, p = (np.asarray(a, float).copy() for a in (rho, u, p))
        if not (rho.shape == u.shape == p.shape and rho.ndim == 1):
            raise ValueError("rho, u, p must be 1D arrays of equal length")
        if np.any(rho <= 0) or np.any(p <= 0) or not np.all(np.isfinite([rho, u, p])):
            raise ValueError("initial state must be finite with rho > 0 and p > 0")
        if flux not in FLUXES or limiter not in LIMITERS or order not in (1, 2):
            raise ValueError("unknown flux, limiter or order")
        for b in bc:
            if b not in ("periodic", "outflow", "reflective"):
                raise ValueError(f"unknown boundary condition {b!r}")
        if (bc[0] == "periodic") != (bc[1] == "periodic"):
            raise ValueError("periodic boundaries must be set on both sides")
        if not 0 < cfl <= 0.5:
            raise ValueError("cfl must be in (0, 0.5]")
        self.n, self.gamma, self.bc = rho.size, float(gamma), tuple(bc)
        self.x0, self.x1 = float(x0), float(x1)
        self.dx = (self.x1 - self.x0) / self.n
        self.x = self.x0 + (np.arange(self.n) + 0.5) * self.dx
        self.U = prim_to_cons(rho, u, p, self.gamma)
        self.flux_name, self.limiter_name, self.order, self.cfl = flux, limiter, order, cfl
        self._flux, self._lim, self.max_retries = FLUXES[flux], LIMITERS[limiter], max_retries
        self.t, self.steps, self.rejected_steps, self.fallback_interfaces = 0.0, 0, 0, 0
        self._U0_total = self.totals()
        self._U0_abs = np.sum(np.abs(self.U), axis=1) * self.dx
        self._boundary_integral = np.zeros(3)  # integral of (F_right - F_left) dt

    # ------------------------------------------------------------ accessors
    def primitives(self):
        return cons_to_prim(self.U, self.gamma)

    def totals(self):
        """Domain-integrated mass, momentum, energy."""
        return np.sum(self.U, axis=1) * self.dx

    def stable_dt(self):
        rho, u, p = self.primitives()
        return self.cfl * self.dx / float(np.max(np.abs(u) + np.sqrt(self.gamma * p / rho)))

    # ------------------------------------------------------------ boundaries
    def _pad(self, W):
        n, ng = self.n, NG
        out = np.empty((3, n + 2 * ng))
        out[:, ng:ng + n] = W
        for side, kind in enumerate(self.bc):
            if kind == "periodic":
                out[:, :ng], out[:, ng + n:] = W[:, n - ng:], W[:, :ng]
                break
            if side == 0:
                src = W[:, :ng][:, ::-1] if kind == "reflective" else np.repeat(W[:, :1], ng, axis=1)
                out[:, :ng] = src
                if kind == "reflective":
                    out[1, :ng] *= -1.0
            else:
                src = W[:, n - ng:][:, ::-1] if kind == "reflective" else np.repeat(W[:, -1:], ng, axis=1)
                out[:, ng + n:] = src
                if kind == "reflective":
                    out[1, ng + n:] *= -1.0
        return out

    # ------------------------------------------------------------ spatial operator
    def _rhs(self, U):
        """Return L(U) = -dF/dx, the two boundary fluxes, and fallback count."""
        gamma, n = self.gamma, self.n
        if np.any(U[0] <= 0.0):
            raise FloatingPointError("non-positive density")
        W = np.array(cons_to_prim(U, gamma))
        Wp = self._pad(W)
        k = n + 1  # number of interfaces
        if self.order == 1:
            WL, WR = Wp[:, 1:k + 1], Wp[:, 2:k + 2]
            fallbacks = 0
        else:
            d = Wp[:, 1:] - Wp[:, :-1]
            s = self._lim(d[:, :-1], d[:, 1:])           # slopes for padded cells 1..n+2
            WL = Wp[:, 1:k + 1] + 0.5 * s[:, :k]
            WR = Wp[:, 2:k + 2] - 0.5 * s[:, 1:k + 1]
            bad = ((WL[0] <= 0) | (WL[2] <= 0) | (WR[0] <= 0) | (WR[2] <= 0))
            fallbacks = int(bad.sum())
            if fallbacks:
                WL = np.where(bad, Wp[:, 1:k + 1], WL)
                WR = np.where(bad, Wp[:, 2:k + 2], WR)
        F = self._flux(WL, WR, gamma)
        return -(F[:, 1:] - F[:, :-1]) / self.dx, F[:, 0].copy(), F[:, -1].copy(), fallbacks

    def _valid(self, U):
        if not np.all(np.isfinite(U)) or np.any(U[0] <= 0.0):
            return False
        return bool(np.all((self.gamma - 1.0) * (U[2] - 0.5 * U[1] ** 2 / U[0]) > 0.0))

    def _attempt(self, U, dt):
        L1, a0, b0, f1 = self._rhs(U)
        U1 = U + dt * L1
        if not self._valid(U1):
            return None
        if self.order == 1:
            return U1, dt * (b0 - a0), f1
        L2, a1, b1, f2 = self._rhs(U1)
        U2 = 0.5 * U + 0.5 * (U1 + dt * L2)          # SSP-RK2 (Heun)
        if not self._valid(U2):
            return None
        return U2, 0.5 * dt * ((b0 - a0) + (b1 - a1)), f1 + f2

    def _advance(self, U, dt, depth=0):
        try:
            res = self._attempt(U, dt)
        except FloatingPointError:
            res = None
        if res is not None:
            self._boundary_integral += res[1]
            self.fallback_interfaces += res[2]
            return res[0]
        if depth >= self.max_retries:
            raise SolverFailure(f"positivity could not be maintained at t={self.t:.6g} "
                                f"even with dt reduced by 2^{depth}")
        self.rejected_steps += 1
        half = self._advance(U, 0.5 * dt, depth + 1)
        return self._advance(half, 0.5 * dt, depth + 1)

    # ------------------------------------------------------------ public control API
    def step(self, dt=None):
        dt = self.stable_dt() if dt is None else float(dt)
        # Default dt uses `cfl` (<=0.4 by default); the hard guard is the absolute bound 0.5, which
        # leaves headroom for operator-split sources (e.g. a gravity kick) that change |u| between
        # choosing dt and taking the step. A real violation still raises.
        if dt <= 0 or dt > self.stable_dt() * (0.5 / self.cfl) * (1 + 1e-12):
            raise ValueError("CFL violation: dt exceeds the stable time step")
        self.U = self._advance(self.U, dt)
        self.t += dt
        self.steps += 1
        return dt

    def run_to(self, t_end, callback=None, max_steps=10_000_000):
        """Advance to t_end (last step clipped). callback(self) is called after each step."""
        while self.t < t_end * (1 - 1e-14):
            dt = min(self.stable_dt(), t_end - self.t)
            self.step(dt)
            if callback is not None:
                callback(self)
            if self.steps > max_steps:
                raise SolverFailure("max_steps exceeded")
        return self

    # ------------------------------------------------------------ diagnostics
    def budget_residual(self):
        """Relative closure error of mass, momentum, energy (should be ~1e-15)."""
        err = np.abs(self.totals() - self._U0_total + self._boundary_integral)
        scale = np.maximum.reduce([self._U0_abs, np.sum(np.abs(self.U), axis=1) * self.dx,
                                   np.abs(self._boundary_integral)])
        return err / np.maximum(scale, np.finfo(float).tiny)

    def diagnostics(self):
        rho, u, p = self.primitives()
        return {"t": self.t, "steps": self.steps, "rejected_steps": self.rejected_steps,
                "fallback_interfaces": self.fallback_interfaces,
                "min_rho": float(rho.min()), "min_p": float(p.min()),
                "budget_residual_rel": [float(v) for v in self.budget_residual()],
                "totals": [float(v) for v in self.totals()]}

    def checkpoint(self):
        """Complete restartable state (bitwise): arrays are copied."""
        return {"U": self.U.copy(), "t": self.t, "steps": self.steps, "rejected_steps": self.rejected_steps,
                "fallback_interfaces": self.fallback_interfaces,
                "boundary_integral": self._boundary_integral.copy(),
                "U0_total": self._U0_total.copy(), "U0_abs": self._U0_abs.copy()}

    def restore(self, cp):
        self.U, self.t, self.steps = cp["U"].copy(), cp["t"], cp["steps"]
        self.rejected_steps, self.fallback_interfaces = cp["rejected_steps"], cp["fallback_interfaces"]
        self._boundary_integral, self._U0_total, self._U0_abs = (cp["boundary_integral"].copy(),
                                                               cp["U0_total"].copy(), cp["U0_abs"].copy())
        return self

    def snapshot(self):
        """Immutable-by-copy snapshot for observers; dimensionless code units."""
        rho, u, p = self.primitives()
        return {"schema": "euler1d/v1", "t": self.t, "x": self.x.copy(), "rho": rho.copy(),
                "u": u.copy(), "p": p.copy(), "units": "dimensionless code units",
                "gamma": self.gamma, "bc": list(self.bc)}
