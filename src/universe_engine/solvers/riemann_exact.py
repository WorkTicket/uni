"""Exact Riemann solver for the 1D Euler equations of an ideal gas.

Reference: E. F. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
3rd ed. (Springer 2009), Chapter 4 (pressure function f_K, Newton iteration for the
star-region pressure, and the sampling procedure of Section 4.5).

Used ONLY as an independent reference for verifying the finite-volume solver; it is
never part of the simulation loop. Dimensionless code units; vacuum states unsupported
(raises ValueError if the initial data would generate vacuum).
"""
import numpy as np


def _fK(p, rho, pK, gamma):
    cK = np.sqrt(gamma * pK / rho)
    if p > pK:  # shock
        A = 2.0 / ((gamma + 1.0) * rho)
        B = (gamma - 1.0) / (gamma + 1.0) * pK
        f = (p - pK) * np.sqrt(A / (p + B))
        df = np.sqrt(A / (B + p)) * (1.0 - (p - pK) / (2.0 * (B + p)))
    else:  # rarefaction
        z = (gamma - 1.0) / (2.0 * gamma)
        f = 2.0 * cK / (gamma - 1.0) * ((p / pK) ** z - 1.0)
        df = (1.0 / (rho * cK)) * (p / pK) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df


def star_state(left, right, gamma=1.4, tol=1e-14, max_iter=100):
    """Return (p_star, u_star). left/right are (rho, u, p)."""
    rL, uL, pL = left
    rR, uR, pR = right
    cL, cR = np.sqrt(gamma * pL / rL), np.sqrt(gamma * pR / rR)
    if 2.0 * (cL + cR) / (gamma - 1.0) <= uR - uL:
        raise ValueError("initial data generates vacuum; exact solver does not cover it")
    z = (gamma - 1.0) / (2.0 * gamma)
    # two-rarefaction initial guess (Toro eq. 4.46)
    p = ((cL + cR - 0.5 * (gamma - 1.0) * (uR - uL)) / (cL / pL**z + cR / pR**z)) ** (1.0 / z)
    p = max(p, 1e-12)
    for _ in range(max_iter):
        fL, dfL = _fK(p, rL, pL, gamma)
        fR, dfR = _fK(p, rR, pR, gamma)
        p_new = max(p - (fL + fR + (uR - uL)) / (dfL + dfR), 1e-12)
        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break
        p = p_new
    else:
        raise RuntimeError("exact Riemann Newton iteration did not converge")
    fL, _ = _fK(p, rL, pL, gamma)
    fR, _ = _fK(p, rR, pR, gamma)
    return p, 0.5 * (uL + uR) + 0.5 * (fR - fL)


def sample(left, right, S, gamma=1.4):
    """Exact (rho, u, p) at similarity coordinate S = (x - x0)/t. S may be an array."""
    S = np.atleast_1d(np.asarray(S, float))
    rL, uL, pL = left
    rR, uR, pR = right
    cL, cR = np.sqrt(gamma * pL / rL), np.sqrt(gamma * pR / rR)
    ps, us = star_state(left, right, gamma)
    g, z = gamma, (gamma - 1.0) / (2.0 * gamma)
    rho = np.empty_like(S); u = np.empty_like(S); p = np.empty_like(S)
    for i, s in enumerate(S):
        if s <= us:  # left of the contact
            if ps <= pL:  # left rarefaction
                cLs = cL * (ps / pL) ** z
                head, tail = uL - cL, us - cLs
                if s <= head:
                    rho[i], u[i], p[i] = rL, uL, pL
                elif s > tail:
                    rho[i], u[i], p[i] = rL * (ps / pL) ** (1.0 / g), us, ps
                else:
                    u[i] = 2.0 / (g + 1.0) * (cL + 0.5 * (g - 1.0) * uL + s)
                    c = 2.0 / (g + 1.0) * (cL + 0.5 * (g - 1.0) * (uL - s))
                    rho[i] = rL * (c / cL) ** (2.0 / (g - 1.0))
                    p[i] = pL * (c / cL) ** (2.0 * g / (g - 1.0))
            else:  # left shock
                SL = uL - cL * np.sqrt((g + 1.0) / (2.0 * g) * ps / pL + z)
                if s <= SL:
                    rho[i], u[i], p[i] = rL, uL, pL
                else:
                    q = ps / pL
                    rho[i] = rL * (q + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * q + 1.0)
                    u[i], p[i] = us, ps
        else:  # right of the contact
            if ps > pR:  # right shock
                SR = uR + cR * np.sqrt((g + 1.0) / (2.0 * g) * ps / pR + z)
                if s >= SR:
                    rho[i], u[i], p[i] = rR, uR, pR
                else:
                    q = ps / pR
                    rho[i] = rR * (q + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * q + 1.0)
                    u[i], p[i] = us, ps
            else:  # right rarefaction
                cRs = cR * (ps / pR) ** z
                head, tail = uR + cR, us + cRs
                if s >= head:
                    rho[i], u[i], p[i] = rR, uR, pR
                elif s <= tail:
                    rho[i], u[i], p[i] = rR * (ps / pR) ** (1.0 / g), us, ps
                else:
                    u[i] = 2.0 / (g + 1.0) * (-cR + 0.5 * (g - 1.0) * uR + s)
                    c = 2.0 / (g + 1.0) * (cR - 0.5 * (g - 1.0) * (uR - s))
                    rho[i] = rR * (c / cR) ** (2.0 / (g - 1.0))
                    p[i] = pR * (c / cR) ** (2.0 * g / (g - 1.0))
    return rho, u, p


def solution_on_grid(x, t, x0, left, right, gamma=1.4):
    return sample(left, right, (np.asarray(x) - x0) / t, gamma)
