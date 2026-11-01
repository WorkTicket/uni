# Decision record R4 (1D slice) — self-gravitating gas (2026-10-30)

## Scope
DONE: 1D periodic gas self-gravity coupled to the verified Euler solver.
NOT DONE: collisionless particles, CIC deposition, 3D, comoving/cosmological scale-factor terms, free-fall test, non-trivial hydrostatic-equilibrium test. R4 is therefore **partially complete**.

## Method
Poisson d2phi/dx2 = 4 pi G (rho - rho_mean) (Jeans swindle), spectral FFT; k=0 dropped; Nyquist of g zeroed so g is real. Antisymmetry of the discrete derivative makes sum(rho*g)=0 to roundoff, so gravity conserves total momentum exactly.
Strang coupling: kick(dt/2), hydro(dt), kick(dt/2). Kick: m += dt*rho*g; E += dt*g*(m_old+m_new)/2 (kinetic energy only). Total energy E_kin+E_int+W, W=0.5*sum((rho-rho_mean)*phi)*dx. dt <= min(hydro CFL, 0.1/sqrt(4 pi G rho_max)).

## Findings recorded (measured)
1. **Test-setup error caught by the analytic check.** First Jeans runs used uniform pressure (isobaric entropy perturbation), which has no restoring force; the no-gravity control did not oscillate (+0.99994 instead of cos(3.77) = -0.809). The Jeans/sound mode needs dp = c^2 d(rho). Fixed; the control now gives -0.8084.
2. **Reference error, not solver error.** A Gaussian force "plateau" at 1.1e-6 came from linearly interpolating a fine-grid reference. Against the exact Fourier series the spectral solver converges 6.3e-3 (16 cells) -> 1.5e-7 (32) -> 1e-14 (48) -> roundoff (64).
3. **Insensitive evaluation point.** Evaluating the stable oscillation at omega*t ~ 2 pi (flat part of cos) hid the convergence order; at omega*t = 4.67 the order is 1.97/2.00/2.00/2.00.
4. **Observed order** of the unstable Jeans amplitude error: 2.21, 2.10, 2.05 (32..256 cells).
5. **Energy conservation is splitting-level, not roundoff.** Drift <= 3.6e-6 for an eps=0.3 collapse; falls from 3.6e-6 to 1.2e-7 as the time step shrinks 4x, then reaches a floor near 1e-7..3e-7 (not investigated further). Mass and momentum close to ~1e-16.
6. **Pause semantics.** Checkpoint/replay is bitwise only on step boundaries; pausing at an arbitrary time clips a step and changes the trajectory by ~1.5e-5. The observatory must pause on step boundaries.
7. **kick_cfl is non-binding** at ordinary densities (identical drift for 0.2/0.1/0.05); a dedicated test with G=200 verifies it binds when it should.
8. **Sabotage check:** 6/6 broken variants detected (sign flip, halved G, kick without energy update, uncentered energy kick, 1e-6 net force bias, first-order splitting). The missing-energy-update bug is detected only by the energy-conservation check.
