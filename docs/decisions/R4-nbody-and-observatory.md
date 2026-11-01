# Decision record R4 (particles) and R16 step 1 (observatory) — 2026-10-30

## 1D N-body + gas (solvers/nbody.py)
Collisionless sheets, CIC deposit/interpolation with identical weights, one spectral Poisson solve shared with the gas, KDK leapfrog. Exact references: Zel'dovich sheet x = q + psi0 cosh(Gamma t) (exact and nonlinear before shell crossing) and the linear two-fluid system solved by eigen-decomposition.
Findings (measured):
1. **Apparent convergence stall was a time-step floor, not a bug.** Zel'dovich error order decayed 1.85 -> 1.75 -> 1.37 because the gravity limit 0.1/sqrt(4 pi G rho) fixed dt independent of n. Energy drift fell exactly 4x per halving of dt (8.3e-4, 2.1e-4, 5.4e-5, 1.4e-5), and with dt ~ dx the order is 1.96/2.01/1.97.
2. **Exact momentum exchange:** gas +7.66e-3, particles -7.66e-3, sum -4e-18 (exchange is real, not zero).
3. **Gas-only mode is bitwise identical** to the earlier SelfGravityEuler1D.
4. **Scale-aware tolerances:** a periodic-wrap check failed at 2.1e-14 vs an absolute 1e-14 because the density there is ~8; fixed as relative to 1/dx. A no-op check (`... or True`) was removed and replaced by a real shell-crossing check.
5. **Stable two-fluid gas error is not a clean power law** (tiny but non-monotone); only its magnitude is asserted, no order is claimed.
6. 6 of 6 sabotage variants detected (NGP deposit, NGP interpolation, doubled mass, flipped force, single kick, gas ignoring particles).

## Observatory (observatory/)
- FastAPI + WebSocket service around a Session that advances whole steps only (pause/save always on step boundaries). Verified: determinism, run/pause/resume bitwise vs uninterrupted, save->load->resume bitwise (also across app instances), path-traversal-safe ids, tamper/hash-mismatch rejection, non-standard JSON numbers (Infinity/NaN) rejected, WebSocket streaming, inspect endpoint, 422/409/404 error contracts. Driven end to end against a real uvicorn process.
- Viewer: single static HTML (Canvas 2D, no external libraries). Its script is executed in Node against real snapshots (catches syntax/undefined-identifier/drawing-path errors). **It has never been rendered in a browser here.**
- Honesty features: status carries `scientific_status`; a warning is raised when energy drift > 1e-3 (the gas+particle collapse reaches 3.8e-3 by t=2: beyond the verified regime).
- This is a 1D laboratory viewer, not the 3D observatory required by R16.
