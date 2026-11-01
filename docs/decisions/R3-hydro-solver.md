# Decision record R3 — 1D Euler solver design (2026-10-30)

## Equations and variables
U = [rho, rho*u, E], E = p/(gamma-1) + rho*u^2/2; F = [rho*u, rho*u^2+p, u(E+p)].
Dimensionless code units (no physical units attached at this layer; gamma passed explicitly, default 1.4).

## Discretisation
| Choice | Value | Why |
|---|---|---|
| Spatial | cell-centred finite volume, conservative flux difference | exact discrete conservation |
| Reconstruction | MUSCL on **primitive** variables (rho,u,p), minmod or MC limiter | guarantees positive reconstructed rho,p for TVD limiters |
| Riemann flux | HLL (Davis speeds) and HLLC (Toro 10.37) | HLL robust baseline; HLLC resolves contacts slightly better |
| Time | SSP-RK2 (Heun); `order=1` gives forward-Euler Godunov baseline | 2nd order, strong-stability preserving |
| CFL | dt = CFL*dx/max(abs(u)+c), CFL <= 0.5 enforced, default 0.4 | stability of MUSCL+RK2 |
| Ghost cells | 2 per side; periodic / outflow (zero gradient) / reflective (mirror, momentum sign flip) | |
| Positivity | (1) interface fallback to 1st order, (2) step rejection + 2 half-steps up to 6 levels, (3) `SolverFailure` | no hidden clipping |

## Findings recorded during implementation (measured, not assumed)
1. **Convergence order.** Smooth acoustic wave (eps=1e-6, 64..1024 cells): observed order 1.87 -> 1.97 (MC), 1.79 -> 1.91 (minmod); `order=1` scheme 0.94 -> 0.98. Sod (discontinuous): L1 order 0.8-0.9 for density (theory: ~2/3 contact, ~1 shock), as expected.
2. **Nonlinear floor.** At amplitude 1e-4 the error against the *linear* acoustic reference plateaus at 2.5e-4 (relative) on every grid. This is physical steepening, not a defect: the floor scales linearly with amplitude (2.4e-2 at 1e-2, 2.4e-3 at 1e-3, 2.5e-4 at 1e-4). The convergence benchmark therefore uses eps=1e-6; the floor itself is a separate benchmark.
3. **Positivity recovery is defensive.** TVD limiters on primitives cannot produce negative reconstructed states, and no benchmark (including 1e12 pressure ratio and vacuum-generating flows) triggered a rejection or fallback. These paths are verified only by fault injection (tests/unit/test_hydro_unit.py), not by a physical problem that needs them.
4. **Sabotage check.** Seven deliberately broken variants (energy flux without pressure work, silent 1st-order, unlimited slopes, missing wall sign flip, 1e-9/step mass leak, underestimated HLL speeds, outflow acting as a wall) were all detected by the suite; three are kept as permanent regression tests.

## Tolerance policy
Accuracy statements = comparison with exact solution + observed orders. Error tolerances are regression guards at ~1.5x measured error (headroom 1.2-1.5x on the tightest checks). Conservation/exactness tolerances are roundoff scale (1e-12..1e-13).

## Not done / known limits
1D only; ideal gas only; no gravity, cooling, expansion, or source terms in the verified solver (those arrive in R4/R5). The older `physical_models.gas_step` (first-order Rusanov with ad hoc cooling/star formation) is retained unchanged for the `universe-physics` demo and has NOT been upgraded to this solver.

## Addendum: CFL guard (found during R4)
Operator-split sources (gravity kick) change |u| between choosing dt and stepping. `step()` now enforces the absolute bound CFL <= 0.5 while the default dt uses `cfl` (0.4). A dt above 0.5 still raises.
