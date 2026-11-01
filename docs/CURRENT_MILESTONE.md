# Current milestone (updated 2026-10-30)

Authoritative, machine-generated numbers: `docs/IMPLEMENTATION_STATUS.json` (regenerate with `python scripts/write_status.py`).

**R3 hydrodynamics: complete (T1).  R4 gravity + 1D collisionless particles: complete in 1D (T1).  R16 step 1: 1D observatory working.**

| Item | Result (measured) |
|---|---|
| Tests | 146 pass (24 original unchanged) |
| `universe benchmark hydro` | 106/106 |
| `universe benchmark gravity` | 40/40 |
| `universe benchmark nbody` | 37/37 |
| Sabotage detection | 19 of 19 deliberately broken variants caught |
| Observatory | API + WebSocket + save/load verified; real uvicorn process driven end to end |

## What this is NOT (read SCIENTIFIC_LIMITS.md)
Everything verified is 1D, periodic, dimensionless. There is no cosmological expansion, thermochemistry, radiative transfer, stars, galaxies, planets or life simulated, and abiogenesis is unsolved. The viewer script was executed in Node against real data but never rendered in a browser. Beyond shell crossing only conservation (not accuracy) is checked; the observatory warns when energy drift exceeds 1e-3.

## Resume here
`docs/IMPLEMENTATION_STATUS.json` -> `next_action`: R4b comoving expansion (EdS, exact growth factor D(a)=a), then R5 thermochemistry (sourced, checked rate tables), then 3D particle-mesh so the observatory can show 3D data.

## Observatory 3D viewer
The observatory now serves a WebGL2 3D universe viewer at `/` (big bang to planetary surface, seed-generated, view-driven detail). The 1D laboratory viewer moved to `/lab`.
