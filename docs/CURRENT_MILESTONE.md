# Current milestone (updated 2026-11-01)

Authoritative, machine-generated numbers: `docs/IMPLEMENTATION_STATUS.json` (regenerate with `python scripts/write_status.py`).

Branch: `feature/fundamental-cosmology` (baseline commit `93713c4`, current HEAD `b10f15b`).

**Passes 1–6 complete:** future cosmology/timeline (scale factor up to 80, stopAtNow defaults to false), Keplerian orbits with absolute-time ephemeris, frame-rate-independent orbital integration, follow/surface/free/micro camera modes, Galaxy Map (`G`) / Solar System Map (`M`), multiverse manager (versioned JSON+npz checkpoints, branching, isolation), coupled habitat (planet/chemistry/protocells/replicators with mass ledger).

**Release A — Scientific Foundation (this sprint):**

1. **Camera-to-time coupling removed** (`index.html`, `app.js`, `ui.js`):
   - Removed `btn-now` ("Present") button — navigation never jumps to cosmic time.
   - Removed `stopAtNow` present-epoch pause — playback continues past `a=1` without stopping.
   - Removed `setS(U.Cosmo.NOW_S)` / `pause()` from navigation handlers.
   - `setS()` clamps `s` to [0,1] as a cinematic parameter only; it does not gate cosmic time.

2. **Physical vs comoving coordinate transformations** (`src/universe_engine/cosmos/coordinates.py`, `tests/cosmos/test_coordinates.py`):
   - `x_proper = a * x_comoving`, `x_comoving = x_proper / a`.
   - `v_pec = a * dx_comoving/dt`, `v_proper = H * x_proper + v_pec`.
   - `rho_proper = rho_comoving / a^3`, `p_comoving = a^2 * p_proper`.
   - Peculiar acceleration: `d(v_pec)/dt = -H * v_pec + g_comoving / a`.
   - Round-trip consistency tests (11 tests, all pass).

3. **3D expanding-universe particle-mesh gravity** (`src/universe_engine/solvers/pm3d.py`, `tests/solvers/test_pm3d.py`):
   - Comoving KDK integration in `s = ln a` with cosmological drag `2 + dlnH/dlna`.
   - Force: `-(3/2) Omega_m(a) * grad_x Delta^{-1} delta` (periodic Poisson, FFT, CIC).
   - Proper time `t(a)` from FLRW background integral (exact, no drift).
   - Outputs: proper position `a*x`, peculiar velocity `a*H*dx/ds`, physical density `rho_comoving/a^3`.
   - Checkpoint/restore bitwise at step boundaries.
   - 10 regression tests: uniform density zero-force, linear growth vs theory, proper time, proper position, peculiar velocity, force symmetry, density mean zero, checkpoint replay.

4. **Scientific Simulation mode in observatory** (`server.py`, `app.js`, `layers.js`, `shaders.js`, `index.html`, `ui.js`):
   - `/api/pm3d` (create/step/get) — serves actual PM3D numerical state, labeled `SIMULATED`.
   - Frontend mode toggle: Illustrative (procedural) vs Scientific Simulation (actual PM3D particle state).
   - `ScientificParticleLayer` + `simPointVS/simPointFS` shaders render actual particle positions.
   - Procedural cinematic view preserved as explicitly labeled Illustrative mode.

| Item | Result (measured) |
|---|---|
| Tests | 374 passed, 1 skipped |
| Coordinate transform tests | 11/11 |
| PM3D gravity tests | 10/10 |
| `universe benchmark hydro` | 106/106 |
| `universe benchmark gravity` | 40/40 |
| `universe benchmark nbody` | 37/37 |
| Observatory API | `/api/lab/*`, `/api/habitat/*`, `/api/brane/*`, `/api/geometry/*`, `/api/origin/events`, `/api/pm3d/*` verified |
| JS syntax | all `static/js/*.js` files pass `node --check` |
| Duplicate DOM IDs | none (btn-now removed) |
| Time jump on navigation | removed (no setS(NOW_S) or pause on navigation) |
| PM3D API test | create at a=0.2 (t=1.53 Gyr, z=4), step to a=1.0 (t=13.80 Gyr, z≈0), classification SIMULATED |

## What this is NOT (read SCIENTIFIC_LIMITS.md)

PM3D uses a synthetic Gaussian initial field (not a Boltzmann/CAMB transfer function); linear-growth validation is approximate (30% tolerance) due to CIC and particle noise. All verified 1D solvers remain periodic and dimensionless. No thermochemistry, radiative transfer, stars, galaxies, planets, or life are simulated. Brane cosmology and higher-dimensional models are speculative research, not established physics. The browser viewer has not been rendered in a real browser in this environment.

## Resume here

Next actions: T04 (ingest CLASS/CAMB transfer functions for physical initial conditions), R4b (validated 3D cosmological gravity with transfer-function ICs), then R5 (thermochemistry). Integration tests in a real Chromium browser and Windows executable packaging verification remain outstanding.
