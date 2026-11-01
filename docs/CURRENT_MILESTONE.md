# Current milestone (updated 2026-11-01)

Authoritative, machine-generated numbers: `docs/IMPLEMENTATION_STATUS.json` (regenerate with `python scripts/write_status.py`).

Branch: `feature/release-b-calibrated-cosmology` (base commit `dc2fd34`, verified).

**Release A foundation preserved:** comoving 3D PM gravity, coordinate transforms, Scientific Simulation mode, 374-test baseline + 1 skipped.

**Release B — Physically Calibrated 3D Cosmology (this sprint):**

1. **Physical box units** (`src/universe_engine/solvers/pm3d.py`):
   - `box_size_Mpc` parameter with explicit conversion to physical SI units.
   - Separate quantities: dimensionless box-fraction position, comoving distance (Mpc), proper distance (Mpc/meters), peculiar velocity (km/s), comoving mass density (M_sun/Mpc^3), proper mass density, density contrast (dimensionless).
   - `set_physical_masses()` defines particle masses consistently for a specified cosmological box (Omega_m * rho_crit * box_size_Mpc^3 / N in M_sun).
   - `snapshot()` reports physical units; backward-compatible code-unit fields preserved.
   - All 10 existing PM3D tests pass unchanged.

2. **Cosmological initial conditions** (`src/universe_engine/cosmos/initial_conditions.py`, new):
   - Versioned reference dataset (`data/reference/planck_lcdm_powerspectrum_v1`) with cosmology, provenance, checksum (sha256 verified at load).
   - `PowerSpectrum` loads the reference dataset or CLASS/CAMB-style two-column text; neither CLASS nor CAMB required for startup.
   - Gaussian random fields with Hermitian Fourier symmetry, real-valued density fields, Nyquist cutoff.
   - Zel'dovich/1LPT displacement + growing-mode velocity field.
   - Configurable seed, box size, particle count, mesh resolution, starting redshift.
   - Deterministic per seed; different seeds differ.

3. **Numerical integration** (`pm3d.py` + `tests/solvers/test_pm3d_integration.py`, new):
   - KDK integrator with analytic-drag treatment (default `kdk_analytic_drag`) — second-order accurate in s=ln a even with non-negligible drag; explicit `kdk_drag` retained.
   - Validated against an independently implemented RK4 reference; timestep convergence verified.
   - Error measurements: mass conservation exact (<1e-14), density mean zero (<1e-12), force symmetry (<1e-8), growth error bounded, timestep sensitivity bounded.

4. **Cosmological diagnostics** (`src/universe_engine/cosmos/diagnostics.py`, new):
   - `measure_power_spectrum`: isotropically binned 3D FFT power spectrum with physical box volume, correct k units, zero mode excluded, Nyquist limit, shot noise reported not subtracted.
   - `dimensionless_power_spectrum`: Delta^2(k) = k^3 P(k) / (2 pi^2).
   - `compare_linear_growth`, `cross_seed_statistics`, `resolution_convergence`, `numerical_error_summary`.
   - 9 tests (analytic Fourier mode peak, dimensionless conversion, PM3D snapshot spectrum, linear growth bounded error, cross-seed identical/different seeds, resolution convergence, error summary).

5. **Live observatory playback** (`observatory/server.py`, `static/js/app.js`, `static/js/ui.js`, `static/index.html`):
   - `/api/pm3d/{sid}/play`, `/pause`, `/speed`, `/status` — continuous playback with speed and a_max.
   - Bounded decimated snapshots (max 2000 pts/frame) + density statistics (mean/rms/min/max) — no enormous arrays to JSON every frame.
   - Display: scale factor, redshift, cosmic age, box size, model parameters, density stats.
   - Checkpoint/restore via `/api/pm3d/{sid}/checkpoint` and `/checkpoints/{cid}/restore`.
   - Frontend: play/pause/speed/checkpoint/restore buttons, HUD stats display.

6. **Multiverse integration** (`observatory/server.py`):
   - PM3D simulations bound to persistent multiverse universe IDs (`pm3d_universe` mapping).
   - Each universe owns cosmology, seed, box size, physical clock, model config, checkpoints.
   - Creating another universe does not overwrite the first; universe switching restores actual PM3D state.
   - Provenance labels: observational (reference cosmology) vs speculative.
   - `/api/pm3d/switch/{uid}` restores the selected universe's PM3D state.

| Item | Result (measured) |
|---|---|
| Tests | 389 passed, 1 skipped |
| PM3D regression tests | 10/10 |
| Coordinate transform tests | 11/11 |
| PM3D integration convergence tests | 6/6 |
| Diagnostics tests | 9/9 |
| `universe benchmark gravity` | 40/40 |
| `universe benchmark nbody` | 37/37 |
| PM3D performance | 145 steps/s (512pt/16^3), 74 steps/s (1728pt/32^3), 9 steps/s (4096pt/64^3/500Mpc) |
| Observatory PM3D API | create/step/get/play/pause/speed/status/checkpoint/restore/switch verified via TestClient |
| Multiverse PM3D independence | two universes independent; switch restores state |

## What this is NOT (read SCIENTIFIC_LIMITS.md)

PM3D uses Zel'dovich/1LPT initial conditions from an analytic power spectrum (not CLASS/CAMB by default); no gas, stars, galaxies, planets, or life; beyond shell crossing only conservation is checked; the browser viewer has not been rendered in a real browser.

## Resume here

Next actions: real-browser observatory validation (Task 8), final commit and deliverables (Task 10). Post-Release B: hydrodynamics and halo formation.
