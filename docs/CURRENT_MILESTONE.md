# Current milestone (updated 2026-11-01)

Authoritative, machine-generated numbers: `docs/IMPLEMENTATION_STATUS.json` (regenerate with `python scripts/write_status.py`).

Branch: `feature/fundamental-cosmology` (baseline commit `93713c4`).

**Passes 1–6 complete:** future cosmology/timeline (scale factor up to 80, stopAtNow defaults to false), Keplerian orbits with absolute-time ephemeris, frame-rate-independent orbital integration, follow/surface/free/micro camera modes, Galaxy Map (`G`) / Solar System Map (`M`), multiverse manager (versioned JSON+npz checkpoints, branching, isolation), coupled habitat (planet/chemistry/protocells/replicators with mass ledger).

**Pass 2 — Timeline defects fixed (this session):**
- Removed duplicate `id="btn-now"` element in `index.html` (only the "Present" button remains).
- Removed forced `setS(U.Cosmo.NOW_S)` and `pause()` from `planDive()` — navigation never changes cosmic time.
- Replaced outdated intro copy ("universe ends at today") with future-continuing text and scientific-fidelity classification (SIMULATED / DERIVED / ILLUSTRATIVE).
- `stopAtNow` defaults to false; playback continues into the future cosmology (scale factor 80).
- `setS()`/`setT()`/timeline mapping verified against future boundaries.

**Pass 3 — Planetary navigation repaired:**
- Orbital layout now uses absolute cosmic time `this.t` instead of the decoupled `sysClk`, so orbital positions are a pure function of time (frame-rate independent, identical time → identical position).
- Follow/surface camera now transforms `this.pos` into the target's frame when switching targets, keeping a stable offset as the planet orbits.
- Free-flight mode added: WASD translation, mouse-look yaw/pitch, movement-speed control.
- Camera transitions independent of physical time.

**Pass 4 — Orbital presentation:**
- `U.orbitPos(d, t, ...)` uses absolute-time Keplerian evaluation (Newton solver for Kepler's equation).
- Orbital periods depend on physical parameters (Kepler's third law: P = sqrt(a³/M)·YEAR).
- Planet radius exaggeration never changes orbital physics (radius used only for visual size).
- Inclination and eccentricity preserved in the orientation transform.

**Pass 5 — Black-hole exploration:**
- Persistent Schwarzschild black-hole entity (`U.makeBlackHoleNode`) with mass, Schwarzschild radius, photon sphere, critical impact parameter.
- Event-horizon visualization (black sphere), optional accretion disk flag.
- Weak-lensing analytic approximation explicitly documented (`alpha = 2*r_s/b`, valid b >> r_s), not a full ray-traced GR simulation.
- Physical inspector rows (model, mass, r_s, photon sphere, critical impact, classification, provenance).
- Rare black-hole companions generated deterministically in stellar systems.

**Pass 6 — Higher-dimensional research view:**
- New `/hyperspace` frontend served by the observatory (`src/universe_engine/observatory/static/hyperspace.html`).
- Backend APIs already exposed: `/api/geometry/*`, `/api/kk/*`, `/api/brane/*`, `/api/planck/*`, `/api/origin/events`.
- View distinguishes SIMULATED / DERIVED / ILLUSTRATIVE fidelity; all higher-dimensional content labeled speculative research.

**Pass 7 — Desktop experience:**
- `UniverseEngine.exe` not present in the repository. `packaging/entry.py` → `universe_engine.launcher.main()` starts the FastAPI server and opens the system browser (packaged launcher, not a native desktop UI).
- No standalone executable window found; release cannot be described as a full native desktop app.

**Pass 9 — WebGL shader audit:**
- Shader compile/link status checked with diagnostics on every program (`gl.js` `Program` constructor).
- WebGL context loss/restoration handlers added (`main.js`).
- Framebuffer/texture cleanup via `RT.free()` and `Post.resize()`.

| Item | Result (measured) |
|---|---|
| Tests | 353 passed, 1 skipped |
| `universe benchmark hydro` | 106/106 |
| `universe benchmark gravity` | 40/40 |
| `universe benchmark nbody` | 37/37 |
| Observatory API | `/api/lab/*`, `/api/habitat/*`, `/api/brane/*`, `/api/geometry/*`, `/api/origin/events` verified |
| JS syntax | all `static/js/*.js` files pass `node --check` |
| Duplicate DOM IDs | none (btn-now removed) |
| Time jump on navigation | removed (planDive no longer sets NOW_S or pauses) |

## What this is NOT (read SCIENTIFIC_LIMITS.md)

All verified solvers remain 1D, periodic, dimensionless. The 3D viewer is procedural (generated from seed), not coupled to the physics engine. The habitat/life simulation uses illustrative, placeholder parameters. Brane cosmology and higher-dimensional models are speculative research, not established physics. Black-hole rendering uses the weak-lensing analytic approximation, not full GR ray tracing.

## Resume here

Next actions: integration tests in a real Chromium browser, Windows executable packaging verification, end-to-end navigation scenario (Galaxy → Star → Planet → Follow → Surface → Free Flight → Orbit → Star System), black-hole inspection, higher-dimensional research view, and final CURRENT_MILESTONE/IMPLEMENTATION_STATUS update.
