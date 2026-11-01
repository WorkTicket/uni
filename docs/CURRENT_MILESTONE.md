# Current milestone (updated 2026-11-01)

Authoritative, machine-generated numbers: `docs/IMPLEMENTATION_STATUS.json` (regenerate with `python scripts/write_status.py`).

Branch: `feature/release-b-calibrated-cosmology` (base commit `dc2fd34`, verified).

**Release A foundation preserved:** comoving 3D PM gravity, coordinate transforms, Scientific Simulation mode, 374-test baseline + 1 skipped.

**Release B — Physically Calibrated 3D Cosmology (this sprint):**

[Items 1-6 unchanged from Release B]

7. **Observatory playback & navigation repair (PASS 1-2):**
   - `src/universe_engine/observatory/static/js/cosmo.js`: rewritten as an authoritative cosmic background clock owning (t, a, z=1/a-1, T=T0/a) with a validated future horizon `a_max` (default 80) that is a model-validity bound, not the end of the universe. `extendHorizon(new_a_max)` continues to a later finite horizon preserving (t,a). Redshift negative in the future (a>1); temperature never rounds to 0; `tOfA`/`aOfT` agree within the table; no silent extrapolation.
   - `src/universe_engine/observatory/static/js/app.js`: removed `TOTAL_SECONDS = 96` and the `[0,1]` normalized timeline as the physics timebase. Added a `PlaybackController` (presets Paused/Observe/Explore/Fast Forward/Custom/Cinematic Tour; rates in sim-seconds per wall-clock second) driving the authoritative clock. Default: paused at the initial epoch with a Start control. Horizon boundary pauses with "Reached current model horizon — extend to continue" (never "Universe ended"). Navigation never changes cosmic time.
   - `src/universe_engine/observatory/static/js/camera.js` (new): centralized `CameraController` with explicit reference frames — FREE_FLIGHT, ORBIT_TARGET, FOLLOW_TARGET, PLANET_SURFACE, SYSTEM_MAP, GALAXY_MAP, MICROSCOPE; each mode owns its movement, no mode-fighting. Surface mode reaches ground-level altitude (not an orbital clamp). Planet workflow: focus → orbit → follow → descend → surface → return. Double-precision positions.
   - `src/universe_engine/observatory/static/js/ui.js`: playback preset UI, rate display, horizon-notice, horizon-extension controls, adaptive temperature/redshift formatting, precise region-size label.
   - `static/index.html`: added `tl-rate`, `horizon-notice`, model-horizon controls; loaded `camera.js`.

| Item | Result (measured) |
|---|---|
| Tests | 396 passed, 1 skipped (389 baseline + 7 new cosmo-clock tests) |
| PM3D regression tests | 10/10 |
| Coordinate transform tests | 11/11 |
| PM3D integration convergence tests | 6/6 |
| Diagnostics tests | 9/9 |
| `universe benchmark gravity` | 40/40 |
| `universe benchmark nbody` | 37/37 |
| Observatory PM3D API | create/step/get/play/pause/speed/status/checkpoint/restore/switch verified via TestClient |
| Multiverse PM3D independence | two universes independent; switch restores state |
| JS parse check | all viewer scripts pass `node --check` |
| Cosmo clock tests | 7/7 (time↔scale-factor agreement, future redshift<0, nonzero T, horizon extend, boundary state) |

## What this is NOT (read SCIENTIFIC_LIMITS.md)

PM3D uses Zel'dovich/1LPT initial conditions from an analytic power spectrum (not CLASS/CAMB by default); no gas, stars, galaxies, planets, or life; beyond shell crossing only conservation is checked; the browser viewer has not been rendered in a real browser.

## Resume here

All core repairs complete and committed:
- PASS 1 (playback): c2830d4
- PASS 2 (camera + object explorer): c2830d4 + 3cf36fa
- PASS 3 (habitat inspector): 3cf36fa
- PASS 4 (Life Laboratory): existing /research-lab (research_lab.html), linked from observatory top bar
- PASS 5 (Brane Laboratory): existing /hyperspace (hyperspace.html), linked from observatory top bar
- PASS 6 (cohesive UI): 3cf36fa

Remaining: PASS 7 end-to-end verification (real-browser automation, desktop Windows app, frame-time measurement, screenshots) and final commit.
