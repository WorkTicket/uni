# Current milestone (updated 2026-11-01)

Authoritative, machine-generated numbers: `docs/IMPLEMENTATION_STATUS.json` (regenerate with `python scripts/write_status.py`).

Branch: `feature/multiverse-overhaul` (from baseline commit `021a093`).

**Passes 1–6 complete:** future cosmology/timeline (`a` up to 10), Keplerian orbits with `argument of periapsis`, frame-rate-independent absolute-time integration, follow/surface camera, Galaxy Map (`G`) / Solar System Map (`M`), multiverse manager (versioned JSON+npz checkpoints, branching, isolation, no pickle), coupled habitat (planet/chemistry/protocells/replicators with mass ledger; sterile unless explicitly seeded).

**Pass 7 complete:** simulation-connected planet appearance (oceans/atmosphere gated by `simLiqW`/`simAtm`), chemical-field overlay / microscopic view (`V`), quality modes low/medium/high/ultra, inspector sim-state rows.

**Pass 8 complete:** Research Laboratory at `/research-lab` — FastAPI APIs (`/api/lab/*`): create/step/run/pause/delete, stats, history, trajectory (plotting), lineage, extinctions, checkpoint/restore, export; HTML UI with experiment list, live status, population trajectory canvas, history, extinction events, lineage inspector. Connected to `MicrobialWorld` (life simulation). Backfill fix: `TraitValue` import added to `life/population.py`.

| Item | Result (measured) |
|---|---|
| Tests | 239 passed, 1 skipped (was 199 at overhaul start) |
| `universe benchmark hydro` | 106/106 |
| `universe benchmark gravity` | 40/40 |
| `universe benchmark nbody` | 37/37 |
| Observatory API | `/api/lab/*` research lab endpoints verified (50 observatory tests pass) |
| JS syntax | all `static/js/*.js` files pass `node --check` |

## What this is NOT (read SCIENTIFIC_LIMITS.md)

All verified solvers remain 1D, periodic, dimensionless. The 3D viewer is procedural (generated from seed), not coupled to the physics engine. The habitat/life simulation uses illustrative, placeholder parameters — it does not model real abiogenesis, and a sterile world stays sterile. No claim of observationally validated planet, chemistry, or life outcomes.

## Resume here

Next actions:
- Continue R4b/R5 per `CLAUDE_EXECUTION_PLAN.md` (comoving expansion, thermochemistry, 3D particle-mesh).
- The research lab at `/research-lab` is usable: `universe-observatory --port 8000`, then open `http://127.0.0.1:8000/research-lab`.

## Observatory

3D universe viewer at `/`; 1D laboratory viewer at `/lab`; research laboratory at `/research-lab`.
