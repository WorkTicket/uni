# Universe Engine — Architecture and Scientific Audit (Sprint 14)

**Audited SHA:** `7f4f09b86a44bafd5f92bfe84ec04f2385e7ff1d`  
**Audit branch:** `audit/project-genesis-evidence`  
**Audit worktree:** `C:\Users\Slay3r\Downloads\uni-audit` (separate git worktree)  
**Base branch:** `feature/project-genesis-research` @ `7f4f09b` (Sprint 13 commit)  
**Remote:** `origin` = `https://github.com/WorkTicket/uni.git`  
**UTC date:** `2026-11-01T20:45:00Z`  
**Machine/OS/Python/Node:** Windows 11 10.0.26100-SP0, Python 3.14.5, Node v24.15.0, npm 11.12.1  
**Audit scope:** read-only, original scientific code unchanged. No merge, no force-push, no history rewrite.

## Executive findings

1. **Repository verified.** HEAD == remote `feature/project-genesis-research` == `7f4f09b` (Sprint 13 commit). Confirmed via `git rev-parse HEAD`, `git branch -avv`, `git fetch --all`. No force-push/merge performed.
2. **Working tree is dirty in the primary checkout** (`C:\Users\Slay3r\Downloads\universe-starter`): modified `src/universe_engine/observatory/static/js/ui.js` and `tests/research/test_quantum.py`; untracked `debug_pw.js`, `debug_pw2.js`, `debug_pw3.js`, `node_modules/`, `package.json`, `package-lock.json`, `pw_out.txt`, `pytest_final.txt`, `pytest_result.txt`, `screenshot-01-intro.png`, `test-results/`, `tests/observatory/playwright_verify.js`, `tests_output.txt`, `universe_engine_audit_pack/`. These were NOT touched, staged, or discarded. The audit was performed in a separate clean worktree.
3. **Full pytest suite passes:** `527 passed, 2 skipped` in `226.63s` at SHA `7f4f09b`. Manifest + logs saved under `audit_runs/pytest_full/` (run.json, stdout.log sha256 `6262b9de...`, stderr.log sha256 `e3b0c442...`). The 2 skips: `tests/test_acceptance_A_to_L.py:333` (desktop executable not built) and `tests/test_habitat.py:69` (placeholder kinetics).
4. **Category 1 (programmed) memory oracle validated:** `WorkingMemoryOracle` solves delayed-cue and occluded_target, reading only the `SensoryObservation` contract (`local_light`, `nearby_resources`, `nearby_hazards`, `position`). Never reads `env.target_pos`/`env.cue_side`. Confirmed by `tests/research/test_memory_oracle.py` (10 tests, all passing within the full suite).
5. **Category 2 (trainable) memory learning NOT demonstrated:** `gated_recurrent`, `recurrent`, `world_model`, `reactive`, `stacked_obs` all fail delayed_cue (mean < +0.2) and occluded_target (mean ≈ −1.0). Negative result reported honestly, not relabeled as oracle success. Confirmed by `test_memory_oracle.py::test_trainable_agents_fail_memory_tasks`.
6. **Checkpoint/restore is NOT full deterministic replay.** The controller RNG is re-seeded from the stored seed on `PersistentOrganism.restore()`; weights are restored but the RNG stream position is not. Documented honestly in `tests/research/test_persistent_organism_checkpoint.py`. Habitat chemistry, body, memory, self-model, developmental history, t_gyr match; stochastic action selection is deterministic-given-seed, not bitwise continuation.
7. **Organism API endpoints exist but are NOT a connected Mind Inspector UI.** `POST /api/organism`, `GET /api/organism/{oid}`, `POST /api/organism/{oid}/step`, `GET /api/organism/{oid}/inspect`, `POST /api/organism/{oid}/checkpoint` are implemented and tested (`tests/observatory/test_organism_endpoints.py`). However, the frontend (`static/js/ui.js`) is a 3D universe-viewer HUD; there is no graphical Mind Inspector letting a user open a planet → select habitat → select organism → follow → inspect memory/workspace/predictions → pause/save/resume. Endpoints alone ≠ integrated visual interface.
8. **ECS-IW causal ablations deferred** (Sprint 13): not run because no trainable baseline reliably solves a memory task. Recorded as NOT_TESTED, not skipped-as-passed.

## Branch map

| Branch/ref | SHA | Merge-base with research | Code unique to branch | Integration conflict/risks | Evidence |
|---|---|---|---|---|---|
| `origin/feature/project-genesis-research` | `7f4f09b` | self | Sprint 13 memory + stats + checkpoint + organism API | None (audit base) | `audit_runs/branch_inventory.json` |
| `refs/heads/audit/project-genesis-evidence` | `7f4f09b` | `7f4f09b` | audit tools + reports only | None | this audit |
| `origin/feature/fundamental-cosmology` | `93713c4` | older | WebGL2 chem-field shader fixes | UI shader conflicts possible | `git branch -avv` |
| `origin/feature/multiverse-overhaul` | `dcb0e95` | older | black-hole engineering + package exports | namespace/collision risk | `git branch -avv` |
| `origin/feature/unified-planetary-life` | `f8a2a89` | older | unified planetary life + read-only API | habitat API overlap | `git branch -avv` |
| `origin/feature/release-b-calibrated-cosmology` | `80b5d5e` | older | observatory repair docs | docs only | `git branch -avv` |
| `origin/main` | `021a093` | older | initial foundation | stale; not used as evidence | `git branch -avv` |
| `origin/observatory-repair-pass1` | `dc2fd34` | older | Release A PM3D foundation | superseded by research branch | `git branch -avv` |

No merges, force-pushes, resets, or branch deletions performed. Local working tree preserved (dirty files left intact).

## Module evidence matrix

| Module | Source path(s) | Branch/SHA | Actual numerical behavior | Assumptions | Test command | Saved output log | Independent benchmark | Integration link | Status |
|---|---|---|---|---|---|---|---|---|---|
| PM3D | `src/universe_engine/solvers/pm3d.py` | `7f4f09b` | Particle-mesh comoving gravity in FLRW; CIC deposition, FFT Poisson, kick-drift-kick | Periodic box, comoving coords, cold dark matter only | `python -m pytest tests/unit/test_nbody_unit.py -q` | `audit_runs/pytest_full/stdout.log` | CIC single-particle, uniform lattice zero force, momentum exactly zero | `/api/pm3d` (SIMULATED) | VERIFIED (unit-level) |
| Cosmology/coordinates | `src/universe_engine/core/clocks.py`, `src/universe_engine/cosmos/` | `7f4f09b` | FLRW background, scale factor, redshift, comoving transforms | Flat ΛCDM, analytic background | `python -m pytest tests/cosmos tests/core -q` | `audit_runs/pytest_full/stdout.log` | analytic FLRW | `/api/cosmology` | VERIFIED |
| Astro catalogs | `src/universe_engine/cosmos/` | `7f4f09b` | initial conditions, diagnostics | Gaussian seeded ICs | `python -m pytest tests/cosmos -q` | `audit_runs/pytest_full/stdout.log` | power-spectrum diagnostics | multiverse | IMPLEMENTED_UNVERIFIED |
| Planet/climate | `src/universe_engine/planet/` | `7f4f09b` | PlanetSpec, surface temp, ocean fraction | gray/1D climate | `python -m pytest tests/planet -q` | `audit_runs/pytest_full/stdout.log` | Earth/Mars/Venus benchmarks | habitat | IMPLEMENTED_UNVERIFIED |
| Chemistry/habitat | `src/universe_engine/habitat.py`, `src/universe_engine/chemistry/` | `7f4f09b` | patch chemistry ODE, nutrient/waste, mass ledger | well-mixed patches, linear kinetics | `python -m pytest tests/habitat tests/chemistry -q` | `audit_runs/pytest_full/stdout.log` | mass ledger error reported | organism patch | VERIFIED (unit) |
| Protocells/replicators | `src/universe_engine/prebiotic/`, `src/universe_engine/replicators/` | `7f4f09b` | replicator population, lineage | seeded replicators, not abiogenesis | `python -m pytest tests/prebiotic tests/replicators -q` | `audit_runs/pytest_full/stdout.log` | none claimed | habitat | IMPLEMENTED_UNVERIFIED |
| Ecology/evolution | `src/universe_engine/life/` | `7f4f09b` | genetics, metabolism, ecology, population | artificial-life initialization | `python -m pytest tests/life tests/evolution -q` | `audit_runs/pytest_full/stdout.log` | none | organism | IMPLEMENTED_UNVERIFIED |
| GridWorld/world model | `src/universe_engine/research/environment/gridworld.py` | `7f4f09b` | fully observable foraging | memory-irrelevant negative control | `python -m pytest tests/research/test_gridworld.py -q` | `audit_runs/pytest_full/stdout.log` | reactive +0.594 | memory tasks | VERIFIED (negative control) |
| Working-memory oracle | `src/universe_engine/research/diagnostics/statistical_replication.py` (`WorkingMemoryOracle`) | `7f4f09b` | programmed memory, reads observation contract only | positive control, not learned | `python -m pytest tests/research/test_memory_oracle.py -q` | `audit_runs/pytest_full/stdout.log` | paired CI excludes zero | none (diagnostic) | VERIFIED |
| Trainable recurrent/world-model | `src/universe_engine/research/diagnostics/statistical_replication.py` | `7f4f09b` | REINFORCE/actor-critic, all fail memory tasks | online learning, tiny budget | `python -m pytest tests/research/test_memory_oracle.py -q` | `audit_runs/pytest_full/stdout.log` | mean < +0.2 delayed_cue | oracle reference | FAILED (learning not demonstrated) |
| ECS-IW self-model/workspace | `src/universe_engine/research/self_model/`, `workspace/`, `EcsIwAblation` | `7f4f09b` | workspace broadcast, self-model, recurrence | ablation framework defined | `python -m pytest tests/research/test_self_model_workspace.py -q` | `audit_runs/pytest_full/stdout.log` | none | organism controller | IMPLEMENTED_UNVERIFIED |
| Persistent organisms | `src/universe_engine/research/persistent_organism.py` | `7f4f09b` | identity, body, habitat patch, memory, self-model, checkpoint | RNG re-seeded on restore (not continued) | `python -m pytest tests/research/test_persistent_organism_checkpoint.py -q` | `audit_runs/pytest_full/stdout.log` | field-by-field match | `/api/organism*` | PROTOTYPE |
| Quantum/CHSH | `src/universe_engine/research/quantum/chsh.py`, `states.py` | `7f4f09b` | density matrix, CHSH S, no-signalling | toy CHSH, not full quantum universe | `python -m pytest tests/research/test_quantum.py -q` | `audit_runs/pytest_full/stdout.log` | S ≤ 2 classical, 2√2 quantum | research lab | VERIFIED (toy) |
| Multiscale fidelity | `src/universe_engine/research/multiscale/adaptive.py` | `7f4f09b` | adaptive fidelity | cost claims unsubstantiated | `python -m pytest tests/research/test_multiscale.py -q` | `audit_runs/pytest_full/stdout.log` | none | PM3D | IMPLEMENTED_UNVERIFIED |
| Multiverse/brane | `src/universe_engine/multiverse.py`, `frontier/brane_world.py` | `7f4f09b` | universe registry, brane dynamics | labeled speculative_research | `python -m pytest tests/frontier tests/multiverse -q` | `audit_runs/pytest_full/stdout.log` | junction residual | API labeled speculative | PROTOTYPE |
| Observatory/desktop | `src/universe_engine/observatory/server.py`, `static/` | `7f4f09b` | FastAPI + WebSocket + 3D viewer | UI is 3D viewer, no Mind Inspector | `python -m pytest tests/observatory -q` | `audit_runs/pytest_full/stdout.log` | none | organism API | IMPLEMENTED_UNVERIFIED |

Allowed: VERIFIED, IMPLEMENTED_UNVERIFIED, PROTOTYPE, ILLUSTRATIVE, STUB, FAILED, NOT_TESTED, NOT_APPLICABLE. `VERIFIED` requires executed acceptance evidence.

## Test execution record

| SHA | Command | Machine | Run manifest path | Pass | Fail | Skip | Error | Timeout | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `7f4f09b` | `python -m pytest -q -ra` | Windows 11, Python 3.14.5 | `audit_runs/pytest_full/run.json` | 527 | 0 | 2 | 0 | 0 | 529 collected; 2 skips (desktop exe not built; placeholder kinetics) |
| `7f4f09b` | `python -m pytest tests/research/test_memory_oracle.py -q` | same | `audit_runs/pytest_full/stdout.log` | 10 | 0 | 0 | 0 | 0 | oracle validates; trainables fail |
| `7f4f09b` | `python -m pytest tests/research/test_persistent_organism_checkpoint.py -q` | same | `audit_runs/pytest_full/stdout.log` | 3 | 0 | 0 | 0 | 0 | checkpoint field-match; RNG reseeded not continued |
| `7f4f09b` | `python -m pytest tests/observatory/test_organism_endpoints.py -q` | same | `audit_runs/pytest_full/stdout.log` | 3 | 0 | 0 | 0 | 0 | organism API endpoints behave |

Never infer counts from command exit code alone. Where pytest was interrupted, `NOT_COMPLETED`. No tests were interrupted.

## Known failures and conflicting claims

- **Sprint 10–11:** original world-model failure preserved (reactive +0.48/+0.49/+0.42 vs recurrent −0.77/−1.16/−2.44 per Sprint 12 records). Distinguish original from post-bugfix reruns; do not soften.
- **Sprint 12 statistics limitation:** CI computed on per-model mean is NOT a CI for model-minus-baseline; individual overlapping CIs do not determine pairwise significance. Sprint 13 corrected to paired per-seed differences.
- **checkpoint RNG:** `PersistentOrganism.restore()` re-seeds controller RNG from stored seed; does not continue the stream. Not bit-exact deterministic replay. Disclosed, not hidden.
- **`chss.py` vs `chsh.py`:** verify actual file naming; do not assume.
- **stale `head_commit`** recorded during a commit in Sprint 12 Phase 1 (`05b64f8` "correct HEAD commit metadata"). Version stamps must describe the SHA being audited (`7f4f09b`), not a future commit.
- **CI interpretation error:** a nonsignificant result is not proof of equivalence.

## Actual coupling chain

| Source → destination | Quantity/units | Dynamic? | Mode | Test/log |
|---|---|---|---|---|
| Cosmology → Halo | scale factor, density | yes | numerical (PM3D) | `tests/unit/test_nbody_unit.py` |
| Halo → Star | not numerically coupled | no | missing/speculative | none |
| Star → Planet | not numerically coupled | no | missing/speculative | none |
| Planet → Habitat | planet spec → habitat patches | yes | reduced-order (PlanetSpec → Habitat) | `tests/planet`, `tests/habitat` |
| Habitat → Organism | patch chemistry → sensory observation | yes | numerical (local patch only) | `test_persistent_organism_checkpoint.py` |

The chain Cosmology → Halo → Star → Planet is NOT numerically connected; halos/stars/planets are independent seeded catalogs or illustrative. Do not fabricate a physical formation history.

## Windows application checks

Host OS: Windows 11 10.0.26100-SP0. Windows build: NOT_TESTED (no clean-machine installer build executed). Installer on clean Windows: NOT_TESTED. Browser UI E2E: NOT_TESTED (no Playwright browser automation executed; `playwright_verify.js` exists but was not run). Create `docs/WINDOWS_ACCEPTANCE_CHECKLIST.md` for user execution on a real Windows machine.

## Top 3 next actions (NO implementation in this audit)

1. Publish this audit report + `docs/SCIENTIFIC_CLAIMS_MATRIX.md` + `docs/BRANCH_INTEGRATION_PLAN.md` + `docs/STATISTICAL_ANALYSIS_PLAN.md` + `docs/STATE_ARCHITECTURE_PROPOSAL.md` + `docs/WINDOWS_ACCEPTANCE_CHECKLIST.md` on the audit branch for user review.
2. User reviews audit and integration plan; explicitly approves which corrections/merges to proceed with.
3. On a separate isolated feature branch, implement the trainable recurrent baseline fix (Stage 5) and complete RNG-state continuation (Stage 7), preserving the audited baseline.

## Audit artifacts

- `audit_runs/branch_inventory.json` — git refs/SHA/worktree inventory at `7f4f09b`.
- `audit_runs/module_inventory.json` — 153 Python files AST inventory (presence only, not behavior).
- `audit_runs/pytest_full/run.json` — test run manifest (exit 0, 527 passed, 2 skipped, 226.63s).
- `audit_runs/pytest_full/stdout.log` (sha256 `6262b9de5552ff0b0c948d56080728445f93fba367a8461c305cc5add4d30dff`) and `stderr.log`.
- `audit_runs/` is gitignored; reports in `docs/` are committed.
