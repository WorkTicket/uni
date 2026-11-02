# Universe Engine — Stage 0 Scientific Audit Report

**Repository:** https://github.com/WorkTicket/uni  
**Audited branch:** `audit/project-genesis-evidence` (worktree `C:\Users\Slay3r\Downloads\uni-audit`)  
**Audited commit:** `ef10c4c87e6ae916e08d459b2c0f68b868c90d75` (based on `feature/project-genesis-research` head `7f4f09b`, Sprint 13)  
**Audit date:** 2026-11-01  
**Machine:** Windows 11 10.0.26100-SP0, Python 3.14.5 (tags/v3.14.5, MSC v.1944 64 bit AMD64)  
**Status:** COMPLETE — documents committed, no source changes, test baseline recorded.

This audit is read-only. No scientific source, test, or historical status file was modified. Per the master plan, this is Stage 0: establish exactly what exists before changing it.

## 1. Branch inventory (executed, command output captured)

`git status --porcelain=v1`, `git remote -v`, `git fetch --all --prune`, `git branch -avv`, `git log --all --decorate --oneline -n 40`, `git rev-parse HEAD`, `python --version` all executed. Output captured in `audit_runs/branch_inventory.json` (commit `ef10c4c`).

Local branches (9), all matched to remote-tracking refs:
- `audit/project-genesis-evidence` = `ef10c4c` (this audit branch)
- `feature/project-genesis-research` = `7f4f09b` Sprint 13 (HEAD of research program)
- `fix/trainable-memory-baseline` = `1f49241` (current working-tree branch, dirty)
- `main` = `021a093` Initial commit
- `observatory-repair-pass1` = `dc2fd34`
- `feature/fundamental-cosmology` = `93713c4`
- `feature/multiverse-overhaul` = `dcb0e95`
- `feature/release-b-calibrated-cosmology` = `80b5d5e`
- `feature/unified-planetary-life` = `f8a2a89`

Note: `main` is at `021a093` ("Initial commit") and is NOT authoritative for the research program. `feature/project-genesis-research` at `7f4f09b` is the latest research state and contains Sprint 11 (`17839a1`) and Sprint 12 (`cb82d86` → `05b64f8`) as ancestors.

**Backup tags created (non-moving, no branch modification):**
`audit-backup-main-021a093`, `audit-backup-feature-project-genesis-research-7f4f09b`, `audit-backup-fix-trainable-memory-baseline-1f49241`, `audit-backup-observatory-repair-pass1-dc2fd34`, `audit-backup-feature-fundamental-cosmology-93713c4`, `audit-backup-feature-multiverse-overhaul-dcb0e95`, `audit-backup-feature-release-b-calibrated-cosmology-80b5d5e`, `audit-backup-feature-unified-planetary-life-f8a2a89`, `audit-backup-audit-project-genesis-evidence-ef10c4c`.

No force-push, no rebase, no history rewriting.

Working tree of the primary worktree (`fix/trainable-memory-baseline`, `1f49241`) is dirty with two pre-existing, unrelated modifications:
- `tests/research/test_quantum.py`: reduced CHSH n_shots 50000→5000 / 30000→5000 (pre-existing, not part of audit).
- `src/universe_engine/observatory/static/js/ui.js`: begin-button guards generation (pre-existing).
These were left untouched. Audit work was done in the clean separate worktree.

## 2. Test baseline (executed, output captured)

Command: `python -m pytest -q -ra`  
Timeout: 300 s  
Environment: Windows 11, Python 3.14.5, pytest 9.2.0, numpy 2.5.4  
Git SHA: `ef10c4c87e6ae916e08d459b2c0f68b868c90d75`  
Output: `audit_runs/pytest_full/{run.json,stdout.log,stderr.log}` (run.json sha256 `acb99d61...`)

Result: **527 passed, 2 skipped, 0 failed, 0 hangs**, exit code 0, 279.48s.

Skipped:
1. `tests/test_acceptance_A_to_L.py:333` — "Desktop executable not built in this environment" (NOT TESTED on Windows).
2. `tests/test_habitat.py:69` — "no division within step budget (placeholder kinetics)" (placeholder).

No test hangs. Full suite collected 529 tests (527 run + 2 skip). `python -m pytest --collect-only -q` collected 529 in 0.69s.

Note: `docs/IMPLEMENTATION_STATUS.json` claims "full suite hangs on import in this environment" — that is NOT supported by this run; the suite passes cleanly. This status field is stale/hand-edited and should be regenerated from run output.

Test command recorded: `python -m pytest -q` (also `python -m pytest -q -ra` used for the baseline with reasons).

## 3. Module inventory (executed)

`python tools/audit/scripts/module_inventory.py --root . --out audit_runs/module_inventory.json` — 153 Python files inventoried. Classifications below are derived from reading source and data flow, not file existence alone.

## 4. Per-module audit status

Each status cites the test, command, or result that supports it. Statuses: VERIFIED / IMPLEMENTED_UNVERIFIED / PROTOTYPE / ILLUSTRATIVE / STUB / FAILED / NOT_TESTED / NOT_APPLICABLE.

### Core engine (`src/universe_engine/`)

| Module | Files | What it really calculates | Numerical or illustrative | Status | Evidence |
|---|---|---|---|---|---|
| CLI | `cli.py`, `doctor.py`, `launcher.py` | Commands: validate, run, doctor, benchmarks | Numerical (doctor) / driver | IMPLEMENTED_UNVERIFIED | `universe doctor` read-only; not executed in this audit |
| World state | `world_state.py` | Versioned world-state spine: entities, IDs, units, provenance, clocks, checkpoints (PASS 4) | Numerical state schema | IMPLEMENTED_UNVERIFIED | `tests/unit/test_world_state.py` (in suite, passed) |
| Physical models | `physical_models.py`, `physics.py` | 1D finite-volume Euler gas, analytic stellar tracks, dust coagulation, mass-action chemistry | Numerical demonstrators | PROTOTYPE | `tests/unit/` physics tests passed; README: "not physically coupled to the 3D cosmological simulation" |
| Habitat | `habitat.py` | Coupled planet climate → chemistry → protocells → replicators; mass ledger; operator split | Numerical | VERIFIED | `tests/test_habitat.py` passed (1 skip placeholder); ledger error tracked |
| Planet | `planet/model.py` | 1D latitudinal energy-balance climate, ice-albedo, carbonate-silicate, water phase, pH | Numerical | VERIFIED | 17 tests (per status_static.json); part of suite |
| Chemistry | `chemistry/network.py` | Element/charge-balanced mass-action network, Radau ODE + Gillespie SSA | Numerical | VERIFIED | 10 tests vs analytic solutions; rate parameters ILLUSTRATIVE |
| Prebiotic | `prebiotic/compartment.py` | Polymer growth/hydrolysis, RAF sets, osmotic vesicle protocells | Numerical (assumed parameters) | PROTOTYPE | 14 tests; kinetics/vesicle parameters assumed placeholders |
| Replicators | `replicators/population.py` | Template copying, monomer accounting, Gillespie + tau-leap, lineage, duplex stability | Numerical | VERIFIED | 9 tests incl. quasispecies/error-threshold |
| Cosmology (1D) | `solvers/hydro.py`, `solvers/nbody.py`, `core/clocks.py` | 1D Euler + gravity, FLRW background, dimensionless code units | Numerical | VERIFIED | T1 benchmarks: exact Riemann, Jeans growth, Zel'dovich, momentum exchange |
| Cosmology (3D PM3D) | `solvers/pm3d.py` | 3D periodic particle-mesh gravity in comoving FLRW, KDK integration | Numerical | VERIFIED | 10 regression tests; PM3D gravity verified T1 |
| Multiverse | `multiverse.py` | Independent universe records, checkpoints, branching, ancestry | Numerical state mgmt | VERIFIED | Part of suite; API endpoints tested |
| Research (all) | `research/*` | Quantum CHSH, GridWorld agents, world-model learning, self-model, workspace, evolution, discovery, planetary agent, multiscale, diagnostics | Mixed (see matrix below) | See §5 | Research tests passed |

### Observatory viewer (`src/universe_engine/observatory/`)

| Module | Files | What it really calculates | Numerical or illustrative | Status | Evidence |
|---|---|---|---|---|---|
| Server | `server.py` | FastAPI: session/state/snapshot/step/run/pause/save/load, habitat, organism, PM3D, multiverse, geometry, brane, planck APIs | Numerical state broker | IMPLEMENTED_UNVERIFIED | Endpoints exercised by suite (fastapi TestClient) |
| Session | `session.py` | 1D laboratory session (GravitySystem1D + Euler1D), bitwise step-boundary pause/save/load | Numerical | VERIFIED | Part of suite; checkpoint/restore bitwise |
| Research lab | `research_lab.py` | Read-only experiment lifecycle for habitat/organism | Numerical | IMPLEMENTED_UNVERIFIED | API endpoints present |
| Viewer JS | `static/js/*.js`, `static/css/app.css`, `static/index.html` | Procedural WebGL2 universe generator (seed/time/camera → geometry). Scientific Simulation mode renders PM3D state; Illustrative mode is view-driven procedural detail | Illustrative (default) / SIMULATED (PM3D mode) | PROTOTYPE | Code executed in Node against real data; **NEVER rendered in a real browser** (status_static.json). Bug A/B/C known but not fixed in this audit stage. |

**Viewer caveat:** `static/index.html` creates the WebGL2 context with `alpha: false, depth: false`. The post chain composites scene+bloom with `gl.disable(gl.BLEND)`. Body/glow/orbit layers enable blending. Scissor test is used per-body in `SystemLayer.drawBody`. These are candidate causes for Bug A (black box); confirmed only by browser reproduction (Stage 1).

### Research program (`src/universe_engine/research/`)

| Program | Status | Key result | Evidence |
|---|---|---|---|
| B Quantum (states.py, chsh.py) | VERIFIED | CHSH quantum S ≈ 2.83 > classical 2; classical S ≈ 2.0; no-signalling holds | `tests/research/test_quantum.py` passed |
| D1-D4 Embodied/world-model | IMPLEMENTED_UNVERIFIED (honest null) | Recurrent world-model UNDERPERFORMS reactive on fully-observable GridWorld; null result retained | `tests/research/test_world_model.py`, `test_gridworld.py` passed |
| D5-D6 Self-model/workspace | VERIFIED | Self-model calibration + workspace broadcasting functional | `tests/research/test_self_model_workspace.py` passed |
| F Discovery | VERIFIED | Agent selects quantum model on held-out data; CHSH ≈ 2.83 without CHSH formula | `tests/research/test_discovery.py` passed |
| A9 Multiscale fidelity | VERIFIED | Exact mass conservation; cost savings vs fixed reference | `tests/research/test_multiscale.py` passed |
| C Evolution | VERIFIED | Diverse lineages, mutation, selection, extinction | `tests/research/test_agent_evolution.py` passed |
| C8/D8 Planetary ecology | VERIFIED | Agent survives in habitat patch, coupled to chemistry | `tests/research/test_planetary_agent.py` passed |
| Sprint 11 Diagnostics | VERIFIED (diagnostic) | Delayed-cue: recurrent -0.06 > reactive -0.21 (memory task); fully-observable: reactive +0.48 > recurrent -0.77 | `research/diagnostics/` scripts; null result on GridWorld |
| Sprint 12 Persistent organism | VERIFIED (Python lifecycle) | `persistent_organism.py`: persistent identity, body, memory, self-model, checkpoint/replay, actions change habitat chemistry | `tests/research/` + `docs/SPRINT_12_FINDINGS.md`; **frontend Mind Inspector NOT wired** |

## 5. Research-integrity items (Sprint 11/12)

- **H1 hypothesis (Sprint 11):** "fully observable task is root cause of recurrent agent losing" is a hypothesis, not proven. Recurrent underperformance can also mean poor training, implementation bug, or unfair baseline. **Not established.**
- **Delayed-cue result** (recurrent −0.06 vs reactive −0.21) is too thin to support a claim until multi-seed statistics (≥20 seeds, paired CIs) are done. **NOT established.**
- **Per-seed failure records:** Sprint 11 removed per-seed failure records from status files. `docs/IMPLEMENTATION_STATUS.json` and `docs/status_static.json` carry aggregate claims only; per-seed detail is absent. To restore, re-run diagnostics with per-seed CSV and regenerate status from run output.
- **`chss.py` vs `chsh.py`:** The audit pack lists `chss.py` as a known typo anomaly, but the actual repo file is `research/quantum/chsh.py` (correct spelling). No `chss.py` exists. This anomaly is noted in the pack, not present in source.
- **Stale `head_commit`:** `docs/IMPLEMENTATION_STATUS.json` records `head_commit: "05b64f8"` (Sprint 12 Phase 1) while the actual `feature/project-genesis-research` head is `7f4f09b` (Sprint 13). Stale field; should be regenerated from run output.
- **Sprint 12 organism persistence:** `PersistentOrganism.checkpoint()` / `restore()` serialize identity, body, memory, controller, self-model, developmental history, habitat, and RNG states (Python + NumPy bit_generator state). Cross-run save/reload equivalence (run A vs run B) has NOT been demonstrated in this audit — it is the Stage 2 acceptance test.

## 6. Integration chain audit: Cosmology → Halo → Star → Planet → Habitat → Organism

| Link | Classification | Basis |
|---|---|---|
| Cosmology → Halo | reduced-order | PM3D density peaks thresholded into halo *candidates* (not a bound FoF/SO halo finder). status_static.json: "Thresholded local density peaks". |
| Halo → Star | illustrative / statistical | Stars sampled statistically per galaxy (`sampleGalaxyStars`), not derived from halo gas physics. |
| Star → Planet | illustrative | Probabilistic planetary system samples with estimated equilibrium temperature; no disk accretion/migration. |
| Planet → Habitat | numerical | `Habitat(planet, network, ...)` reads `planet.environment()` (T, pH, liquid water) each step; real dynamic coupling. |
| Habitat → Organism | numerical | `PersistentOrganism` senses only its local patch and acts on patch chemistry (consumes N, produces Wst); real feedback. |

No link from Cosmology/Planet down to Organism is numerically derived end-to-end; the chain is assembled via `world_state.connect_chain()` as a registry of stable IDs, not a single integrated simulation.

## 7. Windows audit

This machine is Windows 11, but:
- The desktop executable (`UniverseEngine.exe`) has **never been built or run** in this environment (README: "the Windows build has not been run by the author"). `tests/test_acceptance_A_to_L.py:333` skipped: "Desktop executable not built in this environment."
- CI workflow exists but was **never executed** (status_static.json scientific_limitations).
- The 3D viewer has **not been rendered or clicked in a real browser**; only executed in Node against real data.

**Windows items marked NOT TESTED:** desktop build, installer, WebGL2 runtime on Windows, clean-machine acceptance, 20-step E2E demonstration on real Windows.

## 8. Known defects observed (not fixed in Stage 0)

1. **Bug A — Black box over graphics:** opaque rectangle (~630×690) covers star-system scene; starfield draws around it; planet labels project correctly. Candidate: scissor rect/offscreen target/blending/NaN at system scale. Requires browser reproduction (Stage 1).
2. **Bug B — Orbits too fast:** playback rate at cosmic speed when a star system loads; orbits complete thousands of times per second. `sysClk += dt * (YEAR/70)` in `app.js`.
3. **Bug C — Planets move with camera:** view-driven procedural regeneration; position not a pure function of (ID, time). First piece of Stage 2 state contract.
4. **Stale `head_commit`** in `docs/IMPLEMENTATION_STATUS.json` (`05b64f8` vs actual `7f4f09b`).
5. **Missing per-seed failure records** (Sprint 11 removed them).
6. **Viewer never rendered in a real browser** (no E2E visual evidence).
7. **No `/health` endpoint**; intro "Preparing…" can hang forever with no timeout/error/Retry (Bug H).
8. **Full suite "hangs on import" claim** in IMPLEMENTATION_STATUS.json is contradicted by this run (527 pass).

## 9. Recommended next actions

1. Run Stage 1 viewer emergency fixes (Bugs A, B, C) with Playwright browser reproduction, failing→passing tests, before/after screenshots.
2. Regenerate `docs/IMPLEMENTATION_STATUS.json` from this run output (`scripts/write_status.py`) to fix stale `head_commit` and test_counts.
3. Restore per-seed failure records and write `docs/STATISTICAL_ANALYSIS_PLAN.md` before any new recurrent-vs-reactive comparison (≥20 seeds, paired CIs).
4. Implement Stage 2 authoritative state (stable IDs, independent clocks, provenance, deterministic checkpoint hash).

---

## 10. Restored executive findings (from ef10c4c)

The following findings were present in the pre-`54b7fc1` version (`ef10c4c`) of this report and were removed when `54b7fc1` rewrote the document. They are restored here verbatim from `ef10c4c` (diff evidence: `git diff ef10c4c 54b7fc1 -- docs/AUDIT_REPORT.md`).

### 10.1 Executive findings #3–8

- **#3** Full pytest suite passes: `527 passed, 2 skipped` in `226.63s` at SHA `7f4f09b`. Manifest + logs saved under `audit_runs/pytest_full/` (run.json, stdout.log sha256 `6262b9de...`, stderr.log sha256 `e3b0c442...`). The 2 skips: `tests/test_acceptance_A_to_L.py:333` (desktop executable not built) and `tests/test_habitat.py:69` (placeholder kinetics).
- **#4** Category 1 (programmed) memory oracle validated: `WorkingMemoryOracle` solves delayed-cue and occluded_target, reading only the `SensoryObservation` contract (`local_light`, `nearby_resources`, `nearby_hazards`, `position`). Never reads `env.target_pos`/`env.cue_side`. Confirmed by `tests/research/test_memory_oracle.py` (10 tests, all passing).
- **#5** Category 2 (trainable) memory learning NOT demonstrated: `gated_recurrent`, `recurrent`, `world_model`, `reactive`, `stacked_obs` all fail delayed_cue (mean < +0.2) and occluded_target (mean ≈ −1.0). Negative result reported honestly, not relabeled as oracle success. Confirmed by `test_memory_oracle.py::test_trainable_agents_fail_memory_tasks`.
- **#6** Checkpoint/restore is NOT full deterministic replay. The controller RNG is re-seeded from the stored seed on `PersistentOrganism.restore()`; weights are restored but the RNG stream position is not. Documented honestly in `tests/research/test_persistent_organism_checkpoint.py`. Habitat chemistry, body, memory, self-model, developmental history, t_gyr match; stochastic action selection is deterministic-given-seed, not bitwise continuation.
- **#7** Organism API endpoints exist but are NOT a connected Mind Inspector UI. `POST /api/organism`, `GET /api/organism/{oid}`, `POST /api/organism/{oid}/step`, `GET /api/organism/{oid}/inspect`, `POST /api/organism/{oid}/checkpoint` are implemented and tested (`tests/observatory/test_organism_endpoints.py`). However, the frontend (`static/js/ui.js`) is a 3D universe-viewer HUD; there is no graphical Mind Inspector letting a user open a planet → select habitat → select organism → follow → inspect memory/workspace/predictions → pause/save/resume. Endpoints alone ≠ integrated visual interface.
- **#8** ECS-IW causal ablations deferred (Sprint 13): not run because no trainable baseline reliably solves a memory task. Recorded as NOT_TESTED, not skipped-as-passed.

### 10.2 Test execution record (restored)

| SHA | Command | Machine | Run manifest path | Pass | Fail | Skip | Error | Timeout | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `7f4f09b` | `python -m pytest -q -ra` | Windows 11, Python 3.14.5 | `audit_runs/pytest_full/run.json` | 527 | 0 | 2 | 0 | 0 | 529 collected; 2 skips |
| `7f4f09b` | `python -m pytest tests/research/test_memory_oracle.py -q` | same | `audit_runs/pytest_full/stdout.log` | 10 | 0 | 0 | 0 | 0 | oracle validates; trainables fail |
| `7f4f09b` | `python -m pytest tests/research/test_persistent_organism_checkpoint.py -q` | same | `audit_runs/pytest_full/stdout.log` | 3 | 0 | 0 | 0 | 0 | checkpoint field-match; RNG reseeded not continued |
| `7f4f09b` | `python -m pytest tests/observatory/test_organism_endpoints.py -q` | same | `audit_runs/pytest_full/stdout.log` | 3 | 0 | 0 | 0 | 0 | organism API endpoints behave |

Never infer counts from command exit code alone. Where pytest was interrupted, `NOT_COMPLETED`. No tests were interrupted.

### 10.3 Known failures and conflicting claims (restored)

- Sprint 10–11: original world-model failure preserved (reactive +0.48/+0.49/+0.42 vs recurrent −0.77/−1.16/−2.44 per Sprint 12 records). Distinguish original from post-bugfix reruns; do not soften.
- Sprint 12 statistics limitation: CI computed on per-model mean is NOT a CI for model-minus-baseline; individual overlapping CIs do not determine pairwise significance. Sprint 13 corrected to paired per-seed differences.
- checkpoint RNG: `PersistentOrganism.restore()` re-seeds controller RNG from stored seed; does not continue the stream. Not bit-exact deterministic replay. Disclosed, not hidden.
- `chss.py` vs `chsh.py`: verify actual file naming; do not assume.
- stale `head_commit` recorded during a commit in Sprint 12 Phase 1 (`05b64f8`). Version stamps must describe the SHA being audited (`7f4f09b`), not a future commit.
- CI interpretation error: a nonsignificant result is not proof of equivalence.

### 10.4 Actual coupling chain (restored)

| Source → destination | Quantity/units | Dynamic? | Mode | Test/log |
|---|---|---|---|---|
| Cosmology → Halo | scale factor, density | yes | numerical (PM3D) | `tests/unit/test_nbody_unit.py` |
| Halo → Star | not numerically coupled | no | missing/speculative | none |
| Star → Planet | not numerically coupled | no | missing/speculative | none |
| Planet → Habitat | planet spec → habitat patches | yes | reduced-order (PlanetSpec → Habitat) | `tests/planet`, `tests/habitat` |
| Habitat → Organism | patch chemistry → sensory observation | yes | numerical (local patch only) | `test_persistent_organism_checkpoint.py` |

The chain Cosmology → Halo → Star → Planet is NOT numerically connected; halos/stars/planets are independent seeded catalogs or illustrative. Do not fabricate a physical formation history.

### 10.5 Windows application checks (restored)

Host OS: Windows 11 10.0.26100-SP0. Windows build: NOT_TESTED (no clean-machine installer build executed). Installer on clean Windows: NOT_TESTED. Browser UI E2E: NOT_TESTED (no Playwright browser automation executed; `playwright_verify.js` exists but was not run). Create `docs/WINDOWS_ACCEPTANCE_CHECKLIST.md` for user execution on a real Windows machine.

### 10.6 Top 3 next actions (restored, NO implementation in this audit)

1. Publish this audit report + `docs/SCIENTIFIC_CLAIMS_MATRIX.md` + `docs/BRANCH_INTEGRATION_PLAN.md` + `docs/STATISTICAL_ANALYSIS_PLAN.md` + `docs/STATE_ARCHITECTURE_PROPOSAL.md` + `docs/WINDOWS_ACCEPTANCE_CHECKLIST.md` on the audit branch for user review.
2. User reviews audit and integration plan; explicitly approves which corrections/merges to proceed with.
3. On a separate isolated feature branch, implement the trainable recurrent baseline fix (Stage 5) and complete RNG-state continuation (Stage 7), preserving the audited baseline.

### 10.7 Audit artifacts (restored)

- `audit_runs/branch_inventory.json` — git refs/SHA/worktree inventory at `7f4f09b`.
- `audit_runs/module_inventory.json` — 153 Python files AST inventory (presence only, not behavior).
- `audit_runs/pytest_full/run.json` — test run manifest (exit 0, 527 passed, 2 skipped, 226.63s).
- `audit_runs/pytest_full/stdout.log` (sha256 `6262b9de5552ff0b0c948d56080728445f93fba367a8461c305cc5add4d30dff`) and `stderr.log`.
- `audit_runs/` is gitignored; reports in `docs/` are committed.

---

**Gate check (Stage 0):**
- [x] All audit documents committed on dedicated branch `audit/project-genesis-evidence`
- [x] No source changes (audit is read-only; working tree changes in primary worktree are pre-existing and untouched)
- [x] Test baseline recorded: 527 passed, 2 skipped, 0 failed, 0 hangs (commit `ef10c4c`)
