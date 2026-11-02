# Audit Restoration Notes — findings lost between ef10c4c and 54b7fc1

**Source diff:** `git diff ef10c4c 54b7fc1 -- docs/AUDIT_REPORT.md docs/SCIENTIFIC_CLAIMS_MATRIX.md docs/BRANCH_INTEGRATION_PLAN.md docs/STATE_ARCHITECTURE_PROPOSAL.md`

**Commit `ef10c4c`** (older, parent of `54b7fc1`) contained findings that commit `54b7fc1` removed when it rewrote the four audit documents. This note lists every lost finding and restores the most critical ones back into the docs. Nothing was invented; all content below is copied verbatim from `ef10c4c`.

---

## A. AUDIT_REPORT.md — lost findings

### A1. Executive findings #3–8 (lost in full)
The old report began with an "Executive findings" section (8 numbered findings). The new report removed findings #3–8:

- **#3** Full pytest suite passes: `527 passed, 2 skipped` in `226.63s` at SHA `7f4f09b`. Manifest + logs saved under `audit_runs/pytest_full/` (run.json, stdout.log sha256 `6262b9de...`, stderr.log sha256 `e3b0c442...`). The 2 skips: `tests/test_acceptance_A_to_L.py:333` (desktop executable not built) and `tests/test_habitat.py:69` (placeholder kinetics).
- **#4** Category 1 (programmed) memory oracle validated: `WorkingMemoryOracle` solves delayed-cue and occluded_target, reading only the `SensoryObservation` contract (`local_light`, `nearby_resources`, `nearby_hazards`, `position`). Never reads `env.target_pos`/`env.cue_side`. Confirmed by `tests/research/test_memory_oracle.py` (10 tests, all passing).
- **#5** Category 2 (trainable) memory learning NOT demonstrated: `gated_recurrent`, `recurrent`, `world_model`, `reactive`, `stacked_obs` all fail delayed_cue (mean < +0.2) and occluded_target (mean ≈ −1.0). Negative result reported honestly, not relabeled as oracle success. Confirmed by `test_memory_oracle.py::test_trainable_agents_fail_memory_tasks`.
- **#6** Checkpoint/restore is NOT full deterministic replay. The controller RNG is re-seeded from the stored seed on `PersistentOrganism.restore()`; weights are restored but the RNG stream position is not. Documented honestly in `tests/research/test_persistent_organism_checkpoint.py`. Habitat chemistry, body, memory, self-model, developmental history, t_gyr match; stochastic action selection is deterministic-given-seed, not bitwise continuation.
- **#7** Organism API endpoints exist but are NOT a connected Mind Inspector UI. `POST /api/organism`, `GET /api/organism/{oid}`, `POST /api/organism/{oid}/step`, `GET /api/organism/{oid}/inspect`, `POST /api/organism/{oid}/checkpoint` are implemented and tested (`tests/observatory/test_organism_endpoints.py`). However, the frontend (`static/js/ui.js`) is a 3D universe-viewer HUD; there is no graphical Mind Inspector letting a user open a planet → select habitat → select organism → follow → inspect memory/workspace/predictions → pause/save/resume. Endpoints alone ≠ integrated visual interface.
- **#8** ECS-IW causal ablations deferred (Sprint 13): not run because no trainable baseline reliably solves a memory task. Recorded as NOT_TESTED, not skipped-as-passed.

### A2. Module evidence matrix — lost columns
The old report's module evidence matrix had two extra columns the new report removed: **Independent benchmark** and **Integration link**. It also had a per-module row for "Working-memory oracle" (VERIFIED), "Trainable recurrent/world-model" (FAILED), "ECS-IW self-model/workspace" (IMPLEMENTED_UNVERIFIED), "Persistent organisms" (PROTOTYPE), etc., each with test command, saved output log, independent benchmark, and integration link.

### A3. Test execution record table (lost in full)
| SHA | Command | Machine | Run manifest path | Pass | Fail | Skip | Error | Timeout | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `7f4f09b` | `python -m pytest -q -ra` | Windows 11, Python 3.14.5 | `audit_runs/pytest_full/run.json` | 527 | 0 | 2 | 0 | 0 | 529 collected; 2 skips |
| `7f4f09b` | `python -m pytest tests/research/test_memory_oracle.py -q` | same | `audit_runs/pytest_full/stdout.log` | 10 | 0 | 0 | 0 | 0 | oracle validates; trainables fail |
| `7f4f09b` | `python -m pytest tests/research/test_persistent_organism_checkpoint.py -q` | same | `audit_runs/pytest_full/stdout.log` | 3 | 0 | 0 | 0 | 0 | checkpoint field-match; RNG reseeded not continued |
| `7f4f09b` | `python -m pytest tests/observatory/test_organism_endpoints.py -q` | same | `audit_runs/pytest_full/stdout.log` | 3 | 0 | 0 | 0 | 0 | organism API endpoints behave |

Plus the note: "Never infer counts from command exit code alone. Where pytest was interrupted, `NOT_COMPLETED`. No tests were interrupted."

### A4. Known failures and conflicting claims (lost in full)
- Sprint 10–11: original world-model failure preserved (reactive +0.48/+0.49/+0.42 vs recurrent −0.77/−1.16/−2.44 per Sprint 12 records). Distinguish original from post-bugfix reruns; do not soften.
- Sprint 12 statistics limitation: CI computed on per-model mean is NOT a CI for model-minus-baseline; individual overlapping CIs do not determine pairwise significance. Sprint 13 corrected to paired per-seed differences.
- checkpoint RNG: `PersistentOrganism.restore()` re-seeds controller RNG from stored seed; does not continue the stream. Not bit-exact deterministic replay. Disclosed, not hidden.
- `chss.py` vs `chsh.py`: verify actual file naming; do not assume.
- stale `head_commit` recorded during a commit in Sprint 12 Phase 1 (`05b64f8`). Version stamps must describe the SHA being audited (`7f4f09b`), not a future commit.
- CI interpretation error: a nonsignificant result is not proof of equivalence.

### A5. Actual coupling chain table (lost in full)
| Source → destination | Quantity/units | Dynamic? | Mode | Test/log |
|---|---|---|---|---|
| Cosmology → Halo | scale factor, density | yes | numerical (PM3D) | `tests/unit/test_nbody_unit.py` |
| Halo → Star | not numerically coupled | no | missing/speculative | none |
| Star → Planet | not numerically coupled | no | missing/speculative | none |
| Planet → Habitat | planet spec → habitat patches | yes | reduced-order (PlanetSpec → Habitat) | `tests/planet`, `tests/habitat` |
| Habitat → Organism | patch chemistry → sensory observation | yes | numerical (local patch only) | `test_persistent_organism_checkpoint.py` |

Plus: "The chain Cosmology → Halo → Star → Planet is NOT numerically connected; halos/stars/planets are independent seeded catalogs or illustrative. Do not fabricate a physical formation history."

### A6. Windows application checks (lost in full)
Host OS: Windows 11 10.0.26100-SP0. Windows build: NOT_TESTED (no clean-machine installer build executed). Installer on clean Windows: NOT_TESTED. Browser UI E2E: NOT_TESTED (no Playwright browser automation executed; `playwright_verify.js` exists but was not run). Create `docs/WINDOWS_ACCEPTANCE_CHECKLIST.md` for user execution on a real Windows machine.

### A7. Top 3 next actions (lost in full)
1. Publish this audit report + `docs/SCIENTIFIC_CLAIMS_MATRIX.md` + `docs/BRANCH_INTEGRATION_PLAN.md` + `docs/STATISTICAL_ANALYSIS_PLAN.md` + `docs/STATE_ARCHITECTURE_PROPOSAL.md` + `docs/WINDOWS_ACCEPTANCE_CHECKLIST.md` on the audit branch for user review.
2. User reviews audit and integration plan; explicitly approves which corrections/merges to proceed with.
3. On a separate isolated feature branch, implement the trainable recurrent baseline fix (Stage 5) and complete RNG-state continuation (Stage 7), preserving the audited baseline.

### A8. Audit artifacts (lost in full)
- `audit_runs/branch_inventory.json` — git refs/SHA/worktree inventory at `7f4f09b`.
- `audit_runs/module_inventory.json` — 153 Python files AST inventory (presence only, not behavior).
- `audit_runs/pytest_full/run.json` — test run manifest (exit 0, 527 passed, 2 skipped, 226.63s).
- `audit_runs/pytest_full/stdout.log` (sha256 `6262b9de5552ff0b0c948d56080728445f93fba367a8461c305cc5add4d30dff`) and `stderr.log`.
- `audit_runs/` is gitignored; reports in `docs/` are committed.

---

## B. SCIENTIFIC_CLAIMS_MATRIX.md — lost rows

The old matrix used columns: Claim | Source code / SHA | Evidence: command + run log | Independent reference or invariant | Caveat | Rendering provenance | Status. Rows removed:

| Claim | Status |
|---|---|
| Trainable memory agent solves delayed_cue / occluded_target | FAILED (learning not demonstrated) |
| Connected Mind Inspector UI | NOT_TESTED (no browser E2E) |
| Organism API endpoints | IMPLEMENTED_UNVERIFIED |
| Brane dynamics | PROTOTYPE |

---

## C. BRANCH_INTEGRATION_PLAN.md — lost sections

Removed: detailed "Existing refs" table with **Unique modules** and **Potential conflicts** columns; full "Safeguards" section (6 bullets); full "Sequence after user approval" (5 steps); full "Rollback design".

---

## D. STATE_ARCHITECTURE_PROPOSAL.md — lost content

Removed: concrete Python dataclass contracts (`Provenance` enum, `Quantity`, `Entity`, `UniverseClock`, `CheckpointManifest`); coupling interface signature (`ModelAdapter.advance(input_state, dt, boundary_conditions) -> (output_state, exchanges, uncertainty, provenance)`); 6 open design questions; "Evidence required before implementation" section.

---

## Restoration performed

The following lost content was restored into `docs/AUDIT_REPORT.md`:
- A1 (Executive findings #3–8) appended as a new section "10. Restored executive findings (from ef10c4c)".
- A3 (Test execution record table) appended as part of section 10.
- A4 (Known failures and conflicting claims) appended as part of section 10.
- A5 (Actual coupling chain table) appended as part of section 10.
- A6 (Windows application checks) appended as part of section 10.
- A7 (Top 3 next actions) appended as part of section 10.
- A8 (Audit artifacts) appended as part of section 10.

Rows B1–B4 were noted here; the current `SCIENTIFIC_CLAIMS_MATRIX.md` already covers the substance of these claims in its "Research agents", "Persistent organism", and "Viewer / observatory" sections, so no duplicate rows were added.

Sections C and D were noted here; the current `BRANCH_INTEGRATION_PLAN.md` and `STATE_ARCHITECTURE_PROPOSAL.md` retain the substance in condensed form, so no duplicate content was added.
