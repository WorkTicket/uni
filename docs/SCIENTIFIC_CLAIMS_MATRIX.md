# Universe Engine — Scientific Claims Matrix

**Audited commit:** `ef10c4c87e6ae916e08d459b2c0f68b868c90d75` (branch `audit/project-genesis-evidence`, based on `feature/project-genesis-research` `7f4f09b`)  
**Machine/versions:** Windows 11 10.0.26100-SP0, Python 3.14.5, pytest 9.2.0, numpy 2.5.4, PyYAML 6.0.3, scipy (installed), fastapi 0.139.0  
**Provenance tags used:** Simulated, Analytic, Reconstructed, Illustrative, Speculative (per `world_state.Provenance`)

Each row: claim, mathematical validity, empirical validation, evidence path/command/log, and whether on-screen scientific representation is authoritative or illustrative.

## Physics / cosmology

| Claim | Validity | Validation | Evidence | Authoritative or illustrative |
|---|---|---|---|---|
| FLRW background + H(a) from Ωm, Ωr, ΩΛ | Analytic, closed-form | Cross-checked vs independent code: age to 1.2e-5 relative | `status_static.json` `cosmology_cross_check`; `tests/unit/` | Simulated (1D laboratory, dimensionless units) |
| Linear matter growth (scale-independent) | Approximate analytic | Not a CLASS/CAMB transfer function; illustrative Gaussian IC | README §Scientific status | Illustrative initial field |
| 3D PM3D gravity (comoving, KDK, Poisson FFT) | Numerical | 10 regression tests: linear growth, uniform density zero-force, force symmetry, density mean zero | `tests/unit/test_nbody_unit.py`, `solvers/pm3d.py` | Simulated (PM3D state) |
| PM3D linear-growth validation | Approximate | 30% tolerance due to CIC + particle noise | status_static.json scientific_limitations | Simulated |
| 1D hydro (Euler) exact Riemann | Numerical, exact reference | T1 benchmark: 106 checks vs exact Riemann solution | `universe benchmark hydro` | Simulated (1D, dimensionless) |
| 1D gravity Jeans / two-fluid | Numerical | T1: Jeans growth, Poisson, conservation; momentum exact to 4e-18 | `universe benchmark gravity|nbody` | Simulated |
| Energy drift bounded | Numerical | Drift ≤ ~1e-3 in verified regimes; ~4e-3 strongly nonlinear | `session.py` DRIFT_WARNING_LEVEL; status_static.json | Simulated |

## Quantum (Program B)

| Claim | Validity | Validation | Evidence | Authoritative or illustrative |
|---|---|---|---|---|
| CHSH quantum S ≈ 2√2 ≈ 2.83 | Numerical (density matrix) | `test_quantum_chsh_max`, `test_chsh_optimal_settings_produce_violation`; S within 0.1 of 2√2 | `tests/research/test_quantum.py`, `research/quantum/chsh.py` | Simulated (toy two-qubit) |
| Classical CHSH bound S ≤ 2 not violated | Analytic | `test_classical_chsh_no_violation` | `tests/research/test_quantum.py` | Simulated |
| No-signalling marginals | Analytic | `result.no_signalling_alice` | `tests/research/test_quantum.py` | Simulated |
| Quantum universe / global wavefunction | — | **NOT CLAIMED / NOT IMPLEMENTED** (by design) | README, status_static.json | Not applicable |

> Note: `docs/IMPLEMENTATION_STATUS.json` test_command `python -c "from universe_engine.research.review import ..."` and `docs/SCIENTIFIC_REVIEW.md` report 6 of 7 modules passed. CHSH toy agreement does not imply a quantum universe.

## Research agents (Programs D1–D6)

| Claim | Validity | Validation | Evidence | Authoritative or illustrative |
|---|---|---|---|---|
| Reactive ≥ recurrent on fully-observable GridWorld | Empirical (null) | Reactive +0.48/+0.49/+0.42 vs recurrent −0.77/−1.16/−2.44 (original); null result retained | `docs/SPRINT_11_FINDINGS.md`, `tests/research/test_gridworld.py` | Illustrative agent task |
| Delayed-cue recurrent −0.06 > reactive −0.21 | Empirical, thin | Single/limited result; **insufficient for a claim** (≥20-seed paired CI not yet done) | `research/diagnostics/memory_tasks.py` | Illustrative |
| No reliable memory/world-model advantage | Empirical, 15 macro-seeds | CI [−0.138, +0.269] includes zero; Cohen's d reported | `docs/SPRINT_12_FINDINGS.md`, Phase 2 | Illustrative |
| Task-conditioned ECS-IW advantage | Hypothesis | Advantage only when task requires temporal integration | Sprint 11/12 findings | Hypothesis |
| Trainable memory agent solves delayed_cue / occluded_target | Empirical, FAILED | `test_trainable_agents_fail_memory_tasks`; mean < +0.2 delayed_cue; mean ≈ −1.0 occluded_target | `tests/research/test_memory_oracle.py` | learning failure, not demonstrated; not relabeled as oracle success |

## Persistent organism (Sprint 12)

| Claim | Validity | Validation | Evidence | Authoritative or illustrative |
|---|---|---|---|---|
| Persistent identity + body + memory + self-model | Numerical | `PersistentOrganism.checkpoint()`/`restore()`; `inspector_state()` | `research/persistent_organism.py`, `tests/research/`, `docs/SPRINT_12_FINDings.md` Phase 5-6 | Simulated |
| Actions change actual habitat chemistry | Numerical | `act()` consumes N, produces Wst on patch | `persistent_organism.py` `act()` | Simulated |
| Save/reload/resume == uninterrupted (hash) | **NOT YET DEMONSTRATED** | Required Stage 2 acceptance test; run A vs run B over many steps | Planned Stage 2 | Simulated (planned) |
| Consciousness / subjective experience | — | **NOT CLAIMED**; standing disclaimer: functional measures only | AGENTS.md, SCIENTIFIC_LIMITS.md | Not applicable |

## Multiscale (Program A9)

| Claim | Validity | Validation | Evidence | Authoritative or illustrative |
|---|---|---|---|---|
| Exact mass conservation | Numerical | Compared schemes documented; COM accuracy limited by region-based advection | `tests/research/test_multiscale.py`, `research/multiscale/adaptive.py` | Simulated |

## Integration chain (Cosmology → Organism)

| Link | Classification | Evidence |
|---|---|---|
| Cosmology → Halo | reduced-order | Density-peak thresholded candidates (no FoF/SO bound finder) |
| Halo → Star | illustrative / statistical | `sampleGalaxyStars` per galaxy |
| Star → Planet | illustrative | Probabilistic system samples, equilibrium temperature estimate |
| Planet → Habitat | numerical | `Habitat` reads `planet.environment()` each step |
| Habitat → Organism | numerical | Organism senses local patch only; acts on patch chemistry |

## Planetary / chemistry / prebiotic / replicators

| Claim | Validity | Validation | Evidence | Authoritative or illustrative |
|---|---|---|---|---|
| Planet 1D energy-balance climate | Numerical, Earth-tuned | 17 tests; NOT predictive for other worlds | `planet/model.py` | Simulated (Earth-tuned) |
| Chemistry mass-action, element/charge conserved | Numerical | 10 tests vs analytic; **rate parameters illustrative, not measured** | `chemistry/network.py` | Simulated |
| Prebiotic polymer/vesicle | Numerical, assumed params | 14 tests; kinetics/vesicle params assumed placeholders; not coupled to chemistry networks | `prebiotic/compartment.py` | Simulated |
| Replicators copying, tau-leap, lineage | Numerical | 9 tests incl. error threshold; duplex-stability placeholder params; not coupled to protocells/chemistry | `replicators/population.py` | Simulated |

## Viewer / observatory

| Claim | Validity | Validation | Evidence | Authoritative or illustrative |
|---|---|---|---|---|
| Procedural universe = f(seed, time, camera) | Illustrative | View-driven; regenerates detail from camera/zoom (Bug C); not authoritative state | `static/js/app.js`, `layers2.js` | Illustrative |
| Scientific Simulation mode = PM3D state | Simulated | Renders actual PM3D particle state from `/api/pm3d` | `server.py`, `simParticles` layer | Simulated |
| "Whole history Planck→galaxies" | Illustrative | Procedural; no CMB/CLASS/CAMB, no N-body galaxy formation | README §Scientific status | Illustrative |
| Viewer rendered in real browser | — | **NOT DONE**; only Node execution against real data | status_static.json | Not tested |
| Connected Mind Inspector UI | — | no graphical Mind Inspector test executed; endpoints exist but no open-planet→habitat→organism→follow→inspect-memory flow | `tests/observatory/test_organism_endpoints.py` | UI is a 3D universe-viewer HUD; no graphical Mind Inspector | NOT_TESTED (no browser E2E) |
| 20-step E2E on Windows | — | **NOT TESTED**; no desktop executable built/run | status_static.json, README | Not tested |

## Overall

- Verified numerical modules: quantum CHSH, PM3D gravity, 1D hydro/gravity, habitat, planet, chemistry, replicators, multiverse, world-state spine, self-model/workspace, discovery, multiscale, evolution, planetary ecology.
- Illustrative/prototype: viewer procedural mode, agent research tasks, prebiotic kinetics, stellar/planetary sampling.
- Not claimed: quantum universe, consciousness, global wavefunction, abiogenesis, end-to-end numerical cosmology→organism.
