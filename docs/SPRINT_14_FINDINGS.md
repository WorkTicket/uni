# Sprint 14 Findings — Universe Engine Scientific Audit

**Audited SHA:** `7f4f09b86a44bafd5f92bfe84ec04f2385e7ff1d` (`feature/project-genesis-research`, Sprint 13 commit)  
**Audit branch:** `audit/project-genesis-evidence`  
**UTC date:** `2026-11-01T20:45:00Z`  
**No claim of consciousness, subjective experience, or artificial sentience is made or implied.** Measured behavioral and architectural results only.

## What is scientifically validated

- **PM3D cosmology:** particle-mesh comoving gravity in an expanding FLRW background; CIC deposition, FFT Poisson, kick-drift-kick; unit tests pass (CIC, uniform lattice zero force, total momentum exactly zero). Classification: SIMULATED (numerical), cold dark matter only, periodic box.
- **FLRW cosmology/coordinates:** background cosmology, scale factor, redshift, comoving transforms; unit tests pass.
- **Quantum CHSH (toy):** density-matrix CHSH, local classical bound S ≤ 2, quantum expectation 2√2, no-signalling marginals. Not a full quantum universe.
- **Working-memory oracle (programmed):** solves delayed-cue and occluded_target; reads only the `SensoryObservation` contract (`local_light`, `nearby_resources`, `nearby_hazards`, `position`); never reads `env.target_pos`/`env.cue_side`. Paired advantage CIs exclude zero. Positive control / oracle only — NOT learned intelligence.
- **Habitat chemistry:** patch chemistry ODE, nutrient/waste, mass ledger; unit tests pass.

## What is working only as a prototype

- **Persistent organisms:** identity, body, habitat patch, episodic memory, self-model, checkpoint/replay exist and pass field-by-field matching tests. **BUT** controller RNG is re-seeded from the stored seed on restore rather than continued — not bit-exact deterministic replay. Status: PROTOTYPE.
- **ECS-IW self-model/workspace:** architecture defined (workspace broadcast, self-model, recurrence) with ablation framework; not yet demonstrated to confer a memory advantage on held-out tasks. Status: IMPLEMENTED_UNVERIFIED.
- **Multiverse/brane:** universe registry and brane dynamics implemented but labeled `speculative_research`; not integrated physical multiverse. Status: PROTOTYPE.

## Which cognitive abilities are programmed versus learned

| Ability | Programmed | Learned | Evidence |
|---|---|---|---|
| Remember cue/target across delay/occlusion | ✅ `WorkingMemoryOracle` explicitly stores coords | ❌ `recurrent`/`world_model`/`gated_recurrent` all fail | `test_memory_oracle.py` |
| Navigate to remembered location | ✅ oracle moves toward remembered coords | ❌ trainables mean < +0.2 delayed_cue, ≈ −1.0 occluded_target | same |
| Claim on arrival | ✅ oracle claims at remembered target | ❌ not demonstrated | same |
| Internal model that can be wrong | ✅ self-model calibration_error tracked | ❌ not shown to improve control | `test_self_model_workspace.py` |

## Which modules remain disconnected

- Cosmology → Halo → Star → Planet is **not numerically connected**; independent seeded catalogs or illustrative. No fabricated formation history.
- Habitat → Organism is numerically coupled (local patch chemistry → sensory observation → action on habitat).
- Organism API endpoints exist (`/api/organism*`) but are **not connected to a graphical Mind Inspector UI**; the frontend is a 3D universe viewer HUD.
- No open-planet → select-habitat → select-organism → follow → inspect-memory/workspace/predictions → pause/save/resume flow in the actual application.

## Which scientific hypotheses are speculative

- Brane-world / higher-dimensional physics: labeled `speculative_research`.
- Multiverse origin events: labeled speculative; many-worlds interpretations not treated as distinct observed predictions.
- Abiogenesis / emergent replicators: seeded replicators only, not demonstrated abiogenesis.

## Which experiments failed

- **Trainable memory learning:** `gated_recurrent`, `recurrent`, `world_model` fail delayed_cue (mean < +0.2) and occluded_target (mean ≈ −1.0) on held-out seeds. Confirmed negative result; not relabeled as oracle success.
- **changed_dynamics:** no controller (oracle or trainable) solves it reliably; unresolved.
- **ECS-IW causal ablations:** deferred (no trainable baseline to ablate).
- **Full deterministic replay (bit-exact):** not achieved — controller RNG re-seeded, not continued.

## Which tests are blocked

- **Windows desktop executable / installer / clean-machine / browser E2E:** NOT_TESTED (no Windows build executed, no Playwright browser automation run).
- **Causal ablations:** blocked pending a trainable baseline that reliably solves a memory task.

## Which features are accessible in the actual application

- 3D universe viewer (`/`, `/lab`, `/research-lab`, `/hyperspace`) with PM3D cosmology, playback, camera orbit/follow.
- REST observatory API: `/api/cosmology`, `/api/state`, `/api/snapshot`, `/api/step`, `/api/run`, `/api/pause`, `/api/save`, `/api/load`, `/api/inspect`.
- Planetary habitat read-only API: `/api/habitat/{id}`, `/chemistry`, `/population`, `/lineage`, `/resources`.
- Persistent organism API: `POST /api/organism`, `GET /api/organism/{id}`, `POST /api/organism/{id}/step`, `GET /api/organism/{id}/inspect`, `POST /api/organism/{id}/checkpoint`. Verified via FastAPI TestClient.
- Multiverse, brane, Planck-scale, origin-event, geometry endpoints (labeled speculative/research).

**Not accessible:** a graphical Mind Inspector that lets a user open a planet, select a habitat, select an organism, follow it, inspect energy/resources/sensory/memory/predictions/recurrent-workspace state, pause, save/reload, and resume the same organism.
