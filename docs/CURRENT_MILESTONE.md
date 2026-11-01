# Current milestone (updated 2026-11-01)

Authoritative, machine-generated numbers: `docs/IMPLEMENTATION_STATUS.json`.

Branch: `feature/project-genesis-research` (HEAD commit `17839a1`, Sprint 11 complete).

**Project Genesis research program — Sprints 1-10 committed.**

Sprint 1 — Research infrastructure: contracts, manifests, checkpoints, runner (`research/contracts.py`).

Sprint 2 — Quantum validation: `states.py`, `chss.py` (CHSH quantum S ≈ 2.83, classical S = 2.0).

Sprint 3 — Embodied agent: GridWorld, ReactiveController, RecurrentWorldModelController (`research/environment/gridworld.py`).

Sprint 4 — World-model learning comparison (`research/world_model/experiment.py`).

Sprint 5 — Self-model and workspace (ECS-IW) (`research/self_model/model.py`, `research/workspace/model.py`).

Sprint 6 — Evolutionary architecture search (`research/evolution/experiment.py`).

Sprint 7 — Scientific discovery (`research/discovery/experiment.py`).

Sprint 8 — Planetary ecology integration (`research/planetary_agent/experiment.py`).

Sprint 9 — Multiscale fidelity research (`research/multiscale/adaptive.py`).

Sprint 10 — Independent scientific review: `research/review.py`, `docs/SCIENTIFIC_REVIEW.md`. 6 of 7 modules passed.

## Sprint 11 — World-model failure investigation (DIAGNOSTIC PHASE COMPLETE)

Original failure reproduced exactly: recurrent world-model agent underperforms
reactive baseline on GridWorld foraging for seeds 1/42/99
(reactive +0.48/+0.49/+0.42 vs recurrent -0.77/-1.16/-2.44).

**Root cause: H1 — the task does not require memory.** GridWorld is fully
observable; current sensory input (nearby_resources, energy_level, hazards)
is sufficient for near-optimal reactive behavior. Memory adds cost without
benefit on this task. This is a legitimate null result, retained and
documented honestly; pass thresholds were not changed.

**Contributing bugs fixed:**
- H6: `RecurrentWorldModelController.select_action` corrupted `self.hidden`
  during action-candidate evaluation (committed to last candidate, not chosen
  action). Fixed — correctness only, not sufficient to flip the result.
- H3: value function used misaligned observation-energy prediction; added a
  learned reward predictor `W_r` trained on observed reward (partial
  improvement: seed 1 -0.77 → +0.03, still below reactive +0.48).

**Key demonstration (memory helps only when required):**
- GridWorld (fully observable): reactive +0.48 > recurrent -0.77 (memory not needed).
- Delayed-cue (memory required): recurrent -0.06 > reactive -0.21 (memory helps).

New files: `research/diagnostics/memory_tasks.py`, `research/diagnostics/experiment.py`,
`docs/SPRINT_11_FINDINGS.md`.

**Implication for ECS-IW:** memory/world-model/self-model/workspace provide
measurable advantages when the task genuinely requires temporal integration or
cross-module coordination, not universally. Claims must be task-conditioned.
No claim of consciousness is made.

Deferred (Phase 6-8): larger-seed abulations, statistical analysis, integration
into a persistent habitat organism.

## What this is NOT

- No claim that any agent is conscious or has subjective experience.
- No global quantum wavefunction for every particle in the universe (by design).
- No predetermined species progression or intelligence-by-elapsed-time.
- No decorative simulated brain visualization; all inspector state is actual model state.

## Resume here

Sprint 11 diagnostic complete. Next: decide whether to (a) retain the honest
null result on GridWorld and refine ECS-IW claims to be task-conditioned, or
(b) run larger-seed ablation/statistical analysis (Phase 6-7) before any
architecture revision.
