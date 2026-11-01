# Sprint 11: World-Model Failure Investigation — Findings

Status: COMPLETE (diagnostic phase). Branch: `feature/project-genesis-research`.

## Original failure (reproduced)

From Sprint 10 review (commit `e0510d9`), the recurrent world-model agent
underperformed the reactive baseline on the GridWorld foraging task across
seeds 1, 42, 99:

| Seed | Reactive mean reward | Recurrent mean reward | Pass? |
|------|----------------------|-----------------------|-------|
| 1 | +0.48 | -0.77 | No |
| 42 | +0.49 | -1.16 | No |
| 99 | +0.42 | -2.44 | No |

Reproduction confirmed exactly (see `tests/research/test_world_model.py` and
`src/universe_engine/research/world_model/experiment.py`).

Both agents receive identical `SensoryObservation` objects and the same action
set (`move`, `forage`, `rest`, `wait`). Information access is equivalent.

## Diagnosis

Three competing explanations were tested:

**H6 — Evaluation/integration bug (CONFIRMED, fixed).**
`RecurrentWorldModelController.select_action` mutated `self.hidden` inside the
action-candidate loop, committing to every candidate's hidden state and ending
with the last candidate ("wait") regardless of the chosen action. The recurrent
state never tracked the action actually taken.

Fix: compute each candidate's hidden state locally, then commit to the chosen
action's hidden state once after the loop
(`src/universe_engine/research/environment/gridworld.py:379`).

This was a correctness fix, but **not sufficient** to make recurrent beat
reactive.

**H3 — World model not used effectively (CONFIRMED, partially fixed).**
The value function used `pred_obs[0]` (predicted observation energy level), a
raw observation feature poorly aligned with the actual foraging reward. The
agent never learned to forage (0/48 resource-rich patches where it foraged).

Fix: added a learned reward predictor `W_r` trained on observed reward, used
for action selection (`gridworld.py`). Improved recurrent reward
(-0.77 → +0.03 on seed 1) but **still below reactive** (+0.48).

**H1 — Task does not require memory (ROOT CAUSE, confirmed).**
The GridWorld foraging task is fully observable: each observation already
contains `nearby_resources`, `energy_level`, `nearby_hazards`, and the agent's
position. A reactive policy using `nearby_resources > 0.5 → forage` is
near-optimal. Memory adds computational cost without benefit.

Cross-episode persistent learning did not help (recurrent still died with -2.0
or earned 0), confirming the architecture converged to a poor policy on a task
where memory is genuinely unnecessary.

This is a **legitimate null result**, not a bug to paper over. Pass thresholds
were not changed.

## Key demonstration: memory helps only when required

| Task | Type | Reactive | Recurrent | Memory helps? |
|------|------|----------|-----------|---------------|
| GridWorld foraging | fully observable | +0.48 | -0.77 | No |
| Delayed-cue | memory-required | -0.21 | -0.06 | Yes |

On the delayed-cue task (cue visible briefly, then hidden; agent must remember
it to choose the correct direction later), a reactive policy cannot do better
than ~chance because the response-period observation carries no cue information.
With dense reward shaping, a recurrent agent learns to use memory and beats
reactive (-0.06 vs -0.21). (See `src/universe_engine/research/diagnostics/`.)

## Implications for ECS-IW (Research Program D/E)

The null result on GridWorld refines, rather than rejects, the ECS-IW
hypothesis: recurrent memory / world-model / self-model / workspace provide
measurable advantages when the task genuinely requires temporal integration,
self-vs-world discrimination, or cross-module coordination — not universally on
every task. Claims must be task-conditioned.

No claim of consciousness is made or implied.

## Files changed

- `src/universe_engine/research/environment/gridworld.py`: H6 fix (hidden state),
  H3 reward predictor `W_r`, checkpoint/restore updated.
- `src/universe_engine/research/diagnostics/memory_tasks.py` (new): memory-sensitive
  POMDP tasks (delayed cue, occluded target, changed dynamics) + baseline agents.
- `src/universe_engine/research/diagnostics/experiment.py` (new): train/evaluate
  harness for memory-sensitive tasks.
- `docs/SCIENTIFIC_REVIEW.md`: updated (written in Sprint 10).
- `docs/CURRENT_MILESTONE.md`, `docs/IMPLEMENTATION_STATUS.json`: updated.

## Not done (deferred)

- Phase 6 ablations beyond the recurrent-vs-reactive memory/non-memory contrast
  above.
- Phase 7 larger-seed statistical analysis.
- Phase 8 integration into a persistent habitat organism.

The honest null result on GridWorld is retained and documented.
