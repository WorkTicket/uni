# Sprint 13 Findings — Memory Tasks, Paired Statistics, and Programmed vs Learned Memory

Branch: `feature/project-genesis-research`

**No claim of consciousness, subjective experience, or artificial sentience is made or implied.** These are measured behavioral and architectural results only. Results are divided into two clearly separated categories:

| Category | Example | What it demonstrates |
|----------|---------|----------------------|
| **Category 1 — Programmed intelligence** | `working_memory_oracle`: explicitly remembers target/cue coordinates and navigates to them | The environment + memory mechanism work (positive control / oracle) |
| **Category 2 — Learned intelligence** | `gated_recurrent`, `recurrent`, `world_model`: trainable agents | Adaptation through experience — **not yet demonstrated** on memory tasks |

Category 1 success is NOT relabeled as Category 2 learning. Where trainable agents fail, the negative result is reported.

---

## 1. Statistical method (corrected)

The primary estimand is the **paired reward difference**:

`difference_i = reward_memory_controller_i - reward_reactive_i`

computed on **matched macro-seeds** (same environment seeds, same training/eval protocol) so environment variance cancels and the difference isolates the controller advantage. Confidence intervals are calculated **directly on these paired differences** (normal approximation, 95%). A CI excluding zero indicates a statistically reliable advantage.

This corrects the earlier error of reporting an individual controller's mean-reward CI and interpreting it as an advantage-over-reactive CI — a mean-reward CI is not an advantage CI.

Each paired result below uses **15 macro-seeds × 8 evaluation episodes = 120 paired differences** per comparison, with training (25 train-seeds × 60 steps) and held-out evaluation on disjoint seeds. Evaluation controllers are cloned from trained weights but never tuned on the final test seeds.

---

## 2. GridWorld foraging (fully observable) — 15 macro-seeds

| Controller | Mean reward | Paired advantage vs reactive | 95% paired CI | Excludes zero? |
|------------|-------------|------------------------------|---------------|----------------|
| Reactive | −0.594 | 0.000 | [0.000, 0.000] | — |
| Stacked-obs | −0.710 | −0.117 | [−0.354, +0.121] | No |
| Recurrent | −0.241 | +0.353 | [+0.137, +0.570] | **Yes** |
| World-model | −0.396 | +0.197 | [−0.092, +0.487] | No |

**Interpretation:** On the fully-observable foraging task, only the **recurrent** controller shows a reliable paired advantage over reactive (CI excludes zero). This is **not** a memory-required task (current observations suffice), so it does not demonstrate that memory helps. Consistent with Sprint 12: memory is not beneficial when current observations suffice.

---

## 3. Memory-required tasks (Category 1 oracle vs Category 2 trainable)

The memory tasks were repaired so the task-relevant signal is carried by the **sensory observation contract** (not hidden in `extra`):

- `local_light == 1.0` = informative phase (target/cue visible); `0.0` = memory phase (hidden).
- `nearby_resources` / `nearby_hazards` carry normalized target/cue coordinates during the informative phase and the agent's own normalized position during the memory phase.
- `position` is the agent's own observable grid coordinates.

This makes the tasks solvable with memory **without** reading hidden environment state.

| Task | Reactive mean | Category 1 oracle mean | Oracle paired advantage CI | Oracle excludes zero? | Category 2 (trainable) |
|------|---------------|------------------------|----------------------------|-----------------------|------------------------|
| delayed_cue | −0.073 | +0.744 | [+0.800, +0.835] | **Yes** | All fail (mean < +0.2) |
| occluded_target | −0.976 | +1.957 | [+2.849, +3.018] | **Yes** | All fail (mean ≈ −1.0) |
| changed_dynamics | −0.463 | −0.600 (oracle not designed) | [−0.366, +0.093] | No | All fail |

**delayed_cue (Category 1 oracle):** +0.744 mean, paired advantage CI [+0.800, +0.835] excludes zero. The oracle remembers the cue side during the cue phase and moves in that direction during the response phase. **Validated:** the environment + memory mechanism works.

**occluded_target (Category 1 oracle):** +1.957 mean, paired advantage CI [+2.849, +3.018] excludes zero. The oracle remembers the target (x, y) during the visible phase, tracks its own position during occlusion, moves toward the remembered target, and claims on arrival. **Validated:** the environment + memory mechanism works.

**changed_dynamics:** No controller (oracle or trainable) solves it reliably. It is a change-detection/adaptation task, not a pure memory task, and remains unresolved.

**Category 2 (learned intelligence):** `gated_recurrent`, `recurrent`, and `world_model` all fail to solve delayed_cue and occluded_target despite 25 training episodes × 60 steps. This is a confirmed **learning/optimization failure**, reported honestly — not relabeled as the oracle's success.

---

## 4. ECS-IW causal ablations — deferred

Not run in this pass. Per the assignment: ablations are only meaningful after a trainable architecture reliably solves at least one task and establishes a matched baseline. Since Category 2 learning is not yet demonstrated, there is no established success to attribute to any mechanism (recurrent state, self-model, workspace broadcasting, world model). Ablations on null results would not support a causal claim and are deferred to a later pass.

---

## 5. Regression tests

Added `tests/research/test_memory_oracle.py` (10 tests), all passing:

- Coordinate contract: visible phase encodes target coords; occlusion hides target (`target_pos` None, `local_light == 0.0`).
- Claiming: claim at target → hit_reward; claim away → miss_penalty.
- Oracle behavior: solves delayed_cue (≥90% win rate over 40 seeds) and occluded_target (≥90% win rate).
- Information leakage: oracle reads only the observation contract, never environment internals / hidden target.
- Negative result: trainable agents do not solve delayed_cue (mean < +0.2).

---

## 6. Honest status

| Objective | Status |
|-----------|--------|
| Category 1 memory oracle solves delayed_cue | ✅ Verified (paired CI excludes zero) |
| Category 1 memory oracle solves occluded_target | ✅ Verified (paired CI excludes zero) |
| Category 2 learned memory agent | ❌ Not demonstrated (learning failure) |
| changed_dynamics solved | ❌ Unresolved |
| ECS-IW causal ablations | ⏳ Deferred (no baseline to ablate) |
| Full pytest suite | ✅ Passes (512 passed, 1 skipped, ~6 min) |

**Bottom line:** The environment and memory mechanism are validated (Category 1 oracle reliably solves delayed_cue and occluded_target with statistically reliable paired advantages). Trainable memory-based learning is **not yet demonstrated** — the gated recurrent, recurrent, and world-model controllers fail on these tasks. This negative result is reported honestly. The path forward is a stronger learner, more training budget, eligibility traces / credit assignment over longer horizons, and held-out evaluation discipline before any ECS-IW mechanism can be causally attributed.
