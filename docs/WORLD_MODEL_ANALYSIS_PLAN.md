# World-Model Analysis Plan — Universe Engine (Stage 0b.4)

**Version/date:** 1.0, 2026-11-01  
**SHA:** `7f4f09b` (feature/project-genesis-research, Sprint 13)  
**Machine:** Windows 11 10.0.26100-SP0, Python 3.14.5, NumPy 2.5.1  
**Status:** PREREGISTERED — written before any new comparison runs.

---

## 1. Research question / confirmatory hypothesis

Can a recurrent (hidden-state) controller learn to imitate a hand-coded reactive policy on memory-required tasks, under matched observations and matched training budgets?  
Primary comparison: **recurrent/world-model controllers vs. learned feedforward reactive baseline** on one prespecified memory-required environment.  
Negative control: fully observable GridWorld (memory-irrelevant), where recurrence should NOT help.

> Per the Sprint 11 finding, "fully observable task is root cause of recurrent agent losing" is treated as H1 *hypothesis*, not established fact. Recurrent underperformance can also reflect poor training, an implementation bug, or an unfair baseline. This plan tests those alternatives explicitly.

---

## 2. Agents under comparison (same observation contract, identical action space)

| Agent | Type | Memory | Trainable? | Source |
|---|---|---|---|---|
| ReactiveController | feedforward reactive | none | Yes (policy weights W) | `statistical_replication.py` |
| RecurrentController | recurrent hidden state (REINFORCE) | hidden state | Yes | `statistical_replication.py` |
| WorldModelController | predictive world model | hidden + model | Yes | `statistical_replication.py` |
| GatedRecurrentController | gated recurrent (actor-critic) | hidden | Yes | `statistical_replication.py` |
| StackedObsController | feedforward with obs stack | finite stack | Yes | `memory_tasks.py` |
| WorkingMemoryOracle | programmed memory | exact | No (positive control) | `statistical_replication.py` |

All agents read the same `SensoryObservation` contract (`local_light`, `nearby_resources`, `nearby_hazards`, `position`, `local_temperature`, `energy_level`, `nutrient_level`, `nearby_agents`) and the same `AgentAction` action space (`move`/`forage`/`rest`/`wait` with `dx`/`dy` parameters). No agent gets privileged access to global simulator state.

---

## 3. Primary comparison (confirmatory)

- **Primary environment:** `DelayedCueTask` (memory required: cue visible, then hidden during delay/response; a memoryless policy cannot do better than chance).
- **Primary outcome:** held-out episodic return (total reward per episode; higher is better).
- **Primary hypothesis:** the best trained recurrent/world-model controller achieves a statistically significantly higher held-out return than the trained feedforward reactive baseline.
- **Secondary (exploratory):** each candidate vs. reactive on `DelayedCueTask`; each candidate vs. reactive on `OccludedTargetTask`; all candidates vs. reactive on fully observable GridWorld (negative control, expect no advantage).

Do NOT substitute a hand-coded (non-learned) reactive policy for the *learned* reactive baseline in the primary comparison. If a hand-coded reactive baseline is also reported, it is labeled as a separate, non-primary reference.

---

## 4. Independent replication unit

- One independent training run = one training seed.
- Within each run, average held-out evaluations across matched held-out episodes (same held-out environment seeds for all agents in that run).
- Target: **>= 20 independent training seeds** per agent where feasible. Justify any smaller number and mark the result INCONCLUSIVE if underpowered.
- No overlap between training seeds, hyperparameter-tuning seeds, and final-evaluation seeds.
- The oracle (WorkingMemoryOracle) is not trained; it is evaluated once per evaluation seed as a positive control.

---

## 5. Matched budgets (documented BEFORE running)

For every trainable agent, fixed in advance and identical across agents:
- Training episodes per seed (e.g., 500)
- Training steps per episode
- Compute time wall-clock cap per seed
- Policy capacity (hidden_dim, W shape, lr) matched where architecture permits
- Action space, observation vector, reward function
- RNG reset rule: fresh `np.random.default_rng(seed)` per agent per seed; controller RNG state recorded
- Evaluation: fixed number of held-out episodes per run with fixed held-out environment seeds

---

## 6. Primary analysis

For each independent training replicate `r`:
1. Train the recurrent/world-model candidate and the learned reactive baseline on matched training data with matched budgets.
2. Evaluate both on the SAME held-out episodes (same held-out env seeds).
3. Compute per-replicate paired difference: `d_r = mean(candidate return) - mean(learned reactive return)` over matched held-out episodes.

**Primary test:** 95% CI for the **mean paired difference** `d_bar` across independent training replicates (bootstrap at the training-replicate level; pre-specify B = 10,000 bootstrap resamples, bootstrap seed fixed in advance).  
- If CI excludes 0 in the positive direction → reject null, candidate advantage supported.  
- If CI includes 0 → insufficient evidence of advantage (not proof of equivalence).

**Effect size:** paired standardized difference (Cohen's d for paired replicates). Report estimate and CI.

**Do not** equate a CI for each policy's mean return with a CI for the difference; overlapping individual CIs do NOT determine pairwise significance.

---

## 7. Secondary comparisons / multiplicity

List in advance:
1. Recurrent vs. reactive on DelayedCue (primary, confirmatory)
2. World model vs. reactive on DelayedCue
3. Gated recurrent vs. reactive on DelayedCue
4. Stacked obs vs. reactive on DelayedCue
5. Each candidate vs. reactive on OccludedTarget
6. Each candidate vs. reactive on fully observable GridWorld (negative control)

Apply **Holm step-down correction** to secondary p-values; report unadjusted and adjusted p-values. No post-hoc subgroup selection for confirmatory claims.

---

## 8. Tuning vs. evaluation seeds (separate)

- **Tuning seeds:** used only to select hyperparameters (lr, hidden_dim, episodes). Kept separate from evaluation seeds.
- **Evaluation seeds:** used for the final confirmatory comparison. Never used for tuning.
- Report which seeds were used for tuning and which for evaluation; never reuse evaluation data for tuning after seeing outcomes.

---

## 9. Failure / missingness handling

- Crashes and timeouts are logged as failures; do not exclude outcomes without a predeclared rule.
- Provide all seeds and raw outputs (per-seed CSV).
- Never re-run only unfavorable seeds ("cherry-picking").
- If a run times out or errors, record it as a failure for that seed; do not silently drop it.

---

## 10. Stop rule

- Fixed number of independent training runs and evaluations defined before seeing outcomes.
- No optional stopping for favorable results (no "stop when significant").
- If resources run out before 20 seeds, report the number achieved and mark results INCONCLUSIVE.

---

## 11. Environment / reproducibility

Record and commit:
- Git commit SHA, branch
- Python version, NumPy version, all library versions
- Hardware (CPU, OS), CPU thread counts (OMP_NUM_THREADS, etc.)
- RNG settings and seeds (training, tuning, evaluation, bootstrap)
- Benchmark config and raw JSON/CSV outputs (per-seed)

---

## 12. Report wording

- A nonsignificant result is **insufficient evidence of positive advantage**; it does NOT prove equivalence or no benefit.
- Report the effect estimate and its uncertainty (CI), not "proved null."
- Label negative results and bugs distinctly; never hide them.
- Functional measures only; do not claim subjective experience or consciousness.

---

## 13. Deliverables

- `docs/WORLD_MODEL_ANALYSIS_PLAN.md` (this document, committed before runs)
- Per-seed raw CSV of paired evaluation outcomes (same task + same eval seed for both agents)
- Manifest + run logs with SHA256
- `docs/IMPLEMENTATION_STATUS.json` regenerated from run output (never hand-edited), with per-seed failure records restored
