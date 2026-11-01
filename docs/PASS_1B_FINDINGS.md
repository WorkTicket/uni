# PASS 1B Findings — RL Sanity, Trainable Memory, and Delayed-Cue Learning

Branch: `fix/trainable-memory-baseline`
Code revision (git SHA): `d6c0f93af099a7e8a0ea300ff55077a8f2e805af`
Full pytest: **542 passed, 1 skipped** (before PHASE 2 test addition); **543 passed, 1 skipped** (after adding the deterministic checkpoint/replay test).

**No claim of consciousness, subjective experience, or artificial sentience is made or implied.** These are measured behavioral and architectural results only.

Results are divided into clearly separated categories:

| Category | Example | What it demonstrates |
|----------|---------|----------------------|
| **Supervised recurrent memory** | `GRUActorCriticController` on binary-cue retention | The recurrent network can retain a cue across a delay (100% accuracy at delays 1, 3, 5, 10, 20). |
| **Reward-driven action learning (bandits)** | REINFORCE `TorchGRUActorCriticController` on two-action and contextual bandits | The policy-gradient machinery learns to select the higher-reward action from rewards alone. |
| **Reward-driven use of memory (delayed-cue)** | `GRUActorCriticController` / `FeedForwardController` on delayed_cue and occluded_target | **NOT reliably demonstrated.** Reported as a negative result. |

---

## 1. Supervised GRU memory retention — demonstrated

A PyTorch GRU (`GRUActorCriticController` in `src/universe_engine/research/diagnostics/trainable_memory.py`) was trained on a supervised binary-cue retention task:

- Task: observe a cue side during an informative phase (`local_light == 1.0`), retain it across a delay, then act on it during a response phase.
- Delays evaluated: 1, 3, 5, 10, 20 steps.
- Result: **100% accuracy across all five delays.**

This is **supervised learning** with a cue-prediction auxiliary objective (backprop-through-time through the GRU). It demonstrates that the recurrent architecture can retain information in hidden state across a delay. It is NOT, by itself, a demonstration of reward-driven control on a memory-required task.

Source: `src/universe_engine/research/diagnostics/trainable_memory.py:180` (`GRUActorCriticController`), `_run_episode` / `train_and_evaluate`.

---

## 2. Corrected two-action bandit — reward-driven action learning demonstrated

`tests/research/test_torch_rl_sanity.py::test_bandit_learns_positive_action`

- Single state; action index 1 yields +1, action index 0 yields -1.
- The agent must learn the +1 action from rewards alone via REINFORCE.
- Result: the agent learns the positive action (final mean reward over last 500 episodes > +0.5). **Passes.**

This demonstrates basic reward-driven action learning (policy gradient works on a memory-free problem).

---

## 3. Corrected contextual bandit — state-dependent reward learning demonstrated

`tests/research/test_torch_rl_sanity.py::test_contextual_bandit_learns_state_dependent_policy`

- Two states; best action == current state (state toggles each episode).
- A feedforward/recurrent policy must learn the state->action mapping via RL.
- Result: the agent learns the state-dependent mapping (final mean reward > +0.5). **Passes.**

---

## 4. Action-contract mismatch that caused the original false-positive bandit result

The original bandit test incorrectly mapped `AgentAction.action_type == "move"` to a default action instead of interpreting its `dx`/`dy` parameters. Because of this, the agent's selected action was not actually the action whose reward was being measured, so the "learning" signal was decoupled from the agent's behavior — a false positive.

Correction (in `tests/research/test_torch_rl_sanity.py:42-45`):

```python
def action_to_index(action: AgentAction) -> int:
    if action.action_type == "move" and action.parameters.get("dx", 0) > 0:
        return 1
    return 0
```

The reward is now assigned based on the actual `dx` parameter the controller selected, so the credit assignment correctly reflects the agent's behavior. The two bandit tests pass after this correction and are not false positives.

---

## 5. Imitation learning result — failed subsequent verification

An earlier reported imitation-learning result of approximately +0.69 was contradicted by later experiments. It is **NOT treated as independently validated** and is not cited as evidence of successful imitation or reward-driven control.

---

## 6. Reward-only delayed-cue learning — NOT reliably demonstrated

`src/universe_engine/research/diagnostics/trainable_memory.py` (`train_and_evaluate`, `GRUActorCriticController`, `FeedForwardController`) and the Sprint 13 memory tasks (`DelayedCueTask`, `OccludedTargetTask`).

Trainable controllers (GRU actor-critic with eligibility traces, and a feedforward baseline) were trained on delayed_cue and occluded_target:

- Reward is only delivered at the end of the response phase (claim hit / miss).
- Credit must be assigned across the cue->delay->response gap.
- Held-out evaluation seeds are disjoint from training seeds; evaluation controllers are cloned from final trained weights.

**Result: reliable reward-driven use of memory on held-out delayed-cue tasks was NOT demonstrated.** Trainable agents fail to solve delayed_cue and occluded_target at success thresholds. The supervised GRU retains the cue (100% accuracy), but converting retained memory into reward-maximizing control via policy gradient alone remains an open, documented negative result.

This is reported honestly and is NOT relabeled as the programmed working-memory oracle's success (the oracle is Category 1 programmed intelligence; trainable agents are Category 2).

---

## 7. Oracle control vs supervised learning vs imitation learning vs reinforcement learning

| Mechanism | Memory basis | Learning signal | Status on delayed-cue memory control |
|-----------|-------------|-----------------|--------------------------------------|
| Programmed working-memory oracle | Explicitly stores cue/target coords | None (hand-coded) | ✅ Solves delayed_cue and occluded_target (Category 1, validated) |
| Supervised GRU (cue retention) | GRU hidden state | Supervised cue-prediction (auxiliary) | ✅ Retains cue at 100% across delays 1–20; not a control policy |
| Imitation learning | — | Demonstrated expert trajectories | ❌ Reported +0.69 result contradicted, not validated |
| REINFORCE bandits | None (memory-free) | Reward only | ✅ Learns two-action and contextual bandits |
| REINFORCE / actor-critic delayed-cue | GRU hidden state | Reward only (delayed) | ❌ Not reliably learned on held-out delayed-cue / occluded_target |

**Conclusion:** Supervised recurrent memory has been demonstrated. Basic reward-driven action learning has been demonstrated in bandits. Reliable reward-driven use of memory on held-out delayed-cue tasks has **NOT** yet been demonstrated.

Do not claim the RL algorithm is universally correct merely because two simple bandits pass.

---

## 8. Experiments executed — seeds and budget limitations

- GRU binary-cue retention: delays 1, 3, 5, 10, 20; 100% accuracy. Supervised auxiliary task; training budget limited (per `trainable_memory.py` defaults).
- Two-action bandit: 3000 episodes, seed=1, lr=5e-2 (`test_torch_rl_sanity.py`).
- Contextual bandit: 6000 episodes, seed=2, lr=1e-1 (`test_torch_rl_sanity.py`).
- Actor parameter change: 2000 episodes, seed=3 (`test_torch_rl_sanity.py`).
- Terminal non-bootstrapping: single-step (`test_torch_rl_sanity.py`).
- Trainable delayed-cue / occluded-target: disjoint train/eval seeds; evaluation controllers cloned from trained weights. Budgets per `_run_episode` / `train_and_evaluate` defaults in `trainable_memory.py`.

All runs are small-scale diagnostics, not production-scale training. No 50,000-episode run was started or is planned in this pass.

---

## 9. Regression tests

`tests/research/test_torch_rl_sanity.py` (130 lines, added in commit d6c0f93):

- `test_bandit_learns_positive_action`
- `test_contextual_bandit_learns_state_dependent_policy`
- `test_actor_parameters_change_during_training`
- `test_terminal_episode_does_not_bootstrap`

All 4 pass (torch is an optional research dependency; tests skip when torch is not installed). Combined with `test_torch_memory.py`, the focused RL/memory suite runs 14 passed.

Full pytest suite: **542 passed, 1 skipped, 0 failures, 0 errors, 0 timeouts** (~2m53s).

---

## 10. Honest status

| Objective | Status |
|-----------|--------|
| Supervised GRU cue retention at delays 1–20 | ✅ Verified (100% accuracy) |
| Corrected two-action bandit (REINFORCE) | ✅ Passes |
| Corrected contextual bandit (REINFORCE) | ✅ Passes |
| Action-contract mismatch (false positive) identified and corrected | ✅ Documented |
| Imitation +0.69 result validated | ❌ Not validated (contradicted) |
| Reward-only delayed-cue control | ❌ Not reliably demonstrated (negative result) |
| RL universally correct because bandits pass | ❌ Not claimed |
| Full pytest suite | ✅ 542 passed, 1 skipped |

**Bottom line:** Supervised recurrent memory has been demonstrated. Basic reward-driven action learning has been demonstrated in bandits. Reliable reward-driven use of memory on held-out delayed-cue tasks has **NOT** yet been demonstrated. The negative result is documented honestly and deferred to later research (PASS 2+). The code revision used for these results is `d6c0f93af099a7e8a0ea300ff55077a8f2e805af`.
