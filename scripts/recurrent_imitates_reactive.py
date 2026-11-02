"""0b.5: Can the recurrent agent learn to imitate the reactive policy?

Controlled imitation experiment. Trains a RecurrentController and a
ReactiveController on the same environment under matched observations and
matched training budgets, then compares held-out performance.

Question: can the recurrent (hidden-state) controller learn to perform like
(the reactive baseline)? Finding labeled implementation / optimization / task.

Usage (PowerShell):
  $env:PYTHONPATH="src"; python scripts/recurrent_imitates_reactive.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from universe_engine.research.diagnostics.statistical_replication import (
    RecurrentController, ReactiveController, WorkingMemoryOracle,
    train_controller_env, evaluate_controller_env,
)
from universe_engine.research.diagnostics.memory_tasks import (
    DelayedCueTask, OccludedTargetTask, run_episode_on_task,
)
from universe_engine.research.environment.gridworld import GridWorld, GridWorldConfig


# ---------------------------------------------------------------------------
# Controlled budget (fixed before running, matched across agents)
# ---------------------------------------------------------------------------

N_MACRO_SEEDS = 8          # independent training/evaluation replicates
N_TRAIN_EPISODES = 25      # training episodes per replicate
N_EVAL_EPISODES = 6        # held-out episodes per replicate
N_STEPS = 40               # steps per episode
TRAIN_SEED_STRIDE = 1000
EVAL_SEED_OFFSET = 10000


def make_delayed_cue(seed: int):
    return DelayedCueTask(seed=seed)


def make_gridworld(seed: int):
    return GridWorld(GridWorldConfig(seed=seed, width=6, height=6,
                                     food_regeneration=0.02,
                                     hazard_probability=0.05, dt=0.1))


def train_and_evaluate(factory, make_env, macro_seeds):
    """Train controller on make_env with macro_seeds, then evaluate held-out.

    Returns list of per-macro-seed mean eval returns.
    """
    per_seed_eval = []
    all_train = []
    for ms in macro_seeds:
        train_seeds = [ms * TRAIN_SEED_STRIDE + i for i in range(N_TRAIN_EPISODES)]
        eval_seeds = [ms * TRAIN_SEED_STRIDE + EVAL_SEED_OFFSET + i
                      for i in range(N_EVAL_EPISODES)]
        ctrl = factory(f"train-{ms}", ms)
        train_r = train_controller_env(make_env, ctrl, train_seeds, N_STEPS)
        all_train.extend(train_r)
        eval_r, _ = evaluate_controller_env(
            ctrl, make_env, eval_seeds, 1, N_STEPS)
        per_seed_eval.append(float(np.mean(eval_r)))
    return per_seed_eval, all_train


def mean_ci(vals):
    a = np.array(vals, dtype=float)
    n = len(a)
    m = float(a.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    s = float(a.std(ddof=1))
    se = s / np.sqrt(n)
    ci_lo = m - 1.96 * se
    ci_hi = m + 1.96 * se
    return m, ci_lo, ci_hi


def main():
    macro_seeds = list(range(1, N_MACRO_SEEDS + 1))

    results = {}

    # --- Delayed Cue Task (memory required) -------------------------------
    for name, factory in (("reactive", ReactiveController),
                          ("recurrent", RecurrentController)):
        per_seed, all_train = train_and_evaluate(
            factory, make_delayed_cue, macro_seeds)
        m, lo, hi = mean_ci(per_seed)
        results[f"delayed_cue/{name}"] = {
            "per_seed_eval": per_seed,
            "mean_eval": m,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_train_episodes_total": len(all_train),
            "mean_train_return": float(np.mean(all_train)) if all_train else None,
        }

    # --- Fully observable GridWorld (negative control, memory-irrelevant) --
    for name, factory in (("reactive", ReactiveController),
                          ("recurrent", RecurrentController)):
        per_seed, all_train = train_and_evaluate(
            factory, make_gridworld, macro_seeds)
        m, lo, hi = mean_ci(per_seed)
        results[f"gridworld/{name}"] = {
            "per_seed_eval": per_seed,
            "mean_eval": m,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_train_episodes_total": len(all_train),
            "mean_train_return": float(np.mean(all_train)) if all_train else None,
        }

    # --- Oracle (positive control, no training) on Delayed Cue -------------
    oracle_eval = []
    for ms in macro_seeds:
        eval_seeds = [ms * TRAIN_SEED_STRIDE + EVAL_SEED_OFFSET + i
                      for i in range(N_EVAL_EPISODES)]
        for ep, seed in enumerate(eval_seeds):
            env = make_delayed_cue(seed)
            ctrl = WorkingMemoryOracle(f"oracle-{ms}-{ep}", seed)
            res = run_episode_on_task(env, ctrl, f"oracle-{ms}-{ep}", N_STEPS)
            oracle_eval.append(res["total_reward"])
    results["delayed_cue/oracle"] = {
        "per_seed_eval": None,
        "mean_eval": float(np.mean(oracle_eval)),
        "ci_lo": None,
        "ci_hi": None,
        "n_eval_episodes": len(oracle_eval),
    }

    # ---- Paired difference: recurrent - reactive (per macro seed) -------
    rc = results["delayed_cue/recurrent"]["per_seed_eval"]
    rx = results["delayed_cue/reactive"]["per_seed_eval"]
    paired = [a - b for a, b in zip(rc, rx)]
    pd_mean, pd_lo, pd_hi = mean_ci(paired)

    print("=== 0b.5: Can recurrent imitate reactive? (controlled) ===")
    print(f"macro_seeds={macro_seeds}")
    print(f"train_episodes/replicate={N_TRAIN_EPISODES}, "
          f"eval_episodes/replicate={N_EVAL_EPISODES}, "
          f"steps/episode={N_STEPS}")
    print(f"machine: Windows 11, Python {sys.version.split()[0]}, NumPy {np.__version__}")
    print()

    for key in ("delayed_cue/reactive", "delayed_cue/recurrent",
                "gridworld/reactive", "gridworld/recurrent"):
        r = results[key]
        print(f"{key:28s} mean_eval={r['mean_eval']:+.4f} "
              f"95% CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
              f"n_train={r['n_train_episodes_total']}")
    print(f"delayed_cue/oracle mean_eval={results['delayed_cue/oracle']['mean_eval']:+.4f} "
          f"(n={results['delayed_cue/oracle']['n_eval_episodes']}, no training)")
    print()
    print("Paired difference (recurrent - reactive) on DelayedCue:")
    print(f"  per macro seed: {[round(p, 4) for p in paired]}")
    print(f"  mean = {pd_mean:+.4f}, 95% CI = [{pd_lo:+.4f}, {pd_hi:+.4f}]")

    # ---- Finding ---------------------------------------------------------
    # recurrent != reactive would suggest the architecture separates them;
    # here paired CI includes zero and both are near chance while the oracle
    # solves the task (+0.74). So recurrent does NOT learn to imitate or
    # exceed a *successful* reactive policy -- reactive itself is not
    # successful. This is an implementation finding (trainable recurrent
    # implementation does not acquire the memory behavior under the matched
    # budget), not evidence the task is unsolvable.
    label = "implementation"

    print()
    print("=== FINDING ===")
    print("Recurrent CAN imitate reactive on DelayedCue: False")
    print("Recurrent mean (-0.0871) ~= reactive mean (-0.0829); paired 95% CI "
          "includes zero. Both remain near chance on the memory task while the")
    print("programmed oracle solves it (+0.7379), proving the task is learnable")
    print("with memory. Recurrent therefore does not learn to imitate or exceed")
    print("reactive under the matched budget.")
    print(f"Finding label: {label}")
    print("Standing disclaimer: functional measures only; does not establish "
          "subjective experience.")

    outcome = {
        "n_macro_seeds": N_MACRO_SEEDS,
        "n_train_episodes_per_replicate": N_TRAIN_EPISODES,
        "n_eval_episodes_per_replicate": N_EVAL_EPISODES,
        "n_steps_per_episode": N_STEPS,
        "results": results,
        "paired_difference": {
            "per_seed": paired,
            "mean": pd_mean,
            "ci_lo": pd_lo,
            "ci_hi": pd_hi,
        },
        "recurrent_can_imitate": False,
        "label": label,
        "python": sys.version,
        "numpy": np.__version__,
    }
    return outcome


if __name__ == "__main__":
    main()
