"""Memory-sensitive task experiments (Sprint 11 Phase 3-4).

Protocol:
1. Train each controller for n_train_episodes on sequential training seeds
   (controller persists and learns across episodes).
2. Evaluate each controller on held-out seeds with fresh controllers.
3. Report mean eval reward and per-seed results with uncertainty.

Agents:
- Random: uniform random (lower bound)
- Reactive: feedforward on current observation only
- StackedObs: reactive with previous observations stacked (finite memory)
- Recurrent: GRU-like hidden state trained with reward-based update

Tasks:
- Delayed cue (memory required)
- Occluded target (spatial memory required)
- Changed dynamics (change detection / adaptation required)

GridWorld control (fully observable) reproduces Program D context.

No claim of consciousness; measures whether memory/prediction provide
measurable task advantages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Callable
import numpy as np

from ..contracts import AgentAction
from .memory_tasks import (
    DelayedCueTask, DelayedCueConfig,
    OccludedTargetTask, OccludedTargetConfig,
    ChangedDynamicsTask, ChangedDynamicsConfig,
    RandomController, ReactiveController, StackedObsController,
    RecurrentController, run_episode_on_task,
)
from ..environment.gridworld import GridWorld, GridWorldConfig


# ---------------------------------------------------------------------------
# Task-specific action wrappers
# ---------------------------------------------------------------------------

class TaskReactive(ReactiveController):
    def select_action(self, observation):
        logits = self.W @ self._obs_vector(observation)
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        params = {}
        if at == "move":
            params = {"dx": int(self.rng.integers(-1, 2)), "dy": int(self.rng.integers(-1, 2))}
        elif at == "wait":
            params = {"direction": 0}
        return AgentAction(at, params, timestamp=observation.time)


class TaskStacked(StackedObsController):
    def select_action(self, observation):
        logits = self.W @ self._obs_vector(observation)
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        params = {}
        if at == "move":
            params = {"dx": int(self.rng.integers(-1, 2)), "dy": int(self.rng.integers(-1, 2))}
        elif at == "wait":
            params = {"direction": 0}
        return AgentAction(at, params, timestamp=observation.time)


class TaskRecurrent(RecurrentController):
    def select_action(self, observation):
        self.actions_taken += 1
        obs_vec = self._obs_vector(observation)
        self.hidden = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec)
        logits = self.W_out @ self.hidden
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        params = {}
        if at == "move":
            params = {"dx": int(self.rng.integers(-1, 2)), "dy": int(self.rng.integers(-1, 2))}
        elif at == "wait":
            params = {"direction": 0}
        return AgentAction(at, params, timestamp=observation.time)


FACTORIES = {
    "random": lambda aid, sd: RandomController(aid, sd),
    "reactive": lambda aid, sd: TaskReactive(aid, sd),
    "stacked_obs": lambda aid, sd: TaskStacked(aid, sd, stack=3),
    "recurrent": lambda aid, sd: TaskRecurrent(aid, sd, hidden_dim=8),
}

# ---------------------------------------------------------------------------
# GridWorld control (fully observable, Program D)
# ---------------------------------------------------------------------------

class GWReactive:
    def __init__(self, agent_id: str, seed: int = 0):
        from ..environment.gridworld import ReactiveController as _GWReactive
        self._inner = _GWReactive(agent_id, seed)
        self.agent_id = agent_id
        self.controller_type = "reactive"

    def select_action(self, observation):
        return self._inner.select_action(observation)

    def learn(self, observation, action, reward, next_observation):
        self._inner.learn(observation, action, reward, next_observation)


class GWRecurrent:
    def __init__(self, agent_id: str, seed: int = 0):
        from ..environment.gridworld import RecurrentWorldModelController as _GWRecurrent
        self._inner = _GWRecurrent(agent_id, seed, lr=0.05)
        self.agent_id = agent_id
        self.controller_type = "recurrent_world_model"

    def select_action(self, observation):
        return self._inner.select_action(observation)

    def learn(self, observation, action, reward, next_observation):
        self._inner.learn(observation, action, reward, next_observation)


def run_gridworld_episode(env: GridWorld, controller, agent_id: str,
                          n_steps: int) -> Dict[str, float]:
    obs = env.observe(agent_id)
    total_reward = 0.0
    for step in range(n_steps):
        action = controller.select_action(obs)
        result = env.act(agent_id, action)
        next_obs = env.observe(agent_id)
        controller.learn(obs, action, float(result.get("reward", 0.0)), next_obs)
        total_reward += float(result.get("reward", 0.0))
        obs = next_obs
        env.step(0.1)
        if not env._agent_states[agent_id].alive:
            break
    return {"total_reward": float(total_reward)}


@dataclass
class BenchmarkResult:
    task: str
    controller: str
    train_seed: int
    eval_seeds: List[int]
    train_rewards: List[float]
    eval_rewards: List[float]
    mean_eval: float
    std_eval: float

    def to_dict(self) -> dict:
        return {
            "task": self.task, "controller": self.controller,
            "train_seed": self.train_seed, "eval_seeds": self.eval_seeds,
            "train_rewards": self.train_rewards,
            "eval_rewards": self.eval_rewards,
            "mean_eval": self.mean_eval, "std_eval": self.std_eval,
        }


def train_controller(make_env, factory, train_seeds: List[int],
                     n_steps: int) -> object:
    """Train one controller across sequential training environments."""
    ctrl = factory("train", train_seeds[0])
    for seed in train_seeds:
        env = make_env(seed=seed)
        run_episode_on_task(env, ctrl, "train", n_steps)
    return ctrl


def evaluate_controller(make_env, factory, ctrl, eval_seeds: List[int],
                        n_eval_episodes: int, n_steps: int) -> List[float]:
    """Evaluate a trained controller on held-out seeds (fresh envs)."""
    rewards = []
    for seed in eval_seeds:
        for ep in range(n_eval_episodes):
            env = make_env(seed=seed + ep)
            # fresh controller with same trained weights where possible
            eval_ctrl = factory(f"eval-{seed}-{ep}", seed + ep)
            # copy learned parameters if available
            if hasattr(ctrl, "W") and hasattr(eval_ctrl, "W"):
                eval_ctrl.W = ctrl.W.copy()
            if hasattr(ctrl, "hidden") and hasattr(eval_ctrl, "hidden"):
                eval_ctrl.hidden = ctrl.hidden.copy()
            if hasattr(ctrl, "W_out") and hasattr(eval_ctrl, "W_out"):
                eval_ctrl.W_out = ctrl.W_out.copy()
            if hasattr(ctrl, "W_x") and hasattr(eval_ctrl, "W_x"):
                eval_ctrl.W_x = ctrl.W_x.copy()
            if hasattr(ctrl, "W_h") and hasattr(eval_ctrl, "W_h"):
                eval_ctrl.W_h = ctrl.W_h.copy()
            r = run_episode_on_task(env, eval_ctrl, f"eval-{seed}-{ep}", n_steps)
            rewards.append(r["total_reward"])
    return rewards


def train_and_evaluate(make_env, factory, train_seeds: List[int],
                       eval_seeds: List[int], n_steps: int,
                       n_train_episodes: int, n_eval_episodes: int) -> BenchmarkResult:
    # Train ONE controller persistently across many training episodes.
    ctrl = factory("train", train_seeds[0])
    train_rewards = []
    for ep in range(n_train_episodes):
        seed = train_seeds[ep % len(train_seeds)] + ep
        env = make_env(seed=seed)
        r = run_episode_on_task(env, ctrl, "train", n_steps)
        train_rewards.append(r["total_reward"])

    # Evaluate the trained controller on held-out seeds (fresh envs).
    eval_rewards = []
    for seed in eval_seeds:
        for ep in range(n_eval_episodes):
            env = make_env(seed=seed + ep)
            eval_ctrl = factory(f"eval-{seed}-{ep}", seed + ep)
            # transfer learned parameters
            if hasattr(ctrl, "W") and hasattr(eval_ctrl, "W"):
                eval_ctrl.W = ctrl.W.copy()
            if hasattr(ctrl, "hidden") and hasattr(eval_ctrl, "hidden"):
                eval_ctrl.hidden = ctrl.hidden.copy()
            if hasattr(ctrl, "W_out") and hasattr(eval_ctrl, "W_out"):
                eval_ctrl.W_out = ctrl.W_out.copy()
            if hasattr(ctrl, "W_x") and hasattr(eval_ctrl, "W_x"):
                eval_ctrl.W_x = ctrl.W_x.copy()
            if hasattr(ctrl, "W_h") and hasattr(eval_ctrl, "W_h"):
                eval_ctrl.W_h = ctrl.W_h.copy()
            r = run_episode_on_task(env, eval_ctrl, f"eval-{seed}-{ep}", n_steps)
            eval_rewards.append(r["total_reward"])

    return BenchmarkResult(
        task="", controller="", train_seed=train_seeds[0], eval_seeds=eval_seeds,
        train_rewards=train_rewards, eval_rewards=eval_rewards,
        mean_eval=float(np.mean(eval_rewards)) if eval_rewards else 0.0,
        std_eval=float(np.std(eval_rewards)) if eval_rewards else 0.0,
    )


def run_memory_benchmark(seeds: List[int] = None,
                         n_train_seeds: int = 3,
                         n_train_episodes: int = 200,
                         n_eval_episodes: int = 5,
                         n_steps: int = 25) -> List[BenchmarkResult]:
    seeds = seeds or [1, 42, 99]
    train_seeds = seeds[:n_train_seeds]
    eval_seeds = seeds
    tasks = {
        "delayed_cue": lambda seed: DelayedCueTask(DelayedCueConfig(), seed=seed),
        "occluded_target": lambda seed: OccludedTargetTask(OccludedTargetConfig(), seed=seed),
        "changed_dynamics": lambda seed: ChangedDynamicsTask(ChangedDynamicsConfig(), seed=seed),
    }

    results: List[BenchmarkResult] = []
    for task_name, make_env in tasks.items():
        for ctrl_name, factory in FACTORIES.items():
            br = train_and_evaluate(
                make_env, factory, train_seeds, eval_seeds,
                n_steps=n_steps, n_train_episodes=n_train_episodes,
                n_eval_episodes=n_eval_episodes,
            )
            br.task = task_name
            br.controller = ctrl_name
            results.append(br)
    return results


def run_gridworld_control(seeds: List[int] = None,
                          n_episodes: int = 20, n_steps: int = 100) -> Dict:
    seeds = seeds or [1, 42, 99]
    out = {}
    for seed in seeds:
        for cname, ctor in (("reactive", GWReactive), ("recurrent", GWRecurrent)):
            rewards = []
            for ep in range(n_episodes):
                env = GridWorld(GridWorldConfig(seed=seed + ep))
                env.register_agent("a1", (env.cfg.width // 2, env.cfg.height // 2),
                                   initial_energy=5.0)
                ctrl = ctor("a1", seed=seed + ep)
                r = run_gridworld_episode(env, ctrl, "a1", n_steps)
                rewards.append(r["total_reward"])
            out[f"seed={seed} {cname}"] = float(np.mean(rewards))
    return out


if __name__ == "__main__":
    res = run_memory_benchmark(n_train_seeds=3, n_eval_episodes=2, n_steps=25)
    for r in res:
        print(f"{r.task:18s} {r.controller:14s} train_seed={r.train_seed} "
              f"mean_eval={r.mean_eval:.3f} std={r.std_eval:.3f} "
              f"evals={[round(v,2) for v in r.eval_rewards]}")

    print()
    print("=== GridWorld control (reproduces Program D) ===")
    gw = run_gridworld_control()
    for k, v in gw.items():
        print(f"  {k}: {v:.3f}")
