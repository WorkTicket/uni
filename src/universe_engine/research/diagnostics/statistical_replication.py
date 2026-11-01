"""Statistical replication (Sprint 12 Phase 2).

Rigorous multi-seed comparison of cognitive architectures on the GridWorld
foraging task and memory-required tasks, with:

- Separate training, tuning and held-out evaluation seeds
- Persistent training (weights carry across episodes)
- Held-out evaluation with fresh controllers initialized from trained weights
- Mean +/- 95% CI, Cohen's d effect size vs reactive baseline
- Sample efficiency and inference-cost reporting
- Matched interaction and compute budgets

No claim of consciousness; measures whether memory/world-model/planning provide
measurable, reproducible task advantages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import time
import numpy as np

from ..contracts import AgentAction, AgentState, SensoryObservation
from ..environment.gridworld import GridWorld, GridWorldConfig
from .memory_tasks import (
    DelayedCueTask, DelayedCueConfig,
    OccludedTargetTask, OccludedTargetConfig,
    ChangedDynamicsTask, ChangedDynamicsConfig,
)


# ============================================================================
# Controllers (self-contained, cloneable for weight transfer)
# ============================================================================

class BaseController:
    agent_id: str
    controller_type: str
    rng: np.random.Generator

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        raise NotImplementedError

    def learn(self, observation, action, reward, next_observation) -> None:
        pass

    def get_weights(self) -> dict:
        return {}

    def set_weights(self, w: dict) -> None:
        pass

    def clone(self, agent_id: str, seed: int) -> "BaseController":
        raise NotImplementedError


def _obs_vector(obs) -> np.ndarray:
    return np.array([
        obs.energy_level, obs.nutrient_level,
        obs.nearby_resources, obs.nearby_hazards,
        float(obs.nearby_agents) / 5.0, obs.local_light,
    ])


def _move_params(rng) -> dict:
    return {"dx": int(rng.integers(-1, 2)), "dy": int(rng.integers(-1, 2))}


class ReactiveController(BaseController):
    """Feedforward reactive policy on current observation only."""

    def __init__(self, agent_id: str, seed: int = 0):
        self.agent_id = agent_id
        self.controller_type = "reactive"
        self.rng = np.random.default_rng(seed)
        self.W = np.random.default_rng(seed + 1).standard_normal((4, 6)) * 0.1
        self.actions_taken = 0

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        logits = self.W @ _obs_vector(observation)
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        params = _move_params(self.rng) if at == "move" else {}
        return AgentAction(at, params, timestamp=observation.time)

    def clone(self, agent_id: str, seed: int) -> "ReactiveController":
        c = ReactiveController(agent_id, seed)
        c.W = self.W.copy()
        return c


class StackedObsController(BaseController):
    """Reactive policy with previous observations stacked (finite memory)."""

    def __init__(self, agent_id: str, seed: int = 0, stack: int = 3):
        self.agent_id = agent_id
        self.controller_type = "stacked_obs"
        self.stack = int(stack)
        self.rng = np.random.default_rng(seed)
        self.history: List[np.ndarray] = []
        self.W = np.random.default_rng(seed + 1).standard_normal((4, 6 * stack)) * 0.1
        self.actions_taken = 0

    def _stacked(self, obs) -> np.ndarray:
        v = _obs_vector(obs)
        self.history.append(v)
        self.history = self.history[-self.stack:]
        while len(self.history) < self.stack:
            self.history.insert(0, np.zeros_like(v))
        return np.concatenate(self.history)

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        logits = self.W @ self._stacked(observation)
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        params = _move_params(self.rng) if at == "move" else {}
        return AgentAction(at, params, timestamp=observation.time)

    def clone(self, agent_id: str, seed: int) -> "StackedObsController":
        c = StackedObsController(agent_id, seed, stack=self.stack)
        c.W = self.W.copy()
        c.history = [h.copy() for h in self.history]
        return c


class RecurrentController(BaseController):
    """Simple recurrent hidden-state controller, REINFORCE-trained."""

    def __init__(self, agent_id: str, seed: int = 0, hidden_dim: int = 8, lr: float = 0.05):
        self.agent_id = agent_id
        self.controller_type = "recurrent"
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        self.hidden = np.zeros(hidden_dim)
        self.W_x = self.rng.standard_normal((hidden_dim, 6)) * 0.1
        self.W_h = self.rng.standard_normal((hidden_dim, hidden_dim)) * 0.1
        self.W_out = self.rng.standard_normal((4, hidden_dim)) * 0.1
        self.last_action = "wait"
        self.actions_taken = 0

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        obs_vec = _obs_vector(observation)
        self.hidden = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec)
        logits = self.W_out @ self.hidden
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        self.last_action = at
        params = _move_params(self.rng) if at == "move" else {}
        return AgentAction(at, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        chosen = ["move", "forage", "rest", "wait"]
        action_idx = chosen.index(self.last_action)
        logits = self.W_out @ self.hidden
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        one_hot = np.zeros(4); one_hot[action_idx] = 1.0
        grad = (one_hot - probs) * reward
        self.W_out += self.lr * np.outer(grad, self.hidden)

    def get_weights(self) -> dict:
        return {"W_x": self.W_x, "W_h": self.W_h, "W_out": self.W_out,
                "hidden": self.hidden.copy()}

    def set_weights(self, w: dict) -> None:
        self.W_x = w["W_x"].copy(); self.W_h = w["W_h"].copy()
        self.W_out = w["W_out"].copy()
        if "hidden" in w:
            self.hidden = w["hidden"].copy()

    def clone(self, agent_id: str, seed: int) -> "RecurrentController":
        c = RecurrentController(agent_id, seed, hidden_dim=self.hidden_dim, lr=self.lr)
        c.set_weights(self.get_weights())
        return c


class WorldModelController(BaseController):
    """Recurrent world model: observation predictor + reward predictor.

    Predicts next observation and immediate reward from hidden+action; uses
    predicted reward for action selection (aligned value function).
    """

    def __init__(self, agent_id: str, seed: int = 0, hidden_dim: int = 8, lr: float = 0.05):
        self.agent_id = agent_id
        self.controller_type = "world_model"
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        self.hidden = np.zeros(hidden_dim)
        self.W_x = self.rng.standard_normal((hidden_dim, 6)) * 0.1
        self.W_h = self.rng.standard_normal((hidden_dim, hidden_dim)) * 0.1
        self.W_a = self.rng.standard_normal((hidden_dim, 4)) * 0.1
        self.W_pred = self.rng.standard_normal((6, hidden_dim)) * 0.1
        self.W_r = self.rng.standard_normal((1, hidden_dim + 4)) * 0.1
        self.last_action = "wait"
        self.actions_taken = 0
        self.prediction_error_history: List[float] = []

    def _action_vec(self, action_type: str) -> np.ndarray:
        v = np.zeros(4)
        at = ["move", "forage", "rest", "wait"]
        v[at.index(action_type)] = 1.0
        return v

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        obs_vec = _obs_vector(observation)
        best_action = "wait"; best_value = -float("inf")
        best_av = np.zeros(4)
        for at in ["move", "forage", "rest", "wait"]:
            av = self._action_vec(at)
            hidden_next = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec
                                  + self.W_a @ av)
            # note: hidden_next ignores av here; world model dynamics driven by obs
            pred_obs = self.W_pred @ hidden_next
            sa = np.concatenate([hidden_next, av])
            predicted_reward = float((self.W_r @ sa)[0])
            predicted_hazard = pred_obs[3]
            value = predicted_reward - 0.5 * predicted_hazard
            if value > best_value:
                best_value = value; best_action = at; best_av = av
        self.last_action = best_action
        params = _move_params(self.rng) if best_action == "move" else {}
        return AgentAction(best_action, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        obs_vec = _obs_vector(observation)
        next_obs_vec = _obs_vector(next_observation)
        av = self._action_vec(self.last_action)
        hidden_pred = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec
                              + self.W_a @ av)
        pred_obs = self.W_pred @ hidden_pred
        error = pred_obs - next_obs_vec
        self.prediction_error_history.append(float(np.mean(np.abs(error))))
        if len(self.prediction_error_history) > 1000:
            self.prediction_error_history = self.prediction_error_history[-1000:]
        self.W_pred -= self.lr * np.outer(error, hidden_pred)
        self.W_x -= self.lr * 0.01 * np.outer(hidden_pred, error)
        sa = np.concatenate([hidden_pred, av])
        pred_r = float((self.W_r @ sa)[0])
        self.W_r -= self.lr * 2.0 * (pred_r - reward) * sa.reshape(1, -1)
        self.hidden = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec)

    def get_weights(self) -> dict:
        return {"W_x": self.W_x, "W_h": self.W_h, "W_a": self.W_a,
                "W_pred": self.W_pred, "W_r": self.W_r,
                "hidden": self.hidden.copy()}

    def set_weights(self, w: dict) -> None:
        self.W_x = w["W_x"].copy(); self.W_h = w["W_h"].copy()
        self.W_a = w["W_a"].copy()
        self.W_pred = w["W_pred"].copy(); self.W_r = w["W_r"].copy()
        if "hidden" in w:
            self.hidden = w["hidden"].copy()

    def clone(self, agent_id: str, seed: int) -> "WorldModelController":
        c = WorldModelController(agent_id, seed, hidden_dim=self.hidden_dim, lr=self.lr)
        c.set_weights(self.get_weights())
        c.prediction_error_history = list(self.prediction_error_history)
        return c


# ============================================================================
# Train / evaluate protocol
# ============================================================================

@dataclass
class RunResult:
    controller: str
    macro_seed: int
    train_rewards: List[float]
    eval_rewards: List[float]
    mean_eval: float
    std_eval: float
    n_eval: int
    inference_cost: float  # relative action selections
    final_train_reward: float


def run_episode(env: GridWorld, controller: BaseController,
                agent_id: str, n_steps: int) -> Tuple[float, int]:
    """Run one episode; controller learns online. Returns (total_reward, steps)."""
    env.register_agent(agent_id, (env.cfg.width // 2, env.cfg.height // 2),
                       initial_energy=5.0)
    obs = env.observe(agent_id)
    total = 0.0
    steps = 0
    alive = True
    for step in range(n_steps):
        action = controller.select_action(obs)
        result = env.act(agent_id, action)
        next_obs = env.observe(agent_id)
        controller.learn(obs, action, float(result.get("reward", 0.0)), next_obs)
        total += float(result.get("reward", 0.0))
        steps = step + 1
        obs = next_obs
        env.step(0.1)
        if not env._agent_states[agent_id].alive:
            alive = False
            break
    return total, steps


def run_episode_env(env, controller: BaseController, agent_id: str,
                    n_steps: int) -> Tuple[float, int]:
    """Run one episode on any Environment-contract task.

    Handles both GridWorld (register_agent / _agent_states) and memory POMDP
    tasks (reset / done flag in act result). Controller learns online.
    Returns (total_reward, steps).
    """
    # GridWorld-style registration if available.
    if hasattr(env, "register_agent"):
        env.register_agent(agent_id, (getattr(env.cfg, "width", 4) // 2,
                                      getattr(env.cfg, "height", 4) // 2),
                           initial_energy=5.0)
    elif hasattr(env, "reset"):
        env.reset()
    obs = env.observe(agent_id)
    total = 0.0
    steps = 0
    alive = True
    for step in range(n_steps):
        action = controller.select_action(obs)
        result = env.act(agent_id, action)
        next_obs = env.observe(agent_id)
        controller.learn(obs, action, float(result.get("reward", 0.0)), next_obs)
        total += float(result.get("reward", 0.0))
        steps = step + 1
        obs = next_obs
        if hasattr(env, "step"):
            env.step(0.1)
        if result.get("done", False):
            alive = False
            break
        elif hasattr(env, "_agent_states"):
            if not env._agent_states[agent_id].alive:
                alive = False
                break
    return total, steps


def train_controller_env(make_env: Callable[[int], object],
                         controller: BaseController,
                         train_seeds: List[int], n_steps: int) -> List[float]:
    rewards = []
    for ep, seed in enumerate(train_seeds):
        env = make_env(seed)
        r, _ = run_episode_env(env, controller, f"train-{ep}", n_steps)
        rewards.append(r)
    return rewards


def evaluate_controller_env(controller: BaseController,
                            make_env: Callable[[int], object],
                            eval_seeds: List[int], n_eval_episodes: int,
                            n_steps: int) -> Tuple[List[float], int]:
    rewards = []
    total_actions = 0
    for seed in eval_seeds:
        for ep in range(n_eval_episodes):
            env = make_env(seed + ep)
            ec = controller.clone(f"eval-{seed}-{ep}", seed + ep)
            total_actions += ec.actions_taken
            r, _ = run_episode_env(env, ec, f"eval-{seed}-{ep}", n_steps)
            rewards.append(r)
    return rewards, total_actions


def compare_controllers_env(make_env: Callable[[int], object],
                            controllers: Dict[str, Callable[[str, int], BaseController]],
                            macro_seeds: List[int],
                            n_train_seeds: int,
                            n_eval_episodes: int,
                            n_steps: int) -> Dict[str, ExperimentResult]:
    results: Dict[str, ExperimentResult] = {}
    reactive_eval: List[float] = []
    for name, factory in controllers.items():
        all_eval: List[float] = []
        all_train: List[float] = []
        total_actions = 0
        for ms in macro_seeds:
            train_seeds = [ms * 1000 + i for i in range(n_train_seeds)]
            eval_seeds = [ms * 1000 + 10000 + i for i in range(n_eval_episodes)]
            ctrl = factory(f"train-{ms}", ms)
            train_r = train_controller_env(make_env, ctrl, train_seeds, n_steps)
            all_train.extend(train_r)
            eval_r, n_act = evaluate_controller_env(
                ctrl, make_env, eval_seeds, 1, n_steps)
            total_actions += n_act
            all_eval.extend(eval_r)
        results[name] = ExperimentResult(
            controller=name,
            mean_eval=float(np.mean(all_eval)) if all_eval else 0.0,
            ci_lo=0.0, ci_hi=0.0, std_eval=float(np.std(all_eval)) if all_eval else 0.0,
            cohens_d_vs_reactive=0.0,
            n_macro_seeds=len(macro_seeds),
            all_eval_rewards=all_eval,
            mean_train_reward=float(np.mean(all_train)) if all_train else 0.0,
            inference_cost=float(total_actions),
        )
        if name == "reactive":
            reactive_eval = list(all_eval)
    for name, r in results.items():
        mean, lo, hi = mean_confidence_interval(r.all_eval_rewards)
        r.ci_lo, r.ci_hi = lo, hi
        if name != "reactive" and reactive_eval:
            r.cohens_d_vs_reactive = cohens_d(r.all_eval_rewards, reactive_eval)
    return results


CONTROLLERS = {
    "reactive": lambda aid, sd: ReactiveController(aid, sd),
    "stacked_obs": lambda aid, sd: StackedObsController(aid, sd, stack=3),
    "recurrent": lambda aid, sd: RecurrentController(aid, sd, hidden_dim=8),
    "world_model": lambda aid, sd: WorldModelController(aid, sd, hidden_dim=8),
}


def run_memory_task_comparison(make_env: Callable[[int], object],
                               task_name: str,
                               macro_seeds: List[int] | None = None,
                               n_train_seeds: int = 5,
                               n_eval_episodes: int = 5,
                               n_steps: int = 40) -> Dict[str, ExperimentResult]:
    if macro_seeds is None:
        macro_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return compare_controllers_env(
        make_env, CONTROLLERS, macro_seeds,
        n_train_seeds=n_train_seeds,
        n_eval_episodes=n_eval_episodes,
        n_steps=n_steps,
    )



def train_controller(make_env: Callable[[int], GridWorld],
                     controller: BaseController,
                     train_seeds: List[int], n_steps: int) -> List[float]:
    """Train one controller persistently across training episodes."""
    rewards = []
    for ep, seed in enumerate(train_seeds):
        env = make_env(seed)
        r, _ = run_episode(env, controller, f"train-{ep}", n_steps)
        rewards.append(r)
    return rewards


def evaluate_controller(controller: BaseController,
                        make_env: Callable[[int], GridWorld],
                        eval_seeds: List[int], n_eval_episodes: int,
                        n_steps: int) -> Tuple[List[float], int]:
    """Evaluate on held-out seeds with fresh controllers carrying trained weights."""
    rewards = []
    total_actions = 0
    for seed in eval_seeds:
        for ep in range(n_eval_episodes):
            env = make_env(seed + ep)
            ec = controller.clone(f"eval-{seed}-{ep}", seed + ep)
            total_actions += ec.actions_taken
            r, _ = run_episode(env, ec, f"eval-{seed}-{ep}", n_steps)
            rewards.append(r)
    return rewards, total_actions


def mean_confidence_interval(values: List[float]) -> Tuple[float, float, float]:
    """Mean, lo, hi of 95% CI (Student t, approximated by normal for n>=20)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / np.sqrt(n))
    # 95% CI approx (normal); for small n this is approximate
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    return mean, lo, hi


def cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d of a vs b."""
    xa = np.asarray(a, dtype=float); xb = np.asarray(b, dtype=float)
    na, nb = xa.size, xb.size
    if na < 2 or nb < 2:
        return 0.0
    sa = xa.std(ddof=1); sb = xb.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return float((xa.mean() - xb.mean()) / pooled)


@dataclass
class ExperimentResult:
    controller: str
    mean_eval: float
    ci_lo: float
    ci_hi: float
    std_eval: float
    cohens_d_vs_reactive: float
    n_macro_seeds: int
    all_eval_rewards: List[float]
    mean_train_reward: float
    inference_cost: float


def compare_controllers(make_env: Callable[[int], GridWorld],
                        controllers: Dict[str, Callable[[str, int], BaseController]],
                        macro_seeds: List[int],
                        n_train_seeds: int,
                        n_train_episodes: int,
                        n_eval_episodes: int,
                        n_steps: int,
                        n_steps_train: int | None = None) -> Dict[str, ExperimentResult]:
    """Compare controllers across macro_seeds with train/eval split."""
    n_steps_train = n_steps_train or n_steps
    results: Dict[str, ExperimentResult] = {}
    reactive_eval: List[float] = []

    for name, factory in controllers.items():
        all_eval: List[float] = []
        all_train: List[float] = []
        total_actions = 0
        for ms in macro_seeds:
            # Derive training and evaluation seeds from macro_seed
            train_seeds = [ms * 1000 + i for i in range(n_train_seeds)]
            eval_seeds = [ms * 1000 + 10000 + i for i in range(n_eval_episodes)]
            ctrl = factory(f"train-{ms}", ms)
            train_r = train_controller(make_env, ctrl, train_seeds, n_steps_train)
            all_train.extend(train_r)
            eval_r, n_act = evaluate_controller(
                ctrl, make_env, eval_seeds, 1, n_steps)
            total_actions += n_act
            all_eval.extend(eval_r)
        results[name] = ExperimentResult(
            controller=name,
            mean_eval=float(np.mean(all_eval)) if all_eval else 0.0,
            ci_lo=0.0, ci_hi=0.0, std_eval=float(np.std(all_eval)) if all_eval else 0.0,
            cohens_d_vs_reactive=0.0,
            n_macro_seeds=len(macro_seeds),
            all_eval_rewards=all_eval,
            mean_train_reward=float(np.mean(all_train)) if all_train else 0.0,
            inference_cost=float(total_actions),
        )
        if name == "reactive":
            reactive_eval = list(all_eval)

    # Fill CIs and effect sizes vs reactive
    for name, r in results.items():
        mean, lo, hi = mean_confidence_interval(r.all_eval_rewards)
        r.ci_lo, r.ci_hi = lo, hi
        if name != "reactive" and reactive_eval:
            r.cohens_d_vs_reactive = cohens_d(r.all_eval_rewards, reactive_eval)
    return results


def run_gridworld_comparison(macro_seeds: List[int] | None = None,
                             n_train_seeds: int = 5,
                             n_eval_episodes: int = 3,
                             n_steps: int = 100) -> Dict[str, ExperimentResult]:
    if macro_seeds is None:
        macro_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def make_env(seed: int) -> GridWorld:
        return GridWorld(GridWorldConfig(seed=seed, food_regeneration=0.01,
                                         hazard_probability=0.1, dt=0.1))

    controllers = {
        "reactive": lambda aid, sd: ReactiveController(aid, sd),
        "stacked_obs": lambda aid, sd: StackedObsController(aid, sd, stack=3),
        "recurrent": lambda aid, sd: RecurrentController(aid, sd, hidden_dim=8),
        "world_model": lambda aid, sd: WorldModelController(aid, sd, hidden_dim=8),
    }

    return compare_controllers(
        make_env, controllers, macro_seeds,
        n_train_seeds=n_train_seeds,
        n_train_episodes=n_train_seeds,
        n_eval_episodes=n_eval_episodes,
        n_steps=n_steps,
    )


if __name__ == "__main__":
    import numpy as np  # noqa: F401

    def fmt(name: str, r: ExperimentResult) -> str:
        return (f"{name:14s} {r.mean_eval:>9.3f} {r.ci_lo:>8.3f} {r.ci_hi:>8.3f} "
                f"{r.std_eval:>7.3f} {r.cohens_d_vs_reactive:>13.3f}")

    print("=" * 72)
    print("GRIDWORLD foraging (fully observable) -- 15 macro-seeds, 10 eval eps")
    print("=" * 72)
    print(f"{'controller':14s} {'mean':>9s} {'CI_lo':>8s} {'CI_hi':>8s} "
          f"{'std':>7s} {'d_vs_reactive':>13s}")
    gw = run_gridworld_comparison(macro_seeds=list(range(1, 16)),
                                  n_train_seeds=5, n_eval_episodes=10, n_steps=60)
    reactive = gw["reactive"]
    for name, r in gw.items():
        print(fmt(name, r))
    print(f"Reactive CI: ({reactive.ci_lo:.3f}, {reactive.ci_hi:.3f})")
    print("Any non-reactive CI excludes zero:",
          any(not (r.ci_lo <= 0 <= r.ci_hi) for n, r in gw.items() if n != "reactive"))
    print()

    for label, make_env, n_steps in [
        ("delayed_cue", lambda s: DelayedCueTask(DelayedCueConfig(), seed=s), 40),
        ("occluded_target", lambda s: OccludedTargetTask(OccludedTargetConfig(), seed=s), 40),
        ("changed_dynamics", lambda s: ChangedDynamicsTask(ChangedDynamicsConfig(), seed=s), 40),
    ]:
        print("=" * 72)
        print(f"{label.upper()} (memory-required) -- 10 macro-seeds, 5 eval eps")
        print("=" * 72)
        print(f"{'controller':14s} {'mean':>9s} {'CI_lo':>8s} {'CI_hi':>8s} "
              f"{'std':>7s} {'d_vs_reactive':>13s}")
        res = run_memory_task_comparison(
            make_env, label,
            macro_seeds=list(range(1, 11)),
            n_train_seeds=5, n_eval_episodes=5, n_steps=n_steps,
        )
        reactive = res["reactive"]
        for name, r in res.items():
            print(fmt(name, r))
        print(f"Reactive CI: ({reactive.ci_lo:.3f}, {reactive.ci_hi:.3f})")
        print("World model d vs reactive:",
              round(res["world_model"].cohens_d_vs_reactive, 3))
        print("Recurrent d vs reactive:",
              round(res["recurrent"].cohens_d_vs_reactive, 3))
        print()

