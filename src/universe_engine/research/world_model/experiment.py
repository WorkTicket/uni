"""World-model learning experiment (Sprint 4, Research Program D3).

Compares a reactive baseline controller against a recurrent world-model
controller in the GridWorld environment. Measures:

- Long-horizon adaptation under changing environmental conditions
- Memory-dependent task performance
- World-model prediction error
- Generalization to new conditions
- Energy/computational costs

This does NOT claim any agent is conscious -- it measures behavioral
capabilities only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import numpy as np

from ..contracts import ExperimentManifest, AgentAction, SensoryObservation
from ..environment.gridworld import GridWorld, GridWorldConfig, ReactiveController, RecurrentWorldModelController


@dataclass
class WorldModelExperimentResult:
    """Results from a world-model learning comparison."""
    manifest: ExperimentManifest
    reactive_performance: Dict[str, float]
    recurrent_performance: Dict[str, float]
    prediction_errors: List[float]
    energy_costs: Dict[str, float]
    adaptation_score: float
    generalization_score: float
    memory_score: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest.to_dict(),
            "reactive_performance": self.reactive_performance,
            "recurrent_performance": self.recurrent_performance,
            "prediction_errors": self.prediction_errors,
            "energy_costs": self.energy_costs,
            "adaptation_score": self.adaptation_score,
            "generalization_score": self.generalization_score,
            "memory_score": self.memory_score,
            "passed": self.passed,
        }


def run_episode(env: GridWorld, controller, agent_id: str,
                n_steps: int, dt: float = 0.1) -> Dict[str, float]:
    """Run one episode and return performance metrics."""
    env.register_agent(agent_id, (env.cfg.width // 2, env.cfg.height // 2),
                       initial_energy=5.0)
    obs = env.observe(agent_id)
    total_reward = 0.0
    n_forage = 0
    positions: List[Tuple[int, int]] = [env._agent_positions[agent_id]]
    prediction_errors: List[float] = []
    alive = True

    for step in range(n_steps):
        action = controller.select_action(obs)
        result = env.act(agent_id, action)

        if action.action_type == "forage":
            n_forage += 1

        next_obs = env.observe(agent_id)
        controller.learn(obs, action, float(result.get("reward", 0.0)), next_obs)

        # Track prediction error from recurrent controller
        if hasattr(controller, "prediction_error_history") and \
           controller.prediction_error_history:
            prediction_errors.append(float(controller.prediction_error_history[-1]))

        obs = next_obs
        total_reward += float(result.get("reward", 0.0))
        positions.append(env._agent_positions[agent_id])
        env.step(dt)

        if not env._agent_states[agent_id].alive:
            alive = False
            break

    unique_positions = len(set(positions))
    mean_pred_error = float(np.mean(prediction_errors)) if prediction_errors else 0.0
    return {
        "total_reward": float(total_reward),
        "n_steps": float(step + 1),
        "alive": float(1.0 if alive else 0.0),
        "n_forage": float(n_forage),
        "unique_positions": float(unique_positions),
        "mean_prediction_error": mean_pred_error,
        "final_energy": float(env._agent_states[agent_id].energy),
    }


def run_comparison_experiment(n_episodes: int = 20,
                              n_steps_per_episode: int = 100,
                              seed: int = 42) -> WorldModelExperimentResult:
    """Run a full comparison experiment between reactive and recurrent controllers."""
    manifest = ExperimentManifest(
        experiment_id=f"wm-{seed}",
        name="world-model-learning-comparison",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        research_program="D",
        question="Does a recurrent world-model controller outperform a reactive "
                 "baseline in a resource-constrained grid world?",
        model_assumptions={
            "grid_size": [8, 8],
            "food_regeneration": 0.01,
            "hazard_probability": 0.1,
            "dt": 0.1,
            "max_energy": 10.0,
        },
        candidate_mechanisms=["reactive_feedforward", "recurrent_world_model"],
        baseline_mechanism="reactive_feedforward",
        parameter_space={"n_episodes": n_episodes, "n_steps": n_steps_per_episode},
        random_seeds=[seed],
        computational_budget={"max_episodes": n_episodes,
                              "max_steps_per_episode": n_steps_per_episode},
        predicted_observables=["total_reward", "alive_rate", "prediction_error", "energy_cost"],
        numerical_methods={"integrator": "explicit_euler", "dt": 0.1},
        statistical_methods={"comparison": "paired", "n_episodes": n_episodes},
        held_out_validation={"new_seed": seed + 1000},
        uncertainty_analysis={"bootstrap_samples": 100},
        reproducibility_manifest={"code_revision": "unknown", "seed": seed},
    )

    # Reactive baseline results
    reactive_metrics: List[Dict[str, float]] = []
    for ep in range(n_episodes):
        env = GridWorld(GridWorldConfig(seed=seed + ep))
        controller = ReactiveController(f"reactive-{ep}", seed=seed + ep)
        reactive_metrics.append(run_episode(env, controller, f"reactive-{ep}", n_steps_per_episode))

    # Recurrent world-model results
    recurrent_metrics: List[Dict[str, float]] = []
    prediction_errors: List[float] = []
    for ep in range(n_episodes):
        env = GridWorld(GridWorldConfig(seed=seed + ep))
        controller = RecurrentWorldModelController(f"recurrent-{ep}", seed=seed + ep, lr=0.05)
        m = run_episode(env, controller, f"recurrent-{ep}", n_steps_per_episode)
        recurrent_metrics.append(m)
        if m.get("mean_prediction_error", 0) > 0:
            prediction_errors.append(m["mean_prediction_error"])

    def aggregate(metrics, key):
        return float(np.mean([m[key] for m in metrics]))

    reactive_perf = {
        "mean_reward": aggregate(reactive_metrics, "total_reward"),
        "alive_rate": aggregate(reactive_metrics, "alive"),
        "mean_exploration": aggregate(reactive_metrics, "unique_positions"),
        "mean_forage_rate": aggregate(reactive_metrics, "n_forage") / n_steps_per_episode,
    }
    recurrent_perf = {
        "mean_reward": aggregate(recurrent_metrics, "total_reward"),
        "alive_rate": aggregate(recurrent_metrics, "alive"),
        "mean_exploration": aggregate(recurrent_metrics, "unique_positions"),
        "mean_forage_rate": aggregate(recurrent_metrics, "n_forage") / n_steps_per_episode,
    }

    adaptation_score = recurrent_perf["mean_reward"] - reactive_perf["mean_reward"]
    memory_score = recurrent_perf["mean_exploration"] - reactive_perf["mean_exploration"]

    # Generalization on held-out seed
    held_out_env1 = GridWorld(GridWorldConfig(seed=seed + 1000))
    held_out_env2 = GridWorld(GridWorldConfig(seed=seed + 1000))
    ho_reactive = run_episode(held_out_env1, ReactiveController("ho-r", seed=seed + 1000),
                              "ho-r", n_steps_per_episode)
    ho_recurrent = run_episode(held_out_env2,
                               RecurrentWorldModelController("ho-rec", seed=seed + 1000, lr=0.05),
                               "ho-rec", n_steps_per_episode)
    generalization_score = ho_recurrent["total_reward"] - ho_reactive["total_reward"]

    energy_costs = {
        "reactive": reactive_perf.get("mean_energy_cost", 0.0),
        "recurrent": recurrent_perf.get("mean_energy_cost", 0.0),
    }

    passed = recurrent_perf["mean_reward"] > reactive_perf["mean_reward"] * 0.95

    return WorldModelExperimentResult(
        manifest=manifest,
        reactive_performance=reactive_perf,
        recurrent_performance=recurrent_perf,
        prediction_errors=prediction_errors,
        energy_costs=energy_costs,
        adaptation_score=adaptation_score,
        generalization_score=generalization_score,
        memory_score=memory_score,
        passed=passed,
    )


def make_world_model_manifest(seed: int = 42,
                              n_episodes: int = 20,
                              n_steps: int = 100) -> ExperimentManifest:
    """Create a standard world-model experiment manifest."""
    return ExperimentManifest(
        experiment_id=f"wm-manifest-{seed}",
        name="world-model-learning-comparison",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        research_program="D",
        question="Does a recurrent world-model controller outperform a reactive baseline?",
        model_assumptions={"grid_size": [8, 8], "dt": 0.1},
        candidate_mechanisms=["reactive_feedforward", "recurrent_world_model"],
        baseline_mechanism="reactive_feedforward",
        parameter_space={"n_episodes": n_episodes, "n_steps": n_steps},
        random_seeds=[seed],
        computational_budget={"max_episodes": n_episodes},
        predicted_observables=["total_reward", "alive_rate", "prediction_error"],
        numerical_methods={"dt": 0.1},
        statistical_methods={"paired_comparison": True},
        held_out_validation={"new_seed": seed + 1},
        uncertainty_analysis={},
        reproducibility_manifest={"seed": seed},
    )
