"""Self-model module (Research Program D5).

Creates an internal model of the agent's own body, actions, capabilities,
uncertainty, and sensorimotor effects. The self-model allows the agent to
distinguish self-caused sensory changes from external events.

Provenance: self-model is a computational abstraction, not a claim about
subjective self-awareness or consciousness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BodySchema:
    """The agent's model of its own body.

    Parameters
    ----------
    position : tuple
        Estimated body position in the environment frame.
    sensorimotor_map : dict
        Maps action types to expected sensory consequences (e.g., "move" ->
        expected position change).
    capability_uncertainty : dict
        Uncertainty (std) in each capability estimate (0 = certain, 1 = max
        uncertainty).
    energy_capacity : float
        Estimated maximum energy storage.
    """
    position: Tuple[float, ...] = field(default_factory=lambda: (0.0, 0.0))
    sensorimotor_map: Dict[str, np.ndarray] = field(default_factory=dict)
    capability_uncertainty: Dict[str, float] = field(default_factory=dict)
    energy_capacity: float = 10.0

    def predicted_sensory_change(self, action_type: str,
                                 action_params: Dict) -> np.ndarray:
        """Predict the sensory change caused by an action.

        Returns the expected change in sensory inputs based on the current
        body schema. If the action type is not in the map, returns zeros.
        """
        if action_type not in self.sensorimotor_map:
            return np.zeros(2)
        base = self.sensorimotor_map[action_type]
        # Modulate by action parameters
        if action_type == "move" and "dx" in action_params and "dy" in action_params:
            scale = 0.1  # movement scaling
            return base * np.array([action_params["dx"], action_params["dy"]]) * scale
        return base.copy()


@dataclass
class SelfModel:
    """Internal model of the agent's own body, actions, capabilities, and
    sensorimotor effects.

    The self-model is updated through experience: when the agent takes an
    action and observes the resulting sensory change, it updates its
    sensorimotor map and capability uncertainty.

    Key function: distinguish self-caused sensory changes from external
    events by comparing predicted vs actual sensory consequences.
    """

    body: BodySchema
    action_history: List[Dict] = field(default_factory=list)
    calibration_error: float = 0.0  # mean |predicted - actual| sensory change
    n_updates: int = 0

    def predict(self, action_type: str,
                action_params: Dict) -> np.ndarray:
        """Predict the sensory consequence of an action."""
        return self.body.predicted_sensory_change(action_type, action_params)

    def update(self, action_type: str, action_params: Dict,
               predicted: np.ndarray, actual: np.ndarray,
               external_disturbance: float = 0.0) -> Dict[str, float]:
        """Update the self-model from observed sensory consequences.

        Compares predicted vs actual sensory change. The difference that is
        NOT explained by the action is attributed to external events.
        Returns a dict with diagnostics.
        """
        error = np.abs(predicted - actual)
        self.calibration_error = (
            (self.calibration_error * self.n_updates + float(np.mean(error)))
            / (self.n_updates + 1)
        )
        self.n_updates += 1

        # Update sensorimotor map: move predicted toward actual
        if action_type not in self.body.sensorimotor_map:
            self.body.sensorimotor_map[action_type] = actual.copy()
        else:
            old = self.body.sensorimotor_map[action_type]
            lr = 0.1  # learning rate for self-model adaptation
            self.body.sensorimotor_map[action_type] = old + lr * (actual - old)

        # Record the event
        self.action_history.append({
            "action_type": action_type,
            "action_params": dict(action_params),
            "predicted": predicted.tolist(),
            "actual": actual.tolist(),
            "error": float(np.mean(error)),
            "external_disturbance": external_disturbance,
        })
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]

        # Compute attribution: how much of the sensory change is self-caused
        # vs external
        self_caused = float(np.mean(np.abs(predicted)))
        external = float(np.mean(np.abs(actual - predicted)))
        total = self_caused + external
        self_caused_frac = self_caused / total if total > 0 else 0.0

        return {
            "calibration_error": self.calibration_error,
            "self_caused_fraction": self_caused_frac,
            "external_fraction": 1.0 - self_caused_frac,
            "prediction_error": float(np.mean(error)),
            "n_updates": self.n_updates,
        }

    def detect_external_event(self, action_type: str, action_params: Dict,
                              actual_sensory_change: np.ndarray,
                              threshold: float = 0.1) -> Dict[str, object]:
        """Detect whether a sensory change is likely externally caused.

        Compares the predicted sensory consequence of the agent's own action
        against the actual observed change. If the residual exceeds threshold,
        flags an external event.

        Returns dict with: is_external, confidence, residual.
        """
        predicted = self.predict(action_type, action_params)
        residual = np.abs(actual_sensory_change - predicted)
        mean_residual = float(np.mean(residual))
        is_external = bool(mean_residual > threshold)
        confidence = float(np.clip(mean_residual / (threshold + 1e-6), 0.0, 1.0))
        return {
            "is_external": is_external,
            "confidence": confidence,
            "residual": mean_residual,
            "predicted": predicted.tolist(),
            "actual": actual_sensory_change.tolist(),
        }

    def recalibrate(self, disturbance_magnitude: float) -> None:
        """Simulate recalibration after a controlled sensorimotor disturbance.

        Increases capability uncertainty to reflect that the agent's body
        model may be invalid, then allows it to recover through updates.
        """
        for key in self.body.capability_uncertainty:
            self.body.capability_uncertainty[key] = float(
                np.clip(self.body.capability_uncertainty[key] + disturbance_magnitude,
                        0.0, 1.0)
            )

    def checkpoint(self) -> dict:
        return {
            "body_position": list(self.body.position),
            "sensorimotor_map": {k: v.tolist() for k, v in self.body.sensorimotor_map.items()},
            "capability_uncertainty": dict(self.body.capability_uncertainty),
            "energy_capacity": self.body.energy_capacity,
            "calibration_error": self.calibration_error,
            "n_updates": self.n_updates,
            "action_history": self.action_history[-100:],
        }

    @classmethod
    def restore(cls, data: dict) -> "SelfModel":
        body = BodySchema(
            position=tuple(data.get("body_position", [0.0, 0.0])),
            sensorimotor_map={k: np.array(v) for k, v in
                              data.get("sensorimotor_map", {}).items()},
            capability_uncertainty=data.get("capability_uncertainty", {}),
            energy_capacity=float(data.get("energy_capacity", 10.0)),
        )
        return cls(
            body=body,
            calibration_error=float(data.get("calibration_error", 0.0)),
            n_updates=int(data.get("n_updates", 0)),
            action_history=list(data.get("action_history", [])),
        )


def make_default_self_model() -> SelfModel:
    """Create a self-model with default body schema for the GridWorld."""
    body = BodySchema(
        position=(0.0, 0.0),
        sensorimotor_map={
            "move": np.array([1.0, 1.0]),
            "forage": np.array([0.0, 0.0]),
            "rest": np.array([0.0, 0.0]),
            "wait": np.array([0.0, 0.0]),
        },
        capability_uncertainty={"move": 0.1, "forage": 0.05, "rest": 0.0, "wait": 0.0},
        energy_capacity=10.0,
    )
    return SelfModel(body=body)
