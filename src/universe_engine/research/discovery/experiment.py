"""Scientific discovery laboratory (Sprint 7, Research Program F).

Provides a physical experiment apparatus (Bell experiment) that agents can
operate through their own sensory interface. Agents are NOT supplied the CHSH
formula. They must gather data, model measurement correlations, compare
classical and quantum candidate explanations, and predict outcomes for new
settings.

A mathematical discovery mechanism uses symbolic regression / model selection
over a restricted space of candidate correlation functions.

No claim is made that the agent understands quantum mechanics -- it only
selects the model that best predicts held-out observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import math
import numpy as np

from ..contracts import AgentAction, SensoryObservation, ExperimentManifest
from ..quantum import EntangledSystem
from ..quantum.chsh import measurement_in_basis, optimal_chsh_settings


# ---------------------------------------------------------------------------
# Bell experiment apparatus (physical instrument)
# ---------------------------------------------------------------------------

@dataclass
class BellApparatus:
    """A physical Bell experiment apparatus.

    Prepares entangled photon pairs and measures them in selectable bases.
    The agent can set measurement angles for Alice and Bob and receive
    coincidence outcomes. The agent does NOT know the internal quantum state
    or the CHSH formula.

    The apparatus is the ground-truth physical system. Agents interact with
    it only through the measure() method, which returns (outcome_a, outcome_b)
    for given settings.
    """

    bell_state: str = "phi_plus"
    seed: int = 42
    n_shots_per_setting: int = 100

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.entangled = EntangledSystem.bell_state(self.bell_state)
        self.rho = self.entangled.density.matrix.copy()
        self.measurement_history: List[Dict] = []

    def measure(self, angle_a: float, angle_b: float,
                n_shots: Optional[int] = None) -> Dict:
        """Perform one Bell measurement with settings (angle_a, angle_b).

        Returns counts for the four joint outcomes and the correlation E.
        The agent sees only these counts -- not the quantum state or formula.
        """
        if n_shots is None:
            n_shots = self.n_shots_per_setting

        Ma = measurement_in_basis(angle_a)
        Mb = measurement_in_basis(angle_b)

        n_pp = n_pm = n_mp = n_mm = 0
        for _ in range(n_shots):
            # Sample joint outcome using Born rule
            joint_probs = []
            for Pa, oa in zip(Ma.projectors, Ma.outcomes):
                for Pb, ob in zip(Mb.projectors, Mb.outcomes):
                    Pj = np.kron(Pa, Pb)
                    p = float(np.real(np.trace(self.rho @ Pj)))
                    joint_probs.append(max(0.0, p))
            total = sum(joint_probs)
            if total > 0:
                joint_probs = [p / total for p in joint_probs]
            idx = int(self.rng.choice(len(Ma.projectors) * len(Mb.projectors),
                                      p=joint_probs))
            # Decode idx into (oa, ob)
            na = len(Ma.outcomes)
            oa = Ma.outcomes[idx // na]
            ob = Mb.outcomes[idx % na]
            if oa == "+" and ob == "+":
                n_pp += 1
            elif oa == "+" and ob == "-":
                n_pm += 1
            elif oa == "-" and ob == "+":
                n_mp += 1
            else:
                n_mm += 1

        E = ((n_pp + n_mm) - (n_pm + n_mp)) / n_shots

        record = {
            "angle_a": float(angle_a), "angle_b": float(angle_b),
            "n_shots": n_shots, "counts": {"++": n_pp, "+-": n_pm,
                                           "-+": n_mp, "--": n_mm},
            "correlation": float(E),
            "t": time.time(),
        }
        self.measurement_history.append(record)
        return record

    def chsh(self, angles: Tuple[float, float, float, float],
             n_shots: Optional[int] = None) -> float:
        """Compute CHSH S from four measurement settings."""
        a, ap, b, bp = angles
        E_ab = self.measure(a, b, n_shots)["correlation"]
        E_abp = self.measure(a, bp, n_shots)["correlation"]
        E_apb = self.measure(ap, b, n_shots)["correlation"]
        E_apbp = self.measure(ap, bp, n_shots)["correlation"]
        return E_ab - E_abp + E_apb + E_apbp

    def history(self) -> List[Dict]:
        return list(self.measurement_history)

    def reset(self) -> None:
        self.measurement_history = []


# ---------------------------------------------------------------------------
# Candidate correlation models (scientific hypotheses)
# ---------------------------------------------------------------------------

class CorrelationModel:
    """A candidate model for measurement correlations E(angle_a, angle_b).

    Models are fit to data and used to predict correlations at new settings.
    """

    def fit(self, data: List[Dict]) -> None:
        """Fit model parameters to measurement data."""
        raise NotImplementedError

    def predict(self, angle_a: float, angle_b: float) -> float:
        """Predict correlation E(angle_a, angle_b)."""
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError

    def residual_error(self, data: List[Dict]) -> float:
        """Mean absolute error on data."""
        errs = []
        for d in data:
            pred = self.predict(d["angle_a"], d["angle_b"])
            errs.append(abs(pred - d["correlation"]))
        return float(np.mean(errs)) if errs else float("inf")


class ClassicalLocalModel(CorrelationModel):
    """Classical local hidden-variable model: E(a,b) = c * cos(a-b).

    The amplitude c is constrained to [0, 1/sqrt(2)] so that the CHSH
    parameter satisfies |S| <= 2 (the classical local bound). This is a
    phenomenological classical model -- not a claim about local realism.
    """

    def __init__(self) -> None:
        self.c = 0.7071067811865476  # 1/sqrt(2)

    def fit(self, data: List[Dict]) -> None:
        # Fit amplitude c in [0, 1/sqrt(2)] to minimize residual error.
        c_max = 1.0 / np.sqrt(2.0)
        best_c = 0.0
        best_err = float("inf")
        for c in np.linspace(0.0, c_max, 41):
            err = 0.0
            for d in data:
                pred = c * np.cos(d["angle_a"] - d["angle_b"])
                err += abs(pred - d["correlation"])
            err /= len(data)
            if err < best_err:
                best_err = err
                best_c = float(c)
        self.c = best_c

    def predict(self, angle_a: float, angle_b: float) -> float:
        return float(self.c * np.cos(angle_a - angle_b))

    def name(self) -> str:
        return "classical_local"


class QuantumModel(CorrelationModel):
    """Quantum model: E(a,b) = cos(scale * (a-b)).

    For the phi+ Bell state, scale = 2. The model can violate the CHSH
    bound (S up to 2*sqrt(2)).
    """

    def __init__(self) -> None:
        self.scale = 2.0

    def fit(self, data: List[Dict]) -> None:
        # Fit scale parameter freely.
        best_scale = 2.0
        best_err = float("inf")
        for scale in np.linspace(0.5, 4.0, 71):
            err = 0.0
            for d in data:
                pred = np.cos(scale * (d["angle_a"] - d["angle_b"]))
                err += abs(pred - d["correlation"])
            err /= len(data)
            if err < best_err:
                best_err = err
                best_scale = float(scale)
        self.scale = best_scale

    def predict(self, angle_a: float, angle_b: float) -> float:
        return float(np.cos(self.scale * (angle_a - angle_b)))

    def name(self) -> str:
        return "quantum"


class ConstantModel(CorrelationModel):
    """Null model: E(a,b) = c (no correlation structure)."""

    def __init__(self) -> None:
        self.c = 0.0

    def fit(self, data: List[Dict]) -> None:
        if data:
            self.c = float(np.mean([d["correlation"] for d in data]))

    def predict(self, angle_a: float, angle_b: float) -> float:
        return self.c

    def name(self) -> str:
        return "constant"


# ---------------------------------------------------------------------------
# Scientific discovery agent
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryResult:
    """Result of a scientific discovery experiment."""
    n_training_settings: int
    n_test_settings: int
    training_error: Dict[str, float]  # model -> mean abs error
    test_error: Dict[str, float]      # model -> mean abs error on held-out
    selected_model: str               # model with lowest test error
    chsh_quantum: float               # measured CHSH S on quantum settings
    chsh_classical: float             # measured CHSH S on classical baseline
    selected_model_chsh: float        # predicted CHSH S from selected model
    discovery_successful: bool        # quantum model selected on held-out data
    notes: List[str] = field(default_factory=list)


def run_discovery_experiment(seed: int = 42,
                             n_training: int = 24,
                             n_test: int = 12,
                             n_shots: int = 2000) -> DiscoveryResult:
    """Run a scientific discovery experiment.

    An agent operates a Bell apparatus without knowing the CHSH formula.
    It collects training data at various settings, fits three candidate
    models (classical local, quantum, constant), and evaluates them on
    held-out test settings.

    Returns a DiscoveryResult with model comparison and CHSH measurements.
    """
    apparatus = BellApparatus(bell_state="phi_plus", seed=seed,
                              n_shots_per_setting=n_shots)

    # Generate training settings (diverse angle pairs)
    rng = np.random.default_rng(seed + 1)
    training_settings = []
    # Include CHSH-relevant settings so models can be distinguished.
    chsh_settings = [
        (0.0, math.pi / 4.0),
        (0.0, 3.0 * math.pi / 4.0),
        (math.pi / 2.0, math.pi / 4.0),
        (math.pi / 2.0, 3.0 * math.pi / 4.0),
    ]
    training_settings.extend(chsh_settings)
    for _ in range(n_training - len(chsh_settings)):
        a = float(rng.uniform(0, math.pi))
        b = float(rng.uniform(0, math.pi))
        training_settings.append((a, b))

    # Collect training data
    training_data = []
    for a, b in training_settings:
        r = apparatus.measure(a, b, n_shots)
        training_data.append(r)

    # Fit candidate models
    models = [ClassicalLocalModel(), QuantumModel(), ConstantModel()]
    for m in models:
        m.fit(training_data)

    # Training error
    train_error = {m.name(): m.residual_error(training_data) for m in models}

    # Generate held-out test settings
    test_settings = []
    for _ in range(n_test):
        a = float(rng.uniform(0, np.pi))
        b = float(rng.uniform(0, np.pi))
        test_settings.append((a, b))

    # Collect test data
    test_data = []
    for a, b in test_settings:
        r = apparatus.measure(a, b, n_shots)
        test_data.append(r)

    # Test error
    test_error = {m.name(): m.residual_error(test_data) for m in models}

    # Select best model on held-out test data
    selected = min(models, key=lambda m: test_error[m.name()])
    selected_name = selected.name()

    # Measure CHSH S on quantum-optimal settings
    from ..quantum.chsh import optimal_chsh_settings
    q_settings = optimal_chsh_settings()
    chsh_quantum = apparatus.chsh(q_settings, n_shots)

    # Classical model's predicted CHSH S:
    cl_model = ClassicalLocalModel()
    cl_model.fit(training_data)
    a, ap, b, bp = q_settings
    cl_chsh = (cl_model.predict(a, b) - cl_model.predict(a, bp)
               + cl_model.predict(ap, b) + cl_model.predict(ap, bp))

    # Quantum model's predicted CHSH S:
    qm_model = QuantumModel()
    qm_model.fit(training_data)
    qm_chsh = (qm_model.predict(a, b) - qm_model.predict(a, bp)
               + qm_model.predict(ap, b) + qm_model.predict(ap, bp))

    # Discovery successful if quantum model has lowest test error
    discovery_successful = (selected_name == "quantum")

    notes = []
    if discovery_successful:
        notes.append("Agent selected quantum model on held-out data")
    else:
        notes.append(f"Agent selected {selected_name} model; quantum not preferred")
    notes.append(f"Measured CHSH S (quantum apparatus): {chsh_quantum:.3f}")
    notes.append(f"Classical model predicted CHSH S: {cl_chsh:.3f}")
    notes.append(f"Quantum model predicted CHSH S: {qm_chsh:.3f}")
    notes.append("Agent was NOT supplied the CHSH formula")

    return DiscoveryResult(
        n_training_settings=len(training_settings),
        n_test_settings=len(test_settings),
        training_error=train_error,
        test_error=test_error,
        selected_model=selected_name,
        chsh_quantum=float(chsh_quantum),
        chsh_classical=float(cl_chsh),
        selected_model_chsh=float(qm_chsh if selected_name == "quantum" else cl_chsh),
        discovery_successful=bool(discovery_successful),
        notes=notes,
    )
