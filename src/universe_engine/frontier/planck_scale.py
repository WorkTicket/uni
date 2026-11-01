"""Planck-scale theoretical research layer (Requirement 9 / Test I).

Calculates Planck units from physical constants (CODATA/NIST):
  - Planck length:  l_P = sqrt(hbar * G / c^3)
  - Planck time:    t_P = sqrt(hbar * G / c^5)
  - Planck mass:    m_P = sqrt(hbar * c / G)
  - Planck energy:  E_P = m_P c^2 = sqrt(hbar * c^5 / G)

Uses numerically stable dimensionless quantities and unit systems.

At extremely small scales, transitions from validated effective physics to
clearly identified theoretical models. Possible experimental models include
simplified discrete geometry and spin-network-inspired constructions.

Each model needs:
  - Defined degrees of freedom
  - Governing equations
  - Quantum or statistical interpretation
  - Initial state
  - Observables
  - Numerical method
  - Validity and uncertainty metadata

Scientific status: SPECULATIVE / RESEARCH.
  - Do NOT assume spacetime is confirmed to be discrete.
  - Do NOT treat the Planck length as a proven smallest possible distance.
  - Do NOT simulate every Planck-sized cell across an entire universe.
  - Use selected local calculations and effective multiscale descriptions.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# CODATA 2018 physical constants (SI)
# Source: NIST CODATA 2018 (exact where noted)
C_LIGHT = 299792458.0                # m/s, exact (SI definition)
H_BAR = 1.054571817e-34              # J s (reduced Planck constant, CODATA 2018)
G_NEWTON = 6.67430e-11               # m^3 kg^-1 s^-2 (CODATA 2018)
K_BOLTZMANN = 1.380649e-23           # J/K, exact (SI definition)
ELEMENTARY_CHARGE = 1.602176634e-19  # C, exact (SI definition)


@dataclass
class PlanckUnits:
    """Planck units calculated from physical constants."""
    planck_length_m: float           # l_P = sqrt(hbar G / c^3)
    planck_time_s: float             # t_P = sqrt(hbar G / c^5)
    planck_mass_kg: float            # m_P = sqrt(hbar c / G)
    planck_energy_J: float           # E_P = m_P c^2
    planck_temperature_K: float      # T_P = E_P / k_B
    planck_density_kg_m3: float      # rho_P = m_P / l_P^3

    @classmethod
    def from_constants(cls, c: float = C_LIGHT, hbar: float = H_BAR,
                       G: float = G_NEWTON, k_b: float = K_BOLTZMANN):
        """Calculate all Planck units from fundamental constants."""
        l_p = math.sqrt(hbar * G / c**3)
        t_p = math.sqrt(hbar * G / c**5)
        m_p = math.sqrt(hbar * c / G)
        e_p = m_p * c**2
        t_p_k = e_p / k_b
        rho_p = m_p / l_p**3
        return cls(
            planck_length_m=l_p,
            planck_time_s=t_p,
            planck_mass_kg=m_p,
            planck_energy_J=e_p,
            planck_temperature_K=t_p_k,
            planck_density_kg_m3=rho_p,
        )

    def to_dict(self) -> Dict:
        return asdict(self)


# Standard Planck units (CODATA 2018)
PLANCK_UNITS = PlanckUnits.from_constants()


@dataclass
class QuantumGeometryModel:
    """Base class for a quantum-spacetime research model.

    Each concrete model declares:
      - degrees of freedom
      - governing equations
      - quantum/statistical interpretation
      - initial state
      - observables
      - numerical method
      - validity & uncertainty metadata
    """
    model_id: str = ""
    theory_classification: str = "speculative"  # 'speculative' | 'research'
    description: str = ""
    degrees_of_freedom: str = ""
    governing_equations: str = ""
    interpretation: str = ""
    validity_domain: Dict = field(default_factory=dict)
    uncertainty_metadata: Dict = field(default_factory=dict)

    def initial_state(self) -> Dict:
        """Return the model's initial state."""
        raise NotImplementedError

    def observables(self, state: Dict) -> Dict:
        """Return observables computed from a state."""
        raise NotImplementedError

    def step(self, state: Dict, dt: float) -> Dict:
        """Advance the model by dt using its numerical method."""
        raise NotImplementedError

    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "theory_classification": self.theory_classification,
            "description": self.description,
            "degrees_of_freedom": self.degrees_of_freedom,
            "governing_equations": self.governing_equations,
            "interpretation": self.interpretation,
            "validity_domain": self.validity_domain,
            "uncertainty_metadata": self.uncertainty_metadata,
        }


@dataclass
class DiscreteGeometryModel(QuantumGeometryModel):
    """Simplified discrete geometry model (causal-set inspired).

    Spacetime approximated as a finite set of discrete elements with causal
    relations. NOT a claim that spacetime is fundamentally discrete.

    Degrees of freedom: N discrete elements with adjacency matrix.
    Governing equation: element addition with probability governed by
    causal-set dynamics (simplified).
    Interpretation: statistical ensemble of discrete spacetimes.
    """
    n_elements: int = 0
    max_elements: int = 10000
    adjacency: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def __post_init__(self):
        if self.model_id == "":
            self.model_id = "discrete_geometry"
        if self.theory_classification == "":
            self.theory_classification = "speculative"

    def initial_state(self) -> Dict:
        self.n_elements = 1
        self.adjacency = np.zeros((1, 1), dtype=np.int64)
        return {
            "n_elements": self.n_elements,
            "adjacency": self.adjacency.copy(),
            "t": 0.0,
        }

    def observables(self, state: Dict) -> Dict:
        n = state["n_elements"]
        adj = state["adjacency"]
        n_links = int(np.sum(adj)) // 2  # undirected links
        return {
            "n_elements": n,
            "n_links": n_links,
            "link_density": n_links / max(n * (n - 1) / 2, 1),
            "discrete_volume_proxy": float(n),
        }

    def step(self, state: Dict, dt: float) -> Dict:
        """Add one element with a causal link to a random existing element."""
        n = state["n_elements"]
        if n >= self.max_elements:
            return state
        rng = np.random.default_rng()
        new_adj = np.zeros((n + 1, n + 1), dtype=np.int64)
        new_adj[:n, :n] = state["adjacency"]
        # New element links to one random previous element (causal relation)
        target = int(rng.integers(0, n))
        new_adj[target, n] = 1
        state["n_elements"] = n + 1
        state["adjacency"] = new_adj
        state["t"] += dt
        return state


@dataclass
class SpinNetworkInspiredModel(QuantumGeometryModel):
    """Spin-network-inspired quantum geometry model.

    Graph with edges labelled by spins j (half-integers). Area and volume
    operators have discrete spectra:
      A_j = 8*pi*gamma*l_P^2 * sqrt(j(j+1))   (area)
    where gamma is the Barbero-Immirzi parameter.

    NOT a claim of loop quantum gravity; this is an illustrative model.
    """
    gamma: float = 0.234  # Barbero-Immirzi parameter (illustrative)
    edges: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if self.model_id == "":
            self.model_id = "spin_network_inspired"
        if self.theory_classification == "":
            self.theory_classification = "speculative"

    def initial_state(self) -> Dict:
        self.edges = [{"spin": 0.5, "nodes": (0, 1)}]
        return {
            "edges": [dict(e) for e in self.edges],
            "gamma": self.gamma,
            "planck_length_sq": PLANCK_UNITS.planck_length_m**2,
            "t": 0.0,
        }

    def area_spectrum(self, j: float) -> float:
        """Area eigenvalue for spin j: A = 8*pi*gamma*l_P^2 * sqrt(j(j+1))."""
        l_p2 = PLANCK_UNITS.planck_length_m**2
        return 8.0 * math.pi * self.gamma * l_p2 * math.sqrt(j * (j + 1))

    def observables(self, state: Dict) -> Dict:
        edges = state["edges"]
        total_area = sum(self.area_spectrum(e["spin"]) for e in edges)
        return {
            "n_edges": len(edges),
            "total_area_m2": total_area,
            "planck_area_units": total_area / PLANCK_UNITS.planck_length_m**2,
        }

    def step(self, state: Dict, dt: float) -> Dict:
        """Add a new edge with spin 1/2 connecting two random nodes."""
        edges = [dict(e) for e in state["edges"]]
        n_nodes = max(max(e["nodes"]) for e in edges) + 1 if edges else 2
        rng = np.random.default_rng()
        n1 = int(rng.integers(0, n_nodes))
        n2 = int(rng.integers(0, n_nodes))
        if n1 == n2:
            n2 = (n2 + 1) % max(n_nodes, 2)
        edges.append({"spin": 0.5, "nodes": (n1, n2)})
        state["edges"] = edges
        state["t"] += dt
        return state


@dataclass
class PlanckScaleLayer:
    """Manages Planck-scale research models.

    Provides Planck unit calculations and access to quantum-spacetime
    research models. All models are explicitly labeled speculative/research.
    """
    planck_units: PlanckUnits = field(default_factory=PlanckUnits.from_constants)
    models: Dict[str, QuantumGeometryModel] = field(default_factory=dict)

    def register_model(self, model: QuantumGeometryModel):
        self.models[model.model_id] = model

    def planck_length(self) -> float:
        return self.planck_units.planck_length_m

    def planck_time(self) -> float:
        return self.planck_units.planck_time_s

    def planck_mass(self) -> float:
        return self.planck_units.planck_mass_kg

    def planck_energy(self) -> float:
        return self.planck_units.planck_energy_J

    def dimensionless_scale(self, length_m: float) -> float:
        """Return length in Planck units (dimensionless)."""
        return length_m / self.planck_units.planck_length_m

    def dimensionless_time(self, t_s: float) -> float:
        """Return time in Planck units (dimensionless)."""
        return t_s / self.planck_units.planck_time_s

    def to_dict(self) -> Dict:
        return {
            "planck_units": self.planck_units.to_dict(),
            "models": {k: m.to_dict() for k, m in self.models.items()},
        }


# Global Planck-scale layer instance
PLANCK_LAYER = PlanckScaleLayer()
PLANCK_LAYER.register_model(DiscreteGeometryModel())
PLANCK_LAYER.register_model(SpinNetworkInspiredModel())
