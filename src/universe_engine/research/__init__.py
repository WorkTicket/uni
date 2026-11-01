"""Project Genesis research program infrastructure.

Sprint 1: shared agent/environment contracts, experiment manifests, and
reproducible checkpoints. This is the foundation for Research Programs
B (quantum), C (evolution), D (embodied minds), E (consciousness hypotheses),
and F (scientific discovery).
"""

from .contracts import (
    PhysicalQuantity,
    AgentIdentity,
    AgentState,
    SensoryObservation,
    AgentAction,
    Environment,
    AgentController,
    ExperimentManifest,
    ExperimentResult,
    ResearchCheckpoint,
    ExperimentRunner,
    now_iso,
)

__all__ = [
    "PhysicalQuantity",
    "AgentIdentity",
    "AgentState",
    "SensoryObservation",
    "AgentAction",
    "Environment",
    "AgentController",
    "ExperimentManifest",
    "ExperimentResult",
    "ResearchCheckpoint",
    "ExperimentRunner",
    "now_iso",
]
