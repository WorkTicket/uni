"""Scientific discovery laboratory (Research Program F).

Agents operate physical experiment apparatus and discover scientific models
from data. No agent is supplied the true equations -- it must infer them
from observations.
"""

from .experiment import (
    BellApparatus, CorrelationModel, ClassicalLocalModel, QuantumModel,
    ConstantModel, DiscoveryResult, run_discovery_experiment,
)

__all__ = [
    "BellApparatus", "CorrelationModel", "ClassicalLocalModel",
    "QuantumModel", "ConstantModel", "DiscoveryResult",
    "run_discovery_experiment",
]
