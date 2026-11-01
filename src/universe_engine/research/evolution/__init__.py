"""Evolutionary architecture search (Research Program C6).

Allows candidate controllers and sensory/developmental traits to vary through
constrained evolutionary experiments. Maintains an archive of diverse
evolutionary lineages.
"""

from .experiment import (
    AgentGenome, AgentPhenotype, EvolvableAgent, LineageRecord,
    EvolutionArchive, EvolutionaryExperiment, make_controller, mutate_genome,
)

__all__ = [
    "AgentGenome", "AgentPhenotype", "EvolvableAgent", "LineageRecord",
    "EvolutionArchive", "EvolutionaryExperiment", "make_controller", "mutate_genome",
]
