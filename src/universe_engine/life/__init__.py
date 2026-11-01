"""Milestone E+: open-ended biological architecture.

Genome -> DevelopmentalProgram -> Phenotype -> Organism -> Population, with
lineage/species tracking, ecological networking, cognition, and an evolution
scheduler. Every layer carries declared assumptions and testable
responsibilities. No organism is conscious; behaviour is a modelled
perception-action loop, not evidence of subjective experience.
"""
from .genome import Genome, ALPHABET
from .genetics import Genetics, GeneticOperators, MutationEvent
from .development import DevelopmentalProgram, Phenotype, TraitValue
from .organism import Organism, OrganismState, MemoryEntry
from .population import Population, PopulationStats
from .environment import ChemicalResources, Patch, HabitatGrid, PlanetEnvironment, SECONDS_PER_GYR
from .metabolism import MetabolicNetwork, MetabolicFlux
from .ecology import EcologicalNetwork, Interaction
from .lineage import SpeciesLineage, LineageNode
from .cognition import CognitiveController, SensoryInput, CognitiveState
from .scheduler import EvolutionScheduler, Clock
from .history import LifeHistoryDatabase, OrganismRecord, PopulationRecord
from .simulation import MicrobialWorld, MicrobialWorldConfig, WorldStats

__all__ = [
    "Genome", "ALPHABET",
    "Genetics", "GeneticOperators", "MutationEvent",
    "DevelopmentalProgram", "Phenotype", "TraitValue",
    "Organism", "OrganismState", "MemoryEntry",
    "Population", "PopulationStats",
    "ChemicalResources", "Patch", "HabitatGrid", "PlanetEnvironment", "SECONDS_PER_GYR",
    "MetabolicNetwork", "MetabolicFlux",
    "EcologicalNetwork", "Interaction",
    "SpeciesLineage", "LineageNode",
    "CognitiveController", "SensoryInput", "CognitiveState",
    "EvolutionScheduler", "Clock",
    "LifeHistoryDatabase", "OrganismRecord", "PopulationRecord",
    "MicrobialWorld", "MicrobialWorldConfig", "WorldStats",
]
