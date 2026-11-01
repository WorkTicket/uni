"""Genotype-to-phenotype developmental mapping.

A DevelopmentalProgram interprets a Genome into a Phenotype by applying
specified developmental rules to each functional module. Modules express traits
with weights given by regulatory marks. The mapping is many-to-one and
context-dependent: the same module can contribute differently depending on
neighbouring modules and regulatory state.

No module is "good" or "bad" by default; trait value and cost emerge from the
developmental rules and are evaluated against the environment by the fitness
model.

Provenance: developmental rules are declared inputs, not derived from molecular
biology. They are simplified, bounded abstractions of genotype-phenotype maps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from .genome import Genome, ALPHABET


@dataclass
class TraitValue:
    """A single phenotypic trait with expression level and maintenance cost."""
    name: str
    value: float          # expressed trait level (>= 0 typically)
    cost: float           # per-unit-time maintenance cost (energy/nutrient units)
    modules: Tuple[str, ...]  # source modules


@dataclass
class Phenotype:
    """The expressed organism structure: modular traits plus derived attributes."""
    traits: Dict[str, TraitValue] = field(default_factory=dict)
    size: float = 1.0             # effective cell/organism size (volume units)
    motility: float = 0.0         # movement capability
    sensing_range: float = 0.0    # sensory radius
    reproduction_rate: float = 0.0
    defence: float = 0.0
    metabolic_power: float = 0.0  # energy acquisition potential
    structure: Dict[str, float] = field(default_factory=dict)  # arbitrary structural descriptors
    developmental_cost: float = 0.0  # one-time cost to build this phenotype

    def trait(self, name: str) -> Optional[TraitValue]:
        return self.traits.get(name)

    def total_maintenance_cost(self) -> float:
        return sum(t.cost for t in self.traits.values())

    def module_composition(self) -> Dict[str, int]:
        comp: Dict[str, int] = {}
        for t in self.traits.values():
            for m in t.modules:
                comp[m] = comp.get(m, 0) + 1
        return comp


@dataclass
class DevelopmentalProgram:
    """Maps a Genome to a Phenotype via declared module->trait rules.

    Parameters
    ----------
    module_traits:
        module symbol -> list of (trait_name, value_per_module, cost_per_module).
    base_size:
        baseline organism size per module.
    regulatory_gain:
        how strongly regulatory marks scale trait expression (0..1).
    """

    module_traits: Dict[str, List[Tuple[str, float, float]]] = field(default_factory=dict)
    base_size: float = 0.5
    regulatory_gain: float = 0.5

    def __post_init__(self) -> None:
        if not self.module_traits:
            self._set_defaults()

    def _set_defaults(self) -> None:
        """Default module->trait rules (illustrative, declared assumptions)."""
        self.module_traits = {
            "A": [("metabolism", 1.0, 0.1)],            # baseline energy harvesting
            "C": [("photosynthesis", 1.2, 0.15)],
            "G": [("chemotrophy", 1.0, 0.12)],
            "T": [("structure", 1.0, 0.08)],
            "M": [("transport", 0.9, 0.07)],
            "S": [("sensing", 1.0, 0.06)],
            "L": [("motility", 1.0, 0.1)],
            "D": [("defence", 1.0, 0.09)],
            "R": [("reproduction", 3.0, 0.05)],   # reproduction: higher propensity
            "N": [("signalling", 1.0, 0.11)],
            "X": [("regulation", 0.5, 0.03)],
            "Y": [("transfer_reception", 0.4, 0.02)],
            "Z": [],  # neutral spacer
        }
        self._defaults = True

    def develop(self, genome: Genome) -> Phenotype:
        """Develop a Phenotype from a Genome using module traits and regulation."""
        traits: Dict[str, TraitValue] = {}
        size = self.base_size
        motility = 0.0
        sensing = 0.0
        reproduction = 0.0
        defence = 0.0
        metabolic_power = 0.0
        structure: Dict[str, float] = {}
        dev_cost = 0.0

        n = genome.length
        for i, mod in enumerate(genome.modules):
            weight = genome.regulatory.get(i, 1.0)
            # regulatory attenuation: expression ~ weight^regulatory_gain
            expr = weight ** self.regulatory_gain
            for tname, val_per, cost_per in self.module_traits.get(mod, []):
                value = val_per * expr
                cost = cost_per * expr
                if tname in traits:
                    old = traits[tname]
                    traits[tname] = TraitValue(tname, old.value + value,
                                               old.cost + cost, old.modules + (mod,))
                else:
                    traits[tname] = TraitValue(tname, value, cost, (mod,))
            dev_cost += 0.05 * expr  # per-module developmental overhead
            size += 0.1 * expr
            if "motility" in [t[0] for t in self.module_traits.get(mod, [])]:
                motility += 1.0 * expr
            if "sensing" in [t[0] for t in self.module_traits.get(mod, [])]:
                sensing += 1.0 * expr
            if "reproduction" in [t[0] for t in self.module_traits.get(mod, [])]:
                reproduction += 1.0 * expr
            if "defence" in [t[0] for t in self.module_traits.get(mod, [])]:
                defence += 1.0 * expr
            if "metabolism" in [t[0] for t in self.module_traits.get(mod, [])] or \
               "photosynthesis" in [t[0] for t in self.module_traits.get(mod, [])] or \
               "chemotrophy" in [t[0] for t in self.module_traits.get(mod, [])]:
                metabolic_power += 1.0 * expr
            if "structure" in [t[0] for t in self.module_traits.get(mod, [])]:
                structure["support"] = structure.get("support", 0.0) + 1.0 * expr

        phen = Phenotype(
            traits=traits,
            size=size,
            motility=motility,
            sensing_range=sensing,
            reproduction_rate=reproduction,
            defence=defence,
            metabolic_power=metabolic_power,
            structure=structure,
            developmental_cost=dev_cost,
        )
        return phen
