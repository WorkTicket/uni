"""Black hole engineering - observational to highly speculative capabilities.

Account for gravitational effects, radiation, energy collection, waste heat,
operational distance, infrastructure costs. Label speculative engineering
appropriately. Do not implement unsupported faster-than-light travel or
black-hole universe creation as established physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class BlackHole:
    bh_id: str
    mass_solar: float = 10.0
    spin: float = 0.0  # Kerr parameter a/M
    charge: float = 0.0
    position: tuple = (0.0, 0.0, 0.0)
    has_accretion_disk: bool = False


@dataclass
class BlackHoleEngineering:
    """Black hole engineering capabilities with confidence levels."""
    # Observed/sound
    observation: bool = False
    orbital_stations: bool = False
    # Physically motivated but challenging
    accretion_collection: bool = False
    accretion_efficiency: float = 0.0  # fraction of rest mass
    # Speculative (GR-based)
    kerr_extraction: bool = False  # Penrose/Blandford-Znajek
    kerr_efficiency: float = 0.0
    # Highly speculative
    blackhole_computing: bool = False
    bh_computation_flops: float = 0.0
    # Not implemented as established
    universe_creation: bool = False  # hypothetical research only, not implemented

    def energy_output_watts(self, bh: BlackHole) -> float:
        if self.accretion_collection and bh.has_accretion_disk:
            # ~10% of rest mass for thin disk
            c2 = (3e8)**2
            mass_kg = bh.mass_solar * 2e30
            # order of magnitude
            return mass_kg * c2 * 0.1 * 1e-6
        if self.kerr_extraction and bh.spin > 0:
            return 1e40 * bh.spin  # highly speculative placeholder
        return 0.0

    def waste_heat_watts(self, bh: BlackHole) -> float:
        return self.energy_output_watts(bh) * 0.3

    def checkpoint(self) -> dict:
        return {"observation": self.observation, "orbital_stations": self.orbital_stations,
                "accretion_collection": self.accretion_collection, "accretion_efficiency": self.accretion_efficiency,
                "kerr_extraction": self.kerr_extraction, "kerr_efficiency": self.kerr_efficiency,
                "blackhole_computing": self.blackhole_computing, "bh_computation_flops": self.bh_computation_flops,
                "universe_creation": self.universe_creation}


class BlackHoleResearch:
    def __init__(self):
        self.bh_engineering: Dict[str, BlackHoleEngineering] = {}  # civ_id -> engineering

    def start_observation(self, civ_id: str, civ) -> bool:
        if not civ.has_technology("blackhole_observation"):
            return False
        if civ_id not in self.bh_engineering:
            self.bh_engineering[civ_id] = BlackHoleEngineering()
        self.bh_engineering[civ_id].observation = True
        return True

    def research_accretion(self, civ_id: str, civ) -> bool:
        if not civ.has_technology("accretion_energy"):
            return False
        eng = self.bh_engineering.get(civ_id)
        if not eng or not eng.observation:
            return False
        eng.accretion_collection = True
        eng.accretion_efficiency = 0.1
        return True

    def research_kerr(self, civ_id: str, civ) -> bool:
        if not civ.has_technology("kerr_extraction"):
            return False
        eng = self.bh_engineering.get(civ_id)
        if not eng or not eng.accretion_collection:
            return False
        eng.kerr_extraction = True
        eng.kerr_efficiency = 0.29  # theoretical maximum ~29%
        return True

    def research_bh_computing(self, civ_id: str, civ) -> bool:
        if not civ.has_technology("blackhole_computing"):
            return False
        eng = self.bh_engineering.get(civ_id)
        if not eng or not eng.kerr_extraction:
            return False
        eng.blackhole_computing = True
        eng.bh_computation_flops = civ.resources.energy_watts * 1e12
        civ.computational_capacity += eng.bh_computation_flops
        return True
