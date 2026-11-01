"""Technology tree with prerequisites, resource costs, and development.

Dependency-based research and engineering system. Each technology requires
materials, energy, knowledge, manufacturing capability, time and maintenance.
Support failed research, infrastructure loss, and abandoned projects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TechRequirement:
    knowledge: Dict[str, float] = field(default_factory=dict)
    energy_watts: float = 0.0
    metals: float = 0.0
    organics: float = 0.0
    volatiles: float = 0.0
    rare_elements: float = 0.0
    compute: float = 0.0
    time_years: float = 1.0


@dataclass
class Technology:
    tech_id: str
    name: str
    category: str  # energy|materials|compute|propulsion|biology|space|blackhole|ai
    description: str
    prerequisites: List[str] = field(default_factory=list)
    cost: TechRequirement = field(default_factory=TechRequirement)
    unlocks: List[str] = field(default_factory=list)
    risk: float = 0.1  # probability of failure
    speculative: bool = False


# Default technology tree
TECH_TREE: Dict[str, Technology] = {
    # Planetary (Type I foundations)
    "agriculture": Technology("agriculture","Agriculture","materials","Basic food production"),
    "metallurgy": Technology("metallurgy","Metallurgy","materials","Metal extraction and working", ["agriculture"]),
    "writing": Technology("writing","Writing & Records","compute","Information storage", ["agriculture"]),
    "electricity": Technology("electricity","Electricity","energy","Electrical power", ["metallurgy"]),
    "combustion": Technology("combustion","Combustion Engines","energy","Fossil fuel extraction", ["metallurgy"]),
    "electronics": Technology("electronics","Electronics","compute","Semiconductors", ["electricity"]),
    "computing": Technology("computing","Digital Computing","compute","Programmable computers", ["electronics"]),
    "nuclear_fission": Technology("nuclear_fission","Nuclear Fission","energy","Fission reactors", ["metallurgy", "electronics"]),
    "rocketry": Technology("rocketry","Rocketry","propulsion","Space access", ["combustion", "metallurgy"]),
    "satellites": Technology("satellites","Satellites","space","Orbital assets", ["rocketry", "electronics"]),
    "nuclear_fusion": Technology("nuclear_fusion","Nuclear Fusion","energy","Fusion power (clean baseload)", ["nuclear_fission", "plasma_physics"]),
    "plasma_physics": Technology("plasma_physics","Plasma Physics","energy","High-temperature plasma", ["nuclear_fission"]),
    # Stellar (Type II)
    "space_colonies": Technology("space_colonies","Space Colonies","space","Self-sufficient habitats", ["satellites", "life_support"]),
    "life_support": Technology("life_support","Life Support Systems","biology","Closed-loop life support", ["agriculture", "chemistry"]),
    "chemistry": Technology("chemistry","Advanced Chemistry","materials","Industrial chemistry", ["metallurgy"]),
    "asteroid_mining": Technology("asteroid_mining","Asteroid Mining","materials","In-situ resource extraction", ["space_colonies"]),
    "solar_arrays": Technology("solar_arrays","Large Solar Arrays","energy","Orbital solar", ["satellites"]),
    "ion_propulsion": Technology("ion_propulsion","Ion Propulsion","propulsion","Efficient interplanetary travel", ["electronics", "rocketry"]),
    "dyson_swarm": Technology("dyson_swarm","Dyson Swarm","energy","Star-scale energy collection", ["asteroid_mining", "solar_arrays", "autonomous_manufacturing"]),
    "autonomous_manufacturing": Technology("autonomous_manufacturing","Autonomous Manufacturing","materials","Self-replicating production", ["computing", "robotics"]),
    "robotics": Technology("robotics","Robotics","compute","Autonomous systems", ["computing", "electronics"]),
    "interstellar_propulsion": Technology("interstellar_propulsion","Interstellar Propulsion","propulsion","Slow interstellar travel", ["nuclear_fusion", "ion_propulsion"], risk=0.2),
    # Galactic (Type III) / Black holes / AI
    "ai": Technology("ai","Artificial Intelligence","ai","Advanced AI systems", ["computing", "neuroscience"]),
    "neuroscience": Technology("neuroscience","Neuroscience","biology","Brain modeling", ["biology"]),
    "biology": Technology("biology","Synthetic Biology","biology","Genetic engineering", ["chemistry"]),
    "blackhole_observation": Technology("blackhole_observation","Black Hole Observation","blackhole","Gravitational astronomy", ["astrophysics"]),
    "astrophysics": Technology("astrophysics","Astrophysics","energy","Stellar physics", ["rocketry"]),
    "accretion_energy": Technology("accretion_energy","Accretion Disk Energy","blackhole","Disk-based energy extraction", ["blackhole_observation", "nuclear_fusion"], risk=0.25, speculative=True),
    "kerr_extraction": Technology("kerr_extraction","Kerr Black Hole Extraction","blackhole","Rotational energy extraction (Penrose-like)", ["accretion_energy", "astrophysics"], risk=0.4, speculative=True),
    "blackhole_computing": Technology("blackhole_computing","Black Hole Computing","blackhole","Speculative BH computation", ["kerr_extraction", "ai"], risk=0.5, speculative=True),
    "galactic_network": Technology("galactic_network","Galactic Communication Network","compute","Sub-light comms across galaxy", ["interstellar_propulsion", "ai"]),
    "matrioshka_brain": Technology("matrioshka_brain","Matrioshka Brain","compute","Star-powered computing sphere", ["dyson_swarm", "ai", "autonomous_manufacturing"], risk=0.3, speculative=True),
    # Simulation research
    "simulation_theory": Technology("simulation_theory","Simulation Theory Research","compute","Investigating computability of reality", ["ai", "astrophysics"], risk=0.2),
    "nested_simulation": Technology("nested_simulation","Nested Simulation Engine","compute","Creating simulated universes", ["matrioshka_brain", "simulation_theory"], risk=0.35, speculative=True),
}


class TechnologyTree:
    def __init__(self):
        self.techs = dict(TECH_TREE)

    def get(self, tech_id: str) -> Optional[Technology]:
        return self.techs.get(tech_id)

    def available(self, civ) -> List[Technology]:
        out = []
        for tid, tech in self.techs.items():
            if civ.has_technology(tid):
                continue
            if all(civ.has_technology(p) for p in tech.prerequisites):
                out.append(tech)
        return out

    def research(self, civ, tech_id: str, rng) -> bool:
        tech = self.techs.get(tech_id)
        if not tech:
            return False
        if civ.has_technology(tech_id):
            return False
        if not all(civ.has_technology(p) for p in tech.prerequisites):
            return False
        # Resource check
        cost = tech.cost
        if civ.resources.energy_watts < cost.energy_watts:
            return False
        if civ.resources.metals < cost.metals:
            return False
        if civ.resources.rare_elements < cost.rare_elements:
            return False
        # Pay
        civ.resources.energy_watts -= cost.energy_watts
        civ.resources.metals -= cost.metals
        civ.resources.rare_elements -= cost.rare_elements
        # Risk
        if rng.random() < tech.risk:
            civ.history.append({"type": "research_failed", "tech": tech_id, "age": civ.age_gyr})
            return False
        civ.add_technology(tech_id)
        # Apply effects
        if tech_id == "nuclear_fission":
            civ.resources.energy_watts += 1e12
        elif tech_id == "nuclear_fusion":
            civ.resources.energy_watts += 1e15
        elif tech_id == "dyson_swarm":
            civ.infrastructure.dyson_swarm_fraction = min(1.0, civ.infrastructure.dyson_swarm_fraction + 0.1)
            civ.resources.energy_watts += 1e26
        elif tech_id == "computing":
            civ.computational_capacity *= 1000
        elif tech_id == "ai":
            civ.computational_capacity *= 10000
        elif tech_id == "matrioshka_brain":
            civ.infrastructure.matrioshka_brain_fraction = min(1.0, civ.infrastructure.matrioshka_brain_fraction + 0.1)
            civ.computational_capacity *= 1e12
        return True



class TechnologyTree:
    def __init__(self):
        self.techs = dict(TECH_TREE)

    def get(self, tech_id: str) -> Optional[Technology]:
        return self.techs.get(tech_id)

    def available(self, civ) -> List[Technology]:
        out = []
        for tid, tech in self.techs.items():
            if civ.has_technology(tid):
                continue
            if all(civ.has_technology(p) for p in tech.prerequisites):
                out.append(tech)
        return out

    def research(self, civ, tech_id: str, rng) -> bool:
        tech = self.techs.get(tech_id)
        if not tech:
            return False
        if civ.has_technology(tech_id):
            return False
        if not all(civ.has_technology(p) for p in tech.prerequisites):
            return False
        # Resource check
        cost = tech.cost
        if civ.resources.energy_watts < cost.energy_watts:
            return False
        if civ.resources.metals < cost.metals:
            return False
        if civ.resources.rare_elements < cost.rare_elements:
            return False
        # Pay
        civ.resources.energy_watts -= cost.energy_watts
        civ.resources.metals -= cost.metals
        civ.resources.rare_elements -= cost.rare_elements
        # Risk
        if rng.random() < tech.risk:
            civ.history.append({"type": "research_failed", "tech": tech_id, "age": civ.age_gyr})
            return False
        civ.add_technology(tech_id)
        # Apply effects
        if tech_id == "nuclear_fission":
            civ.resources.energy_watts += 1e12
        elif tech_id == "nuclear_fusion":
            civ.resources.energy_watts += 1e15
        elif tech_id == "dyson_swarm":
            civ.infrastructure.dyson_swarm_fraction = min(1.0, civ.infrastructure.dyson_swarm_fraction + 0.1)
            civ.resources.energy_watts += 1e26
        elif tech_id == "computing":
            civ.computational_capacity *= 1000
        elif tech_id == "ai":
            civ.computational_capacity *= 10000
        elif tech_id == "matrioshka_brain":
            civ.infrastructure.matrioshka_brain_fraction = min(1.0, civ.infrastructure.matrioshka_brain_fraction + 0.1)
            civ.computational_capacity *= 1e12
        return True
