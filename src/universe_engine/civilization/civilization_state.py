"""Civilization state and Kardashev scale.

Every technological civilization has persistent identity, population, territory,
scientific knowledge, energy production, resources, industry, infrastructure,
communication capacity, computational capacity, political/social organization
and technological history. Multiple independent civilizations can coexist.

Kardashev index: K = (log10(P) - 6) / 10 where P is energy use in watts.
This is a measurement, not a technology unlock button.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .kardashev import kardashev_index, kardashev_type


@dataclass
class ResourcePool:
    """Material and energy resources of a civilization."""
    energy_watts: float = 0.0        # total energy production capacity
    metals: float = 0.0
    organics: float = 0.0
    volatiles: float = 0.0
    rare_elements: float = 0.0
    computing_flops: float = 0.0

    def total_value(self) -> float:
        return self.metals + self.organics + self.volatiles + self.rare_elements


@dataclass
class Territory:
    """Territorial control."""
    homeworld: Optional[str] = None
    colonies: List[str] = field(default_factory=list)
    systems: List[str] = field(default_factory=list)
    outposts: List[str] = field(default_factory=list)
    megastructures: List[str] = field(default_factory=list)


@dataclass
class Infrastructure:
    """Built infrastructure."""
    orbital: float = 0.0
    surface: float = 0.0
    industrial: float = 0.0
    power_plants: float = 0.0
    compute_facilities: float = 0.0
    dyson_swarm_fraction: float = 0.0  # 0-1 of star
    matrioshka_brain_fraction: float = 0.0


@dataclass
class SocialOrganization:
    """Political/social structure."""
    name: str = "tribal"
    cohesion: float = 0.5
    stability: float = 0.5
    ideology: str = "pragmatic"
    governance: str = "decentralized"


@dataclass
class Civilization:
    """A persistent technological civilization."""
    civ_id: str
    name: str
    species_id: str
    homeworld: str
    founded_t_gyr: float = 0.0
    age_gyr: float = 0.0

    population: float = 1000.0
    territory: Territory = field(default_factory=Territory)
    resources: ResourcePool = field(default_factory=ResourcePool)
    infrastructure: Infrastructure = field(default_factory=Infrastructure)
    social: SocialOrganization = field(default_factory=SocialOrganization)

    # Knowledge and tech
    scientific_knowledge: Dict[str, float] = field(default_factory=dict)
    technologies: Dict[str, bool] = field(default_factory=dict)
    research_projects: List[dict] = field(default_factory=list)

    # Capabilities
    communication_range_ly: float = 0.01
    max_ship_speed_c: float = 0.01
    computational_capacity: float = 1e12  # FLOPS equivalent

    # History and state
    history: List[dict] = field(default_factory=list)
    active: bool = True
    collapsed: bool = False
    expansion_drive: float = 0.5  # 0=isolated, 1=expansionist
    sustainability_focus: float = 0.5

    # Nested simulations
    simulations: List[str] = field(default_factory=list)

    def kardashev_k(self) -> float:
        return kardashev_index(self.resources.energy_watts)

    def kardashev_type_str(self) -> str:
        return kardashev_type(self.kardashev_k())

    def energy_per_capita(self) -> float:
        if self.population <= 0:
            return 0.0
        return self.resources.energy_watts / self.population

    def add_technology(self, tech_id: str) -> None:
        if not self.technologies.get(tech_id):
            self.technologies[tech_id] = True
            self.history.append({"type": "technology", "tech": tech_id, "age": self.age_gyr})

    def has_technology(self, tech_id: str) -> bool:
        return bool(self.technologies.get(tech_id))

    def checkpoint(self) -> dict:
        return {
            "civ_id": self.civ_id, "name": self.name, "species_id": self.species_id,
            "homeworld": self.homeworld, "founded_t_gyr": self.founded_t_gyr,
            "age_gyr": self.age_gyr, "population": self.population,
            "territory": {"homeworld": self.territory.homeworld,
                          "colonies": list(self.territory.colonies),
                          "systems": list(self.territory.systems),
                          "outposts": list(self.territory.outposts),
                          "megastructures": list(self.territory.megastructures)},
            "resources": {"energy_watts": self.resources.energy_watts,
                          "metals": self.resources.metals, "organics": self.resources.organics,
                          "volatiles": self.resources.volatiles,
                          "rare_elements": self.resources.rare_elements,
                          "computing_flops": self.resources.computing_flops},
            "infrastructure": {"orbital": self.infrastructure.orbital,
                               "surface": self.infrastructure.surface,
                               "industrial": self.infrastructure.industrial,
                               "power_plants": self.infrastructure.power_plants,
                               "compute_facilities": self.infrastructure.compute_facilities,
                               "dyson_swarm_fraction": self.infrastructure.dyson_swarm_fraction,
                               "matrioshka_brain_fraction": self.infrastructure.matrioshka_brain_fraction},
            "social": {"name": self.social.name, "cohesion": self.social.cohesion,
                       "stability": self.social.stability, "ideology": self.social.ideology,
                       "governance": self.social.governance},
            "scientific_knowledge": dict(self.scientific_knowledge),
            "technologies": dict(self.technologies),
            "research_projects": list(self.research_projects),
            "communication_range_ly": self.communication_range_ly,
            "max_ship_speed_c": self.max_ship_speed_c,
            "computational_capacity": self.computational_capacity,
            "history": list(self.history),
            "active": self.active, "collapsed": self.collapsed,
            "expansion_drive": self.expansion_drive,
            "sustainability_focus": self.sustainability_focus,
            "simulations": list(self.simulations),
        }

    @classmethod
    def restore(cls, data: dict) -> "Civilization":
        civ = cls(civ_id=data["civ_id"], name=data["name"],
                  species_id=data["species_id"], homeworld=data["homeworld"],
                  founded_t_gyr=float(data.get("founded_t_gyr", 0.0)),
                  age_gyr=float(data.get("age_gyr", 0.0)),
                  population=float(data.get("population", 1000.0)))
        t = data.get("territory", {})
        civ.territory = Territory(homeworld=t.get("homeworld"), colonies=list(t.get("colonies",[])),
                                  systems=list(t.get("systems",[])), outposts=list(t.get("outposts",[])),
                                  megastructures=list(t.get("megastructures",[])))
        r = data.get("resources", {})
        civ.resources = ResourcePool(energy_watts=float(r.get("energy_watts",0.0)),
                                    metals=float(r.get("metals",0.0)), organics=float(r.get("organics",0.0)),
                                    volatiles=float(r.get("volatiles",0.0)),
                                    rare_elements=float(r.get("rare_elements",0.0)),
                                    computing_flops=float(r.get("computing_flops",0.0)))
        inf = data.get("infrastructure", {})
        civ.infrastructure = Infrastructure(
            orbital=float(inf.get("orbital",0.0)), surface=float(inf.get("surface",0.0)),
            industrial=float(inf.get("industrial",0.0)), power_plants=float(inf.get("power_plants",0.0)),
            compute_facilities=float(inf.get("compute_facilities",0.0)),
            dyson_swarm_fraction=float(inf.get("dyson_swarm_fraction",0.0)),
            matrioshka_brain_fraction=float(inf.get("matrioshka_brain_fraction",0.0)))
        s = data.get("social", {})
        civ.social = SocialOrganization(name=s.get("name","tribal"), cohesion=float(s.get("cohesion",0.5)),
                                        stability=float(s.get("stability",0.5)),
                                        ideology=s.get("ideology","pragmatic"),
                                        governance=s.get("governance","decentralized"))
        civ.scientific_knowledge = dict(data.get("scientific_knowledge",{}))
        civ.technologies = dict(data.get("technologies",{}))
        civ.research_projects = list(data.get("research_projects",[]))
        civ.communication_range_ly = float(data.get("communication_range_ly",0.01))
        civ.max_ship_speed_c = float(data.get("max_ship_speed_c",0.01))
        civ.computational_capacity = float(data.get("computational_capacity",1e12))
        civ.history = list(data.get("history",[]))
        civ.active = bool(data.get("active",True))
        civ.collapsed = bool(data.get("collapsed",False))
        civ.expansion_drive = float(data.get("expansion_drive",0.5))
        civ.sustainability_focus = float(data.get("sustainability_focus",0.5))
        civ.simulations = list(data.get("simulations",[]))
        return civ
