"""Civilization Evolution Engine - integrates all civilization systems.

Manages multiple civilizations, their development, expansion, technology,
and nested simulations. Resource-limited, no scripted timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .civilization_state import Civilization, ResourcePool, Territory, Infrastructure, SocialOrganization
from .technology import TechnologyTree
from .expansion import GalacticMap, Mission, StarSystem
from .nested_sim import NestedSimulationEngine, SimulationConfig


@dataclass
class CivilizationEvolutionEngine:
    rng: np.random.Generator
    tech_tree: TechnologyTree = field(default_factory=TechnologyTree)
    galactic_map: GalacticMap = field(default_factory=GalacticMap)
    nested_engine: NestedSimulationEngine = field(default_factory=NestedSimulationEngine)
    civilizations: Dict[str, Civilization] = field(default_factory=dict)
    t_gyr: float = 0.0
    next_civ_id: int = 0
    next_mission_id: int = 0

    def _new_civ_id(self) -> str:
        n = self.next_civ_id; self.next_civ_id += 1
        return f"civ-{n}"

    def _new_mission_id(self) -> str:
        n = self.next_mission_id; self.next_mission_id += 1
        return f"mis-{n}"

    def found_civilization(self, species_id: str, homeworld: str,
                           name: Optional[str] = None) -> Civilization:
        """Found a new civilization on a planet (when prerequisites met)."""
        cid = self._new_civ_id()
        name = name or f"Civilization-{cid}"
        civ = Civilization(civ_id=cid, name=name, species_id=species_id,
                           homeworld=homeworld, founded_t_gyr=self.t_gyr)
        civ.territory.homeworld = homeworld
        civ.territory.systems = [homeworld]
        civ.resources = ResourcePool(energy_watts=1e9)  # early industrial
        civ.population = 1e6
        # Grant basic technologies
        civ.add_technology("agriculture")
        civ.add_technology("metallurgy")
        civ.history.append({"type": "founded", "age": self.t_gyr})
        self.civilizations[cid] = civ
        if homeworld in self.galactic_map.systems:
            self.galactic_map.systems[homeworld].inhabited_by = cid
        return civ

    def step(self, dt_gyr: float) -> List[dict]:
        events = []
        self.t_gyr += dt_gyr
        for civ in list(self.civilizations.values()):
            if not civ.active or civ.collapsed:
                continue
            civ.age_gyr = self.t_gyr - civ.founded_t_gyr
            # Population growth
            growth = 0.01 * civ.social.stability * (1.0 - civ.population / 1e12)
            civ.population *= (1.0 + growth * dt_gyr * 1e9)
            # Energy production growth from technology
            if civ.has_technology("electricity"):
                civ.resources.energy_watts *= (1.0 + 0.05 * dt_gyr * 1e9)
            if civ.has_technology("nuclear_fission"):
                civ.resources.energy_watts *= (1.0 + 0.1 * dt_gyr * 1e9)
            if civ.has_technology("nuclear_fusion"):
                civ.resources.energy_watts *= (1.0 + 0.2 * dt_gyr * 1e9)
            # Research attempt
            available = self.tech_tree.available(civ)
            if available and self.rng.random() < 0.1:
                tech = self.rng.choice(available)
                if self.tech_tree.research(civ, tech.tech_id, self.rng):
                    events.append({"type": "tech_research", "civ": civ.civ_id,
                                   "tech": tech.tech_id})
                else:
                    events.append({"type": "research_failed", "civ": civ.civ_id,
                                   "tech": tech.tech_id})
            # Expansion attempt
            if civ.has_technology("rocketry") and civ.expansion_drive > 0.3 and self.rng.random() < 0.02:
                # Find uninhabited habitable system
                for sid, sys in self.galactic_map.systems.items():
                    if sys.habitable and sys.inhabited_by is None and sid != civ.homeworld:
                        mis = self.galactic_map.launch_mission(civ, sid, "colonization",
                                                               self._new_mission_id(), self.t_gyr)
                        if mis:
                            events.append({"type": "expansion", "civ": civ.civ_id, "target": sid})
                        break
            # Collapse check
            if civ.social.stability < 0.1 and self.rng.random() < 0.01:
                civ.collapsed = True
                events.append({"type": "collapse", "civ": civ.civ_id})

        # Process missions
        arrived = self.galactic_map.step_missions(self.t_gyr)
        for m in arrived:
            if m.mission_type == "colonization" and m.target_system in self.galactic_map.systems:
                sys = self.galactic_map.systems[m.target_system]
                if sys.inhabited_by is None:
                    sys.inhabited_by = m.origin_civ
                    civ = self.civilizations.get(m.origin_civ)
                    if civ:
                        civ.territory.colonies.append(m.target_system)
                        civ.territory.systems.append(m.target_system)
                        events.append({"type": "colony_founded", "civ": m.origin_civ,
                                       "system": m.target_system})

        return events

    def create_simulation_for_civ(self, civ_id: str, config: Optional[SimulationConfig] = None) -> Optional[object]:
        civ = self.civilizations.get(civ_id)
        if not civ:
            return None
        if not civ.has_technology("nested_simulation") and not civ.has_technology("simulation_theory"):
            return None
        return self.nested_engine.create_simulation(civ, "root", config)

    def checkpoint(self) -> dict:
        return {
            "t_gyr": self.t_gyr,
            "civilizations": {k: v.checkpoint() for k, v in self.civilizations.items()},
            "galactic_map": {"systems": {k: {"sys_id": s.sys_id, "name": s.name,
                                             "position": s.position, "star_type": s.star_type,
                                             "habitable": s.habitable, "inhabited_by": s.inhabited_by}
                                        for k, s in self.galactic_map.systems.items()},
                              "missions": {k: {"mission_id": m.mission_id, "origin_civ": m.origin_civ,
                                               "origin_system": m.origin_system, "target_system": m.target_system,
                                               "mission_type": m.mission_type, "launch_time": m.launch_time,
                                               "arrival_time": m.arrival_time, "arrived": m.arrived}
                                           for k, m in self.galactic_map.missions.items()}},
            "nested_simulations": {k: s.checkpoint() for k, s in self.nested_engine.simulations.items()},
        }

    @classmethod
    def restore(cls, data: dict, rng: np.random.Generator) -> "CivilizationEvolutionEngine":
        eng = cls(rng=rng)
        eng.t_gyr = float(data.get("t_gyr", 0.0))
        for k, cd in data.get("civilizations", {}).items():
            eng.civilizations[k] = Civilization.restore(cd)
        gm = data.get("galactic_map", {})
        for k, sd in gm.get("systems", {}).items():
            eng.galactic_map.systems[k] = StarSystem(sys_id=sd["sys_id"], name=sd["name"],
                                                     position=tuple(sd["position"]),
                                                     star_type=sd["star_type"],
                                                     habitable=bool(sd.get("habitable", True)),
                                                     inhabited_by=sd.get("inhabited_by"))
        for k, md in gm.get("missions", {}).items():
            m = Mission(mission_id=md["mission_id"], origin_civ=md["origin_civ"],
                        origin_system=md["origin_system"], target_system=md["target_system"],
                        mission_type=md["mission_type"], launch_time=float(md["launch_time"]),
                        arrival_time=float(md["arrival_time"]), speed_c=0.01,
                        distance_ly=0.0, arrived=bool(md.get("arrived", False)),
                        active=not bool(md.get("arrived", False)))
            eng.galactic_map.missions[k] = m
        eng.next_civ_id = len(eng.civilizations)
        eng.next_mission_id = len(eng.galactic_map.missions)
        return eng
