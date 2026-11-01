"""Interstellar expansion, colonies, and travel.

Uses distances, spacecraft speeds, energy budgets, travel time, and
communication delays. No instantaneous interstellar travel unless explicitly
operating in a hypothetical physics sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class StarSystem:
    sys_id: str
    name: str
    position: Tuple[float, float, float]  # ly
    star_type: str
    habitable: bool = True
    inhabited_by: Optional[str] = None


@dataclass
class Mission:
    mission_id: str
    origin_civ: str
    origin_system: str
    target_system: str
    mission_type: str  # exploration|colonization|trade|war
    launch_time: float
    arrival_time: float
    speed_c: float
    distance_ly: float
    population: float = 1000.0
    cargo: Dict[str, float] = field(default_factory=dict)
    active: bool = True
    arrived: bool = False


class GalacticMap:
    def __init__(self, seed: int = 0, n_systems: int = 200):
        self.rng = np.random.default_rng(seed)
        self.systems: Dict[str, StarSystem] = {}
        self.missions: Dict[str, Mission] = {}
        self._generate(seed, n_systems)

    def _generate(self, seed: int, n_systems: int):
        rng = np.random.default_rng(seed)
        types = ["M", "K", "G", "F", "A", "B", "O"]
        for i in range(n_systems):
            # Clustered distribution
            angle = rng.uniform(0, 2*np.pi)
            r = rng.exponential(15.0)
            x = r * np.cos(angle); y = r * np.sin(angle); z = rng.normal(0, 2.0)
            st = rng.choice(types, p=[0.7,0.15,0.08,0.04,0.02,0.01,0.0])
            self.systems[f"sys-{i}"] = StarSystem(
                sys_id=f"sys-{i}", name=f"System-{i}",
                position=(float(x), float(y), float(z)),
                star_type=st, habitable=(st in ["G", "K", "M"] and rng.random() < 0.2))

    def distance(self, a: str, b: str) -> float:
        sa = self.systems.get(a); sb = self.systems.get(b)
        if not sa or not sb:
            return float("inf")
        p1 = np.array(sa.position); p2 = np.array(sb.position)
        return float(np.linalg.norm(p1-p2))

    def launch_mission(self, civ, target_sys: str, mission_type: str,
                       mission_id: str, now: float) -> Optional[Mission]:
        if not civ.has_technology("rocketry"):
            return None
        home = civ.homeworld
        dist = self.distance(home, target_sys)
        if dist == float("inf"):
            return None
        speed_c = civ.max_ship_speed_c
        if speed_c <= 0:
            return None
        travel_years = dist / speed_c
        cost_energy = dist * 1e15  # energy cost scales with distance
        if civ.resources.energy_watts < cost_energy * 100:  # rough budget check
            return None
        civ.resources.energy_watts -= cost_energy * 10
        mission = Mission(mission_id=mission_id, origin_civ=civ.civ_id,
                          origin_system=home, target_system=target_sys,
                          mission_type=mission_type, launch_time=now,
                          arrival_time=now + travel_years / 1e9,  # years to Gyr
                          speed_c=speed_c, distance_ly=dist)
        self.missions[mission_id] = mission
        civ.history.append({"type": "mission_launched", "target": target_sys,
                            "mission": mission_id, "age": civ.age_gyr})
        return mission

    def step_missions(self, now: float) -> List[Mission]:
        arrived = []
        for m in self.missions.values():
            if m.active and not m.arrived and now >= m.arrival_time:
                m.arrived = True
                m.active = False
                arrived.append(m)
        return arrived

    def communicate(self, civ_a, civ_b) -> float:
        """Communication delay in time units (light travel)."""
        if not civ_a.homeworld or not civ_b.homeworld:
            return float("inf")
        dist = self.distance(civ_a.homeworld, civ_b.homeworld)
        return dist  # years per ly roughly, in same units
