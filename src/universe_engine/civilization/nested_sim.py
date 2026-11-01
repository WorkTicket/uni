"""Nested simulation engine - civilizations building simulations inside simulations.

An advanced civilization may allocate part of its available computational
infrastructure to running simulated worlds. Each simulation has parent universe
ID, parent civilization ID, simulation ID, model config, computational resource
budget, simulation clock, random seed, persistent state, fidelity level,
save/restore. Support simulations containing agents and civilizations that may
build further simulations. Enforce configurable depth, CPU, memory, storage
limits. Nested simulations are actual runnable models with resource budgets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid
import numpy as np


@dataclass
class SimulationConfig:
    model: str = "microbial"  # microbial|ecology|civilization
    fidelity: str = "low"     # low|medium|high
    max_depth: int = 3
    cpu_budget_flops: float = 1e12
    memory_gb: float = 1.0
    storage_gb: float = 1.0


@dataclass
class NestedSimulation:
    sim_id: str
    parent_universe_id: str
    parent_civ_id: str
    depth: int = 0  # recursion depth
    config: SimulationConfig = field(default_factory=SimulationConfig)
    seed: int = 0
    running: bool = False
    clock_t: float = 0.0
    cpu_allocated: float = 0.0
    memory_allocated: float = 0.0
    state: dict = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    created_at: float = 0.0
    blackhole_powered: bool = False  # speculative label

    def can_create_child(self) -> bool:
        if self.depth >= self.config.max_depth:
            return False
        if self.cpu_allocated < 1e9:
            return False
        return True

    def resource_usage(self) -> dict:
        return {"cpu_flops": self.cpu_allocated, "memory_gb": self.memory_allocated,
                "depth": self.depth, "children": len(self.children)}

    def checkpoint(self) -> dict:
        return {"sim_id": self.sim_id, "parent_universe_id": self.parent_universe_id,
                "parent_civ_id": self.parent_civ_id, "depth": self.depth,
                "config": {"model": self.config.model, "fidelity": self.config.fidelity,
                           "max_depth": self.config.max_depth,
                           "cpu_budget_flops": self.config.cpu_budget_flops,
                           "memory_gb": self.config.memory_gb, "storage_gb": self.config.storage_gb},
                "seed": self.seed, "running": self.running, "clock_t": self.clock_t,
                "cpu_allocated": self.cpu_allocated, "memory_allocated": self.memory_allocated,
                "state": self.state, "children": list(self.children),
                "created_at": self.created_at, "blackhole_powered": self.blackhole_powered}


class NestedSimulationEngine:
    def __init__(self, max_depth: int = 3):
        self.simulations: Dict[str, NestedSimulation] = {}
        self.max_depth = max_depth
        self.total_cpu: float = 0.0
        self.total_memory: float = 0.0

    def create_simulation(self, parent_civ, parent_universe_id: str,
                          config: Optional[SimulationConfig] = None,
                          depth: int = 0) -> Optional[NestedSimulation]:
        config = config or SimulationConfig()
        config.max_depth = min(config.max_depth, self.max_depth)
        if depth >= config.max_depth:
            return None
        # Resource budget from parent civilization
        if parent_civ.computational_capacity < 1e12:
            return None
        cpu_alloc = min(config.cpu_budget_flops, parent_civ.computational_capacity * 0.01)
        if cpu_alloc < 1e9:
            return None
        parent_civ.computational_capacity -= cpu_alloc
        parent_civ.resources.energy_watts -= cpu_alloc * 1e-6  # rough power

        sim_id = f"sim-{uuid.uuid4().hex[:12]}"
        sim = NestedSimulation(sim_id=sim_id, parent_universe_id=parent_universe_id,
                               parent_civ_id=parent_civ.civ_id, depth=depth,
                               config=config, seed=np.random.randint(0, 2**31),
                               cpu_allocated=cpu_alloc, memory_allocated=config.memory_gb,
                               created_at=0.0)
        self.simulations[sim_id] = sim
        parent_civ.simulations.append(sim_id)
        self.total_cpu += cpu_alloc
        self.total_memory += config.memory_gb
        return sim

    def create_child_simulation(self, parent_sim: NestedSimulation,
                                parent_civ, config: Optional[SimulationConfig] = None) -> Optional[NestedSimulation]:
        if not parent_sim.can_create_child():
            return None
        config = config or SimulationConfig()
        child_depth = parent_sim.depth + 1
        if child_depth >= self.max_depth:
            return None
        cpu_alloc = min(1e11, parent_sim.cpu_allocated * 0.1)
        if cpu_alloc < 1e6:
            return None
        sim_id = f"sim-{uuid.uuid4().hex[:12]}"
        sim = NestedSimulation(sim_id=sim_id, parent_universe_id=parent_sim.parent_universe_id,
                               parent_civ_id=parent_civ.civ_id, depth=child_depth,
                               config=config, seed=np.random.randint(0, 2**31),
                               cpu_allocated=cpu_alloc, memory_allocated=min(0.5, parent_sim.memory_allocated * 0.1),
                               created_at=parent_sim.clock_t)
        self.simulations[sim_id] = sim
        parent_sim.children.append(sim_id)
        self.total_cpu += cpu_alloc
        self.total_memory += sim.memory_allocated
        return sim

    def enter_simulation(self, sim_id: str) -> Optional[NestedSimulation]:
        return self.simulations.get(sim_id)

    def list_from_civ(self, civ_id: str) -> List[NestedSimulation]:
        return [s for s in self.simulations.values() if s.parent_civ_id == civ_id]

    def breadcrumb(self, sim_id: str) -> List[str]:
        """Return path from root to this simulation."""
        path = []
        current = self.simulations.get(sim_id)
        while current:
            path.append(current.sim_id)
            if current.depth == 0:
                break
            # Find parent simulation if any child
            parent = None
            for s in self.simulations.values():
                if sim_id in s.children:
                    parent = s; break
            current = parent
            if not parent:
                break
        return list(reversed(path))
