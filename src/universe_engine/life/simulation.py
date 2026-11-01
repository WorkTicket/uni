"""MicrobialWorld: integrated resource-limited microbial life simulation.

Couples the existing Milestone E habitat (planet environment, chemistry,
protocells, replicators) with the new biological architecture (genome,
developmental phenotype, organism, population, lineage, metabolism, ecology).

Each organism is a persistent agent with a genome, phenotype, energy state and
location. Organisms consume actual chemical resources from their habitat patch
via metabolism; growth and reproduction require material and energy; death
returns material to the resource pool through decomposition. No infinite food,
no free reproduction.

A sterile world (no seeded organisms) remains sterile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .genome import Genome, ALPHABET
from .genetics import Genetics, GeneticOperators
from .development import DevelopmentalProgram
from .organism import Organism, OrganismState
from .population import Population
from .environment import ChemicalResources, Patch, HabitatGrid, PlanetEnvironment
from .metabolism import MetabolicNetwork
from .ecology import EcologicalNetwork, Interaction
from .lineage import SpeciesLineage
from .cognition import CognitiveController, SensoryInput, CognitiveState
from .scheduler import EvolutionScheduler
from .history import LifeHistoryDatabase


@dataclass
class MicrobialWorldConfig:
    """Configuration for a MicrobialWorld run."""
    n_patches: int = 8
    patch_spacing_m: float = 1e6
    seed: int = 0
    n_founders: int = 20
    founder_genome_len: int = 8
    initial_pool: float = 1000.0      # shared free nutrient units
    nutrient_per_patch_mol: float = 10.0
    dt_gyr: float = 1e-4              # biological step in Gyr
    genetics: Genetics = field(default_factory=Genetics)
    cognition_architecture: str = "stimulus_response"


@dataclass
class WorldStats:
    t_gyr: float
    n_organisms: int
    n_species: int
    n_lineages: int
    total_energy: float
    total_nutrients: float
    resource_pool: float
    births: int
    deaths: int
    diversity: dict


class MicrobialWorld:
    """A patch-based microbial world with resource-limited evolution."""

    def __init__(self, config: Optional[MicrobialWorldConfig] = None,
                 planet_env: Optional[PlanetEnvironment] = None):
        self.cfg = config or MicrobialWorldConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.genetics = self.cfg.genetics
        self.ops = GeneticOperators(self.rng, self.genetics)
        self.development = DevelopmentalProgram()
        self.lineage = SpeciesLineage()
        self.metabolism = MetabolicNetwork()
        self.history = LifeHistoryDatabase(metadata={"seed": self.cfg.seed,
                                                    "config": "MicrobialWorld v1"})
        self.scheduler = EvolutionScheduler()

        # ---- environment ---------------------------------------------------
        self.env = planet_env or PlanetEnvironment(t_gyr=0.0,
                                                   global_temperature_K=288.0,
                                                   liquid_water_fraction=1.0,
                                                   ph_ocean=7.0)
        self._build_grid()

        # ---- population ----------------------------------------------------
        self.population = Population(resource_pool=self.cfg.initial_pool,
                                     developmental=self.development,
                                     genetics=self.genetics, rng=self.rng)

        # ---- ecology -------------------------------------------------------
        # Aggregate chemical resources for the whole world (shared reservoir).
        names = ["W", "N", "M", "Wst"]
        init_conc = np.zeros(len(names))
        # N = nutrient at concentration such that total = nutrient_per_patch_mol * n_patches
        total_nutrient = self.cfg.nutrient_per_patch_mol * self.cfg.n_patches
        init_conc[1] = total_nutrient / 1.0  # species N, volume 1 L
        self.world_resources = ChemicalResources(species_names=names,
                                                 concentrations=init_conc,
                                                 nutrient_mol=total_nutrient,
                                                 volume_L=1.0)
        self.ecology = EcologicalNetwork(resources=self.world_resources)
        self.ecology.add_population("microbes", self.population)

        # ---- seeded founders ----------------------------------------------
        self._seed_founders()
        self.t_gyr = 0.0

    # ---- setup -------------------------------------------------------------
    def _build_grid(self) -> None:
        patches: List[Patch] = []
        for i in range(self.cfg.n_patches):
            loc = tuple(float(i * self.cfg.patch_spacing_m) for _ in (0, 1))
            res = ChemicalResources(
                species_names=["W", "N", "M", "Wst"],
                concentrations=np.array([0.0,
                                         self.cfg.nutrient_per_patch_mol,
                                         0.0, 0.0], dtype=float),
                nutrient_mol=self.cfg.nutrient_per_patch_mol,
                volume_L=1.0,
            )
            patches.append(Patch(index=i, location=loc, resources=res,
                                 temperature_K=self.env.global_temperature_K,
                                 pH=self.env.ph_ocean,
                                 liquid_water=True))
        self.env.grid = HabitatGrid(patches=patches,
                                    dims=(self.cfg.n_patches,),
                                    spacing_m=self.cfg.patch_spacing_m)

    def _random_genome(self) -> Genome:
        """Draw a viable founder genome: metabolism + reproduction core, plus
        random functional modules. Mutation then explores from this viable base.

        Declared modeling choice: founders are viable (can harvest energy and
        reproduce); their descendants explore trait space via mutation.
        """
        base = ["A", "R"]  # metabolism core + reproduction
        extra = list(self.rng.choice(ALPHABET, size=self.rng.integers(0, 6)).tolist())
        modules = tuple(base + extra)
        reg = {i: float(self.rng.uniform(0.5, 1.0)) for i in range(len(modules))
               if self.rng.random() < 0.5}
        return Genome(modules=modules, regulatory=reg,
                      metadata={"biochemistry": "carbon-water", "source": "founder seed"})

    def _seed_founders(self) -> None:
        for _ in range(self.cfg.n_founders):
            g = self._random_genome()
            phen = self.development.develop(g)
            gid = self.lineage.register_genome(g, None, 0.0, 0)
            species = self.lineage.assign_species(f"founder-{gid}", g, 0, 0.0)
            org = Organism(identity=self.population._new_id(),
                           genome=g, phenotype=phen,
                           state=OrganismState(
                               energy=phen.developmental_cost,
                               nutrients=1.0),
                           location=(0.0, 0.0),
                           generation=0, species_id=species)
            self.population.add(org)
        # Rebuild next_id so founder ids are consistent (founders used _new_id).
        # Note: _new_id increments; fine.

    # ---- one step ----------------------------------------------------------
    def step(self, dt_gyr: Optional[float] = None) -> dict:
        """Advance the world by dt_gyr (Gigayears). Returns events."""
        dt = dt_gyr if dt_gyr is not None else self.cfg.dt_gyr
        events: List[dict] = []

        # 1. Metabolism: each organism draws from its patch + world resources.
        for org in list(self.population.organisms.values()):
            if not org.state.alive:
                continue
            patch = self.env.grid.nearest(org.location)
            flux = self.metabolism.update(org, patch.resources, dt,
                                          light=self.env.light,
                                          temperature_K=patch.temperature_K)
            # Return waste to world resource pool.
            self.world_resources.add_mol("Wst", flux.waste_produced * 1e-6)
            if flux.net_energy < 0:
                org.state.stress = min(1.0, org.state.stress + 0.01 * dt)

        # 2. Reproduction: organisms with enough energy reproduce (with mutation).
        for org in list(self.population.organisms.values()):
            if not org.state.alive:
                continue
            if not org.can_reproduce():
                continue
            # Reproduction probability per step: reproduction_rate (per-Gyr) * dt.
            prob = org.phenotype.reproduction_rate * dt
            if self.rng.random() > prob:
                continue
            child, evs = self.population.reproduce(org, dt,
                                                   energy_cost_share=0.5)
            events.extend(evs)
            if child is not None:
                # mutate genome further via operators (already done in reproduce)
                phen = self.development.develop(child.genome)
                gid = self.lineage.register_genome(child.genome, org.genome,
                                                   self.t_gyr, 1)
                species = self.lineage.assign_species(child.identity, child.genome,
                                                      child.generation, self.t_gyr)
                child.species_id = species
                child.phenotype = phen
                self.population.resource_pool -= 0.0  # already charged

        # 3. Population maintenance + death (starvation/senescence).
        death_events = self.population.step(dt, energy_income=0.0)
        events.extend(death_events)

        # 4. Ecology interactions (placeholder: competition among microbes).
        self.ecology.step(dt, now=self.t_gyr)

        # 5. Record history snapshot.
        stats = self.population.stats()
        self.history.record_population(self.t_gyr, "microbes", self.population)
        for org in self.population.alive():
            self.history.record_organism(self.t_gyr, org)

        self.t_gyr += dt
        self.scheduler.step_all()

        return {"t_gyr": self.t_gyr, "events": events, "stats": self.population.stats()}

    # ---- queries -----------------------------------------------------------
    def stats(self) -> WorldStats:
        ps = self.population.stats()
        return WorldStats(t_gyr=self.t_gyr, n_organisms=ps.size,
                          n_species=ps.n_species, n_lineages=ps.n_lineages,
                          total_energy=ps.total_energy, total_nutrients=ps.total_nutrients,
                          resource_pool=self.population.resource_pool,
                          births=ps.births, deaths=ps.deaths,
                          diversity=self.history.diversity())

    def is_sterile(self) -> bool:
        return self.population.size() == 0

    def total_resource_mol(self) -> float:
        return self.world_resources.total_mol()

    def checkpoint(self) -> dict:
        return {
            "schema": "microbial-world/v1",
            "t_gyr": self.t_gyr,
            "config": {"n_patches": self.cfg.n_patches,
                       "seed": self.cfg.seed,
                       "n_founders": self.cfg.n_founders,
                       "dt_gyr": self.cfg.dt_gyr},
            "population": self.population.checkpoint(),
            "lineage": self.lineage.checkpoint(),
            "world_resources": self.world_resources.checkpoint(),
            "environment": self.env.checkpoint(),
            "scheduler": self.scheduler.checkpoint(),
        }

    @classmethod
    def restore(cls, data: dict) -> "MicrobialWorld":
        cfg = MicrobialWorldConfig(n_patches=int(data["config"]["n_patches"]),
                                   seed=int(data["config"]["seed"]),
                                   n_founders=int(data["config"]["n_founders"]),
                                   dt_gyr=float(data["config"]["dt_gyr"]))
        world = cls(config=cfg,
                    planet_env=PlanetEnvironment.restore(data["environment"]))
        world.t_gyr = float(data["t_gyr"])
        world.population = Population.restore(data["population"], world.development,
                                              world.genetics, world.cfg.seed)
        world.lineage = SpeciesLineage.restore(data["lineage"])
        world.world_resources = ChemicalResources.restore(data["world_resources"])
        world.scheduler = EvolutionScheduler.restore(data["scheduler"])
        world.ecology = EcologicalNetwork(resources=world.world_resources)
        world.ecology.add_population("microbes", world.population)
        return world
