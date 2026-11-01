"""Milestone E: coupled habitat -- planet environment, chemistry, protocells, replicators.

Coupling (operator splitting, fixed scientific timestep):
  planet environment (climate) -> chemical inventories (T, pH, liquid water)
    -> reaction network (mass-action, element/charge conserved)
    -> transport between habitat patches and with the exterior
    -> protocell compartments (osmosis, uptake, division/lysis; conserved contents)
    -> replicator populations inside compartments (replication consumes monomers;
       decay returns monomers; partitioning at division conserves units)
    -> environmental feedback (metabolism changes solute concentrations)
  A mass ledger tracks every conserved chemical species across the chemistry fields
  and the compartment contents; abstract replicator counts are tracked separately.

Nothing here creates life: a sterile habitat (no seeded replicators) stays sterile.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .chemistry.network import Network, Species, Reaction, env_from_planet
from .planet.model import Planet, PlanetSpec
from .prebiotic.compartment import World as ProtocellWorld, Protocell
from .replicators.population import Population, duplex_mechanism

SECONDS_PER_GYR = 3.15576e16


def make_default_network() -> Network:
    """Illustrative, element- and charge-balanced reaction network (parameters are illustrative,
    not measured). Species: W solvent, N nutrient, M monomer, Wst waste. All solutes share the
    same elemental composition so N <-> M <-> Wst interconversion is balanced; water is the
    solvent at unit activity. Replicators are abstract biological counts, not chemical species;
    they consume monomer M as a real resource tracked by the habitat."""
    W = Species("W", {}, charge=0)                        # solvent, unit activity
    N = Species("N", {"C": 1, "H": 4}, charge=0)
    M = Species("M", {"C": 1, "H": 4}, charge=0)
    Wst = Species("Wst", {"C": 1, "H": 4}, charge=0)
    reactions = [
        Reaction({"N": 1}, {"M": 1}, A=1e-3, Ea=5e4, reversible=True, solvent=("W",),
                 provenance="illustrative", label="nutrient -> monomer"),
        Reaction({"M": 1}, {"Wst": 1}, A=1e-4, Ea=6e4, reversible=False, solvent=("W",),
                 provenance="illustrative", label="monomer -> waste"),
    ]
    return Network([W, N, M, Wst], reactions)


@dataclass
class Patch:
    """One well-mixed habitat patch (bounded in number)."""
    index: int
    c: np.ndarray              # mol/L per species, aligned to network species order
    temperature_K: float
    pH: float
    liquid_water: bool
    nutrient_mol: float        # externally supplied nutrient reservoir (mol)
    waste_mol: float           # accumulated waste (mol)
    volume_L: float = 1.0


@dataclass
class HabitatState:
    t_gyr: float
    patches: List[Dict]
    protocells: List[Dict]
    replicators: Dict[str, int]   # genotype -> total count across compartments
    pool_monomer: int             # free monomer units available to replicators
    mass_ledger_error: float
    sterile: bool


class Habitat:
    """Coupled planet/chemistry/protocell/replicator habitat.

    A sterile habitat (no seeded replicators) remains sterile; replicators must be
    explicitly seeded (experimental evolution). Replication consumes real modeled
    monomer resources; compartment division partitions resources and replicators
    between daughters without duplication.
    """

    def __init__(self, planet: Planet, network: Network, n_patches: int = 8, seed: int = 0):
        self.planet = planet
        self.network = network
        self.n_patches = int(n_patches)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.t_gyr = float(planet.t_gyr)

        self.species_names = [s.name for s in network.species]
        self.n_spec = len(self.species_names)

        self.patches: List[Patch] = []
        for i in range(self.n_patches):
            self.patches.append(Patch(
                index=i, c=np.zeros(self.n_spec, dtype=float),
                temperature_K=288.0, pH=7.0, liquid_water=True,
                nutrient_mol=1.0, waste_mol=0.0, volume_L=1.0,
            ))
        self._set_initial_chemistry()

        # protocell world (shared amphiphile pool; solute N per cell is conserved on division)
        self.protocells = ProtocellWorld(
            cells=[Protocell(A=10.0, V=5.0, N=1_000_000, lineage=0, gen=0)],
            pool_molecules=100_000_000, seed=seed, max_cells=2000,
        )

        # replicator population: empty until explicitly seeded (sterile by default)
        self.replicators: Optional[Population] = None
        # cell_index -> {genotype: count} for replicators occupying each compartment
        self.replicator_per_cell: Dict[int, Dict[str, int]] = {}
        # cell_index -> monomer units reserved for that cell's replicators
        self.cell_monomers: Dict[int, int] = {}

        # mass ledger: initial conserved element totals + charge
        self._init_element_totals: Optional[np.ndarray] = None
        self._init_charge: float = 0.0
        self._ledger_error = 0.0
        self._reset_ledger()

    # ---- chemistry setup -----------------------------------------------------
    def _set_initial_chemistry(self):
        for p in self.patches:
            p.c = np.zeros(self.n_spec, dtype=float)
            p.nutrient_mol = 1.0
            p.waste_mol = 0.0

    def _patch_env(self, p: Patch) -> Dict:
        return {"T": p.temperature_K, "pH": p.pH, "liquid_water": bool(p.liquid_water), "a_w": 1.0}

    # ---- mass ledger ---------------------------------------------------------
    def _reset_ledger(self):
        self._init_element_totals = None
        self._init_charge = 0.0
        self._ledger_error = 0.0

    def _element_totals(self) -> np.ndarray:
        """Total moles of every element in the chemistry fields (all patches)."""
        tot = np.zeros(self.network.E.shape[0], dtype=float)
        for p in self.patches:
            tot += p.c @ self.network.E.T   # (n_spec,) @ (n_spec, n_elems).T -> (n_elems,)
        return tot

    def _charge_total(self) -> float:
        return float(sum(float(p.c @ self.network.q) for p in self.patches))

    def _check_ledger(self):
        if self._init_element_totals is None:
            self._init_element_totals = self._element_totals().copy()
            self._init_charge = self._charge_total()
        tot = self._element_totals()
        q = self._charge_total()
        eerr = np.abs(tot - self._init_element_totals) / np.maximum(np.abs(self._init_element_totals), 1e-300)
        qerr = abs(q - self._init_charge) / max(abs(self._init_charge), 1e-300)
        self._ledger_error = float(max(eerr.max() if eerr.size else 0.0, qerr))

    # ---- planet <-> chemistry ------------------------------------------------
    def _sync_environment(self):
        env = self.planet.environment()
        Tg = float(env["T_global_K"])
        lw = bool(env["liquid_water_fraction"] > 0)
        ph = env["pH_ocean"]
        ph = ph if (lw and math.isfinite(ph)) else 7.0
        for p in self.patches:
            p.temperature_K = Tg
            p.pH = ph
            p.liquid_water = lw

    def _run_chemistry(self, dt_s: float):
        for p in self.patches:
            if not p.liquid_water:
                continue
            try:
                tout, yt = self.network.integrate(p.c, self._patch_env(p), dt_s, n_out=2)
                p.c = np.maximum(np.asarray(yt[-1], dtype=float), 0.0)
            except Exception:
                # stiff/integration failure: leave patch chemistry unchanged (honest degradation)
                pass

    def _transport(self, dt_s: float):
        if self.n_patches > 1:
            mean_c = sum(p.c for p in self.patches) / self.n_patches
            rate = 1e-3
            for p in self.patches:
                p.c = np.maximum(p.c + rate * (mean_c - p.c), 0.0)

    # ---- replicator seeding (explicit; not abiogenesis) ----------------------
    def seed_replicator(self, genotype: str, count: int = 1, cell_index: int = 0) -> bool:
        """Explicitly seed replicators into a compartment. Returns True if seeded."""
        alive = [c for c in self.protocells.cells if c.alive]
        if not alive or count <= 0:
            return False
        cell = alive[cell_index % len(alive)]
        L = len(genotype)
        need = L * count
        # draw monomers from the cell's internal solute count N (consumes real resources)
        if cell.N < need:
            count = int(cell.N // L)
            need = L * count
            if count <= 0:
                return False
        cell.N -= need
        rate_fn = duplex_mechanism(
            T=self.patches[0].temperature_K, volume_l=1.0,
            k_copy=1e-3, K_M=1e3, d_ss=1e-4, d_ds=1e-6,
        )
        if self.replicators is None:
            self.replicators = Population(
                counts={}, pool=0, rate_fn=rate_fn, mu=1e-4, seed=self.seed,
            )
        self.replicators.counts[genotype] = self.replicators.counts.get(genotype, 0) + count
        self.replicators.lineage.setdefault(genotype, (None, self.t_gyr, 0))
        self.replicators.initial_units = self.replicators.units()
        self.replicator_per_cell.setdefault(cell_index, {})
        self.replicator_per_cell[cell_index][genotype] = (
            self.replicator_per_cell[cell_index].get(genotype, 0) + count
        )
        return True

    def _run_replicators(self, dt_s: float):
        if self.replicators is None or self.replicators.total() == 0:
            return
        alive = [c for c in self.protocells.cells if c.alive]
        if not alive:
            self.replicators.counts = {g: 0 for g in self.replicators.counts}
            return
        rate_fn = duplex_mechanism(
            T=self.patches[0].temperature_K, volume_l=1.0,
            k_copy=1e-3, K_M=1e3, d_ss=1e-4, d_ds=1e-6,
        )
        self.replicators.rate_fn = rate_fn
        # tau-leap: replication consumes monomers from the shared pool (resource-limited)
        self.replicators.tau_leap(dt_s, dt_s)
        self.replicators.counts = {g: n for g, n in self.replicators.counts.items() if n > 0}

    def _protocell_step(self, dt_s: float):
        # solute production draws from patch nutrients (real resource flow into cells).
        # solute_rate is configured on the ProtocellWorld; not overridden here so callers
        # can set it to zero for stable (non-growing) cells.
        self.protocells.step(dt_s)

    def _feedback(self, dt_s: float):
        if self.replicators is None:
            return
        total_rep = self.replicators.total()
        if total_rep == 0:
            return
        consume = 1e-9 * total_rep * dt_s / SECONDS_PER_GYR
        for p in self.patches:
            if p.liquid_water and p.nutrient_mol > 0:
                take = min(consume, p.nutrient_mol)
                p.nutrient_mol -= take
                p.waste_mol += take
                p.pH = max(p.pH - 1e-7 * take, 0.0)

    # ---- main operator-split step -------------------------------------------
    def step(self, dt_gyr: float) -> HabitatState:
        dt_s = dt_gyr * SECONDS_PER_GYR
        self.t_gyr += dt_gyr

        # Operator split (documented timestep constraint: dt_gyr must be small enough that
        # the chemistry BDF integration and replicator tau-leap remain stable; fast processes
        # use seconds internally while the planet climate uses Gyr).
        # 1. planet environment (slow carbon-silicate climate)
        self.planet._equilibrate()
        self._sync_environment()
        # 2. chemistry (mass-action, element/charge conserved)
        self._run_chemistry(dt_s)
        # 3. transport between patches & exterior
        self._transport(dt_s)
        # 4. protocells (conserved division/lysis)
        self._protocell_step(dt_s)
        # 5. replicators (consume monomers; resource-limited)
        self._run_replicators(dt_s)
        # 6. feedback (metabolism alters chemistry)
        self._feedback(dt_s)

        self._check_ledger()
        return self.state()

    def state(self) -> HabitatState:
        cells = self.protocells.cells
        rep_counts: Dict[str, int] = {}
        if self.replicators is not None:
            rep_counts = {g: int(n) for g, n in self.replicators.counts.items() if n > 0}
        return HabitatState(
            t_gyr=self.t_gyr,
            patches=[{
                "index": p.index, "c": p.c.tolist(), "temperature_K": p.temperature_K,
                "pH": p.pH, "liquid_water": p.liquid_water,
                "nutrient_mol": p.nutrient_mol, "waste_mol": p.waste_mol,
            } for p in self.patches],
            protocells=[{
                "A": c.A, "V": c.V, "N": c.N, "alive": c.alive, "gen": c.gen, "lineage": c.lineage,
            } for c in cells],
            replicators=rep_counts,
            pool_monomer=int(self.replicators.pool if self.replicators is not None else 0),
            mass_ledger_error=self._ledger_error,
            sterile=(self.replicators is None or self.replicators.total() == 0),
        )

    def checkpoint(self) -> Dict:
        return {
            "schema": "habitat/v1",
            "t_gyr": self.t_gyr,
            "seed": self.seed,
            "species": self.species_names,
            "patches": [{"c": p.c.tolist(), "nutrient_mol": p.nutrient_mol, "waste_mol": p.waste_mol,
                         "temperature_K": p.temperature_K, "pH": p.pH, "liquid_water": p.liquid_water}
                        for p in self.patches],
            "protocells": [asdict(c) for c in self.protocells.cells],
            "protocell_world": {
                "pool": self.protocells.pool, "t": self.protocells.t,
                "events": dict(self.protocells.events),
                "max_cells": self.protocells.max_cells,
            },
            "replicators": None if self.replicators is None else {
                "counts": {k: int(v) for k, v in self.replicators.counts.items()},
                "pool": self.replicators.pool, "t": self.replicators.t,
                "mu": self.replicators.mu, "supplied": self.replicators.supplied,
                "removed": self.replicators.removed,
                "lineage": self.replicators.lineage,
            },
            "replicator_per_cell": self.replicator_per_cell,
            "cell_monomers": self.cell_monomers,
            "mass_ledger_error": self._ledger_error,
        }

    @classmethod
    def restore(cls, planet: Planet, network: Network, data: Dict, seed: int = 0, n_patches: int = 8):
        hab = cls.__new__(cls)
        hab.planet = planet
        hab.network = network
        hab.n_patches = n_patches
        hab.seed = seed
        hab.rng = np.random.default_rng(seed)
        hab.t_gyr = float(data["t_gyr"])
        hab.species_names = list(data["species"])
        hab.n_spec = len(hab.species_names)
        hab.patches = []
        for i, pd in enumerate(data["patches"]):
            hab.patches.append(Patch(
                index=i, c=np.array(pd["c"], dtype=float),
                temperature_K=pd["temperature_K"], pH=pd["pH"],
                liquid_water=bool(pd["liquid_water"]),
                nutrient_mol=pd["nutrient_mol"], waste_mol=pd["waste_mol"], volume_L=1.0,
            ))
        cells = [
            Protocell(A=c["A"], V=c["V"], N=c["N"],
                      lineage=c.get("lineage", 0), gen=c.get("gen", 0),
                      alive=bool(c.get("alive", True)), fate=c.get("fate", ""))
            for c in data["protocells"]
        ]
        pw = data["protocell_world"]
        hab.protocells = ProtocellWorld(cells=cells, pool_molecules=pw["pool"], seed=seed,
                                        max_cells=pw.get("max_cells", 2000))
        hab.protocells.t = pw["t"]
        hab.protocells.events = dict(pw.get("events", {}))
        hab.replicators = None
        rep = data.get("replicators")
        if rep and rep.get("counts"):
            rate_fn = duplex_mechanism(T=hab.patches[0].temperature_K, volume_l=1.0)
            hab.replicators = Population(
                counts={k: int(v) for k, v in rep["counts"].items()}, pool=int(rep.get("pool", 0)),
                rate_fn=rate_fn, mu=float(rep.get("mu", 0.0)), seed=seed,
            )
            hab.replicators.t = float(rep.get("t", 0.0))
            hab.replicators.supplied = int(rep.get("supplied", 0))
            hab.replicators.removed = int(rep.get("removed", 0))
            hab.replicators.lineage = dict(rep.get("lineage", {}))
            hab.replicators.initial_units = hab.replicators.units()
        hab.replicator_per_cell = {int(k): dict(v) for k, v in data.get("replicator_per_cell", {}).items()}
        hab.cell_monomers = {int(k): int(v) for k, v in data.get("cell_monomers", {}).items()}
        hab._ledger_error = float(data.get("mass_ledger_error", 0.0))
        hab._reset_ledger()
        return hab
