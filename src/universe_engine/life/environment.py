"""Environment and resource accounting.

PlanetEnvironment: temperature, atmospheric/water conditions, radiation/light,
chemical nutrients, acidity/salinity, spatial location, terrain, seasonal
variation.
HabitatGrid: spatial partitioning of the planet into patches, each holding
ChemicalResources.
ChemicalResources: tracked chemical inventory (mol) per patch with conservation.

Organisms consume actual available resources; growth/reproduction require
material and energy; waste enters resource inventories; death redistributes
material through decomposition. No infinite food, no free reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

SECONDS_PER_GYR = 3.15576e16


@dataclass
class ChemicalResources:
    """Chemical inventory of one location.

    concentrations: mol/L per species, aligned to a fixed species order.
    nutrient_mol / waste_mol: reservoirs (mol) external to the dissolved pool.
    volume_L: patch volume.
    """
    species_names: List[str]
    concentrations: np.ndarray     # mol/L per species
    nutrient_mol: float = 0.0      # externally supplied nutrient reservoir
    waste_mol: float = 0.0         # accumulated waste
    volume_L: float = 1.0

    def __post_init__(self) -> None:
        self.concentrations = np.asarray(self.concentrations, dtype=float)
        if self.concentrations.shape[0] != len(self.species_names):
            raise ValueError("concentrations length must match species_names")

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    def species_mol(self, name: str) -> float:
        i = self.species_names.index(name)
        return float(self.concentrations[i] * self.volume_L)

    def set_species_mol(self, name: str, mol: float) -> None:
        i = self.species_names.index(name)
        self.concentrations[i] = float(mol) / self.volume_L

    def add_mol(self, name: str, mol: float) -> None:
        """Add `mol` of species (conservation: increases inventory)."""
        i = self.species_names.index(name)
        self.concentrations[i] += float(mol) / self.volume_L

    def take_mol(self, name: str, mol: float) -> float:
        """Take up to `mol` of species; returns amount actually taken."""
        i = self.species_names.index(name)
        available = float(self.concentrations[i] * self.volume_L)
        take = min(mol, available)
        if take > 0:
            self.concentrations[i] -= take / self.volume_L
        return take

    def total_mol(self) -> float:
        return float(np.sum(self.concentrations * self.volume_L)) + self.nutrient_mol + self.waste_mol

    def checkpoint(self) -> dict:
        return {
            "species_names": list(self.species_names),
            "concentrations": self.concentrations.tolist(),
            "nutrient_mol": self.nutrient_mol,
            "waste_mol": self.waste_mol,
            "volume_L": self.volume_L,
        }

    @classmethod
    def restore(cls, data: dict) -> "ChemicalResources":
        return cls(species_names=list(data["species_names"]),
                   concentrations=np.array(data["concentrations"], dtype=float),
                   nutrient_mol=float(data.get("nutrient_mol", 0.0)),
                   waste_mol=float(data.get("waste_mol", 0.0)),
                   volume_L=float(data.get("volume_L", 1.0)))


@dataclass
class Patch:
    """One well-mixed habitat patch (bounded in number)."""
    index: int
    location: Tuple[float, ...]
    resources: ChemicalResources
    temperature_K: float = 288.0
    pH: float = 7.0
    liquid_water: bool = True
    radiation: float = 0.0          # relative radiation dose
    light: float = 1.0              # relative light (photosynthesis)
    salinity: float = 35.0          # psu
    acidity: float = 7.0            # pH alias
    terrain: str = "liquid"         # liquid|sediment|rock|ice
    seasonal_phase: float = 0.0     # 0..1 seasonal cycle

    def checkpoint(self) -> dict:
        return {
            "index": self.index, "location": list(self.location),
            "resources": self.resources.checkpoint(),
            "temperature_K": self.temperature_K, "pH": self.pH,
            "liquid_water": self.liquid_water, "radiation": self.radiation,
            "light": self.light, "salinity": self.salinity, "acidity": self.acidity,
            "terrain": self.terrain, "seasonal_phase": self.seasonal_phase,
        }

    @classmethod
    def restore(cls, data: dict) -> "Patch":
        return cls(index=int(data["index"]), location=tuple(data["location"]),
                   resources=ChemicalResources.restore(data["resources"]),
                   temperature_K=float(data.get("temperature_K", 288.0)),
                   pH=float(data.get("pH", 7.0)),
                   liquid_water=bool(data.get("liquid_water", True)),
                   radiation=float(data.get("radiation", 0.0)),
                   light=float(data.get("light", 1.0)),
                   salinity=float(data.get("salinity", 35.0)),
                   acidity=float(data.get("acidity", 7.0)),
                   terrain=str(data.get("terrain", "liquid")),
                   seasonal_phase=float(data.get("seasonal_phase", 0.0)))


@dataclass
class HabitatGrid:
    """Spatial grid of habitat patches partitioning a planet surface."""
    patches: List[Patch]
    dims: Tuple[int, ...] = field(default_factory=lambda: (1,))
    spacing_m: float = 1e6          # patch spacing (m)

    def __post_init__(self) -> None:
        if not self.patches:
            raise ValueError("HabitatGrid requires at least one patch")
        self._by_index = {p.index: p for p in self.patches}

    @property
    def n_patches(self) -> int:
        return len(self.patches)

    def patch(self, index: int) -> Optional[Patch]:
        return self._by_index.get(index)

    def nearest(self, location: Tuple[float, ...]) -> Patch:
        best = None; best_d = float("inf")
        loc = np.asarray(location, dtype=float)
        for p in self.patches:
            d = float(np.sum((np.asarray(p.location, dtype=float) - loc) ** 2))
            if d < best_d:
                best_d = d; best = p
        return best

    def total_resources(self) -> ChemicalResources:
        """Aggregate resource inventory across all patches (conservation check)."""
        if not self.patches:
            raise ValueError("empty grid")
        names = self.patches[0].resources.species_names
        conc = np.zeros(len(names)); nutrient = 0.0; waste = 0.0; vol = 0.0
        for p in self.patches:
            r = p.resources
            conc += r.concentrations * r.volume_L
            nutrient += r.nutrient_mol; waste += r.waste_mol; vol += r.volume_L
        if vol > 0:
            conc = conc / vol
        return ChemicalResources(species_names=names, concentrations=conc,
                                 nutrient_mol=nutrient, waste_mol=waste, volume_L=vol)

    def checkpoint(self) -> dict:
        return {
            "dims": list(self.dims), "spacing_m": self.spacing_m,
            "patches": [p.checkpoint() for p in self.patches],
        }

    @classmethod
    def restore(cls, data: dict) -> "HabitatGrid":
        patches = [Patch.restore(pd) for pd in data.get("patches", [])]
        return cls(patches=patches, dims=tuple(data.get("dims", [len(patches)])),
                   spacing_m=float(data.get("spacing_m", 1e6)))


@dataclass
class PlanetEnvironment:
    """Whole-planet environment state.

    temperature: global mean and per-patch.
    atmospheric/water conditions, radiation, light, chemistry, acidity/salinity,
    terrain, seasonal/longer-term variation.
    """
    t_gyr: float = 0.0
    global_temperature_K: float = 288.0
    liquid_water_fraction: float = 1.0
    pco2_bar: float = 2.8e-4
    surface_pressure_pa: float = 1e5
    radiation: float = 0.0
    light: float = 1.0
    ph_ocean: float = 7.0
    grid: Optional[HabitatGrid] = None
    seasonal_phase: float = 0.0
    flags: List[str] = field(default_factory=list)

    def environment_dict(self) -> dict:
        return {
            "t_gyr": self.t_gyr, "T_global_K": self.global_temperature_K,
            "T_liquid_mean_K": self.global_temperature_K if self.liquid_water_fraction > 0 else float("nan"),
            "liquid_water_fraction": self.liquid_water_fraction,
            "pCO2_bar": self.pco2_bar, "surface_pressure_pa": self.surface_pressure_pa,
            "pH_ocean": self.ph_ocean, "radiation": self.radiation,
            "light": self.light, "seasonal_phase": self.seasonal_phase,
            "flags": list(self.flags),
        }

    def checkpoint(self) -> dict:
        return {
            "t_gyr": self.t_gyr, "global_temperature_K": self.global_temperature_K,
            "liquid_water_fraction": self.liquid_water_fraction, "pco2_bar": self.pco2_bar,
            "surface_pressure_pa": self.surface_pressure_pa, "radiation": self.radiation,
            "light": self.light, "ph_ocean": self.ph_ocean,
            "seasonal_phase": self.seasonal_phase, "flags": list(self.flags),
            "grid": self.grid.checkpoint() if self.grid is not None else None,
        }

    @classmethod
    def restore(cls, data: dict) -> "PlanetEnvironment":
        env = cls(t_gyr=float(data.get("t_gyr", 0.0)),
                  global_temperature_K=float(data.get("global_temperature_K", 288.0)),
                  liquid_water_fraction=float(data.get("liquid_water_fraction", 1.0)),
                  pco2_bar=float(data.get("pco2_bar", 2.8e-4)),
                  surface_pressure_pa=float(data.get("surface_pressure_pa", 1e5)),
                  radiation=float(data.get("radiation", 0.0)),
                  light=float(data.get("light", 1.0)),
                  ph_ocean=float(data.get("ph_ocean", 7.0)),
                  seasonal_phase=float(data.get("seasonal_phase", 0.0)),
                  flags=list(data.get("flags", [])))
        gd = data.get("grid")
        env.grid = HabitatGrid.restore(gd) if gd is not None else None
        return env
