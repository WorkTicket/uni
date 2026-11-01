"""MetabolicNetwork: energy and material conversion for organisms.

Maps an organism's phenotype and the local ChemicalResources into metabolic
fluxes: energy acquisition (photosynthesis, chemotrophy, baseline metabolism),
nutrient uptake, waste production, and maintenance cost. All fluxes are bounded
by available local resources -- no free energy.

All rates are expressed per Gigayear (Gyr) and `dt` is taken in Gyr, consistent
with the world step. Element/charge conservation is enforced by the underlying
chemistry network; this layer tracks energy/nutrient bookkeeping.

Provenance: fluxes are illustrative, declared assumptions, not measured
metabolic rates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .environment import ChemicalResources
from .organism import Organism
from .development import Phenotype


@dataclass
class MetabolicFlux:
    """Result of one metabolic update for an organism over dt (Gyr)."""
    energy_gained: float       # energy added to organism (units)
    nutrients_gained: float    # nutrient material added
    waste_produced: float      # waste returned to local resources
    maintenance_cost: float    # energy spent on upkeep
    net_energy: float          # energy_gained - maintenance_cost
    resource_draw: Dict[str, float]  # species name -> mol drawn from local resources


@dataclass
class MetabolicNetwork:
    """Converts phenotype traits + local resources into metabolic fluxes.

    All yields and rates are per-Gyr (per unit of the relevant trait).
    """

    # Energy yield per-Gyr per unit of relevant trait.
    yield_per_module: Dict[str, float] = field(default_factory=lambda: {
        "photosynthesis": 30.0, "chemotrophy": 18.0, "metabolism": 12.0,
    })
    nutrient_cost_per_energy: float = 1e-3
    # Baseline maintenance per-Gyr per unit size.
    maintenance_rate: float = 0.05

    def update(self, org: Organism, resources: ChemicalResources, dt: float,
               light: float = 1.0, temperature_K: float = 288.0) -> MetabolicFlux:
        """Compute metabolic fluxes over `dt` (Gyr).

        Draws nutrients from `resources` (mutating it). Temperature/light
        modulate yield.
        """
        phen = org.phenotype
        draw: Dict[str, float] = {}
        energy_gained = 0.0
        nutrients_gained = 0.0

        # Temperature modifier (Q10-like around 288 K).
        q10 = 2.0
        temp_mod = q10 ** ((temperature_K - 288.0) / 10.0)
        temp_mod = float(np.clip(temp_mod, 0.1, 5.0))

        # Photosynthesis: needs light + a carbon/nutrient source (species M).
        photo = phen.traits.get("photosynthesis")
        if photo is not None and photo.value > 0 and light > 0:
            carbon = self._draw(resources, "M", photo.value * light * dt * 1e-3, draw)
            gain = photo.value * light * temp_mod * dt * 1.0  # per-Gyr yield
            energy_gained += gain
            nutrients_gained += carbon * 0.5

        # Chemotrophy: chemical energy from nutrients (species N).
        chemo = phen.traits.get("chemotrophy")
        if chemo is not None and chemo.value > 0:
            chem_nut = self._draw(resources, "N", chemo.value * dt * 1e-3, draw)
            gain = chemo.value * temp_mod * dt * 1.0
            energy_gained += gain
            nutrients_gained += chem_nut * 0.5

        # Baseline metabolism (A module): slow harvest from nutrient pool (species N).
        meta = phen.traits.get("metabolism")
        if meta is not None and meta.value > 0:
            meta_nut = self._draw(resources, "N", meta.value * dt * 1e-4, draw)
            gain = meta.value * temp_mod * dt * 0.5
            energy_gained += gain
            nutrients_gained += meta_nut * 0.2

        # Transport (M module) improves uptake efficiency.
        trans = phen.traits.get("transport")
        trans_mod = 1.0 + 0.5 * (trans.value if trans else 0.0)
        energy_gained *= trans_mod
        nutrients_gained *= trans_mod

        # Maintenance cost per-Gyr, scales with size, motility, defence, cognition.
        phen_cost = phen.total_maintenance_cost()
        size_cost = self.maintenance_rate * phen.size
        motility_cost = 0.2 * phen.motility
        defence_cost = 0.1 * phen.defence
        cog_cost = float(org.metadata.get("cognitive_cost", 0.0))
        maintenance_cost = (phen_cost + size_cost + motility_cost + defence_cost + cog_cost) * dt

        net_energy = energy_gained - maintenance_cost

        # Store into organism state (bounded by storage capacity = 10*size).
        org.state.energy = float(np.clip(org.state.energy + net_energy, 0.0, 10.0 * phen.size))
        org.state.nutrients = float(np.clip(org.state.nutrients + nutrients_gained, 0.0,
                                            10.0 * phen.size))

        # Waste: a fraction of processed nutrients returned to local resources.
        waste_produced = 0.1 * nutrients_gained
        if "Wst" in resources.species_names:
            resources.add_mol("Wst", waste_produced)

        return MetabolicFlux(
            energy_gained=energy_gained, nutrients_gained=nutrients_gained,
            waste_produced=waste_produced, maintenance_cost=maintenance_cost,
            net_energy=net_energy, resource_draw=draw,
        )

    @staticmethod
    def _draw(resources: ChemicalResources, name: str, mol: float,
              draw: Dict[str, float]) -> float:
        if name not in resources.species_names:
            return 0.0
        taken = resources.take_mol(name, mol)
        draw[name] = draw.get(name, 0.0) + taken
        return taken
