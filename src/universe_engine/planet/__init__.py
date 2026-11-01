"""Milestone A: coupled planetary environment (energy balance, carbon-silicate cycle, water state, geothermal energy).
Every parameterisation carries a provenance label; see constants.PROVENANCE. No life is created here."""
from .model import PlanetSpec, Planet, StarSpec
__all__ = ["PlanetSpec", "Planet", "StarSpec"]
