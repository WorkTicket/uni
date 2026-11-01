"""Planetary ecology integration (Research Program C8/D8).

Connects the ECS-IW agent controller to habitats, chemistry, and evolutionary
populations.
"""

from .experiment import (
    PlanetaryAgentConfig, PlanetaryAgent, make_planet_habitat,
    run_planetary_agent_experiment,
)

__all__ = [
    "PlanetaryAgentConfig", "PlanetaryAgent",
    "make_planet_habitat", "run_planetary_agent_experiment",
]
