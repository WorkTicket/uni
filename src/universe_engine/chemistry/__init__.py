"""Milestone B: mass-action chemistry with element/charge conservation, thermodynamically consistent reversibility,
temperature- and pH-dependent rates, deterministic (stiff ODE) and stochastic (Gillespie) solvers, driven by a planet environment."""
from .network import Species, Reaction, Network, env_from_planet
__all__ = ["Species", "Reaction", "Network", "env_from_planet"]
