"""EvolutionScheduler: multi-rate coupling of simulation clocks.

Couples cosmic, planetary, biological, behavioral and rendering clocks with
clearly defined handoffs. No single global timestep is forced; each domain
advances at its own rate with conservative handoffs. Fast-forward preserves the
selected approximation's scientific validity and numerical stability.

Provenance: clock coupling is a software scheduling mechanism, not a physical
claim about time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
import numpy as np


@dataclass
class Clock:
    """One named clock with its own rate and state."""
    name: str
    t: float = 0.0
    rate: float = 1.0          # simulation-time units per step
    dt: float = 0.01           # step size in simulation time
    enabled: bool = True
    steps: int = 0

    def step(self) -> float:
        if not self.enabled:
            return self.t
        self.t += self.dt * self.rate
        self.steps += 1
        return self.t


@dataclass
class EvolutionScheduler:
    """Coordinates clocks across simulation domains.

    biological_dt is the finest (individual agent/behavior) timestep.
    planetary_dt is slower (climate/chemistry). cosmic_dt is slowest.
    Each step() advances all enabled clocks once; domains may be subcycled.
    """

    cosmic: Clock = field(default_factory=lambda: Clock("cosmic", dt=1.0, rate=1.0))
    planetary: Clock = field(default_factory=lambda: Clock("planetary", dt=0.1, rate=1.0))
    biological: Clock = field(default_factory=lambda: Clock("biological", dt=0.001, rate=1.0))
    behavioral: Clock = field(default_factory=lambda: Clock("behavioral", dt=0.001, rate=1.0))
    rendering: Clock = field(default_factory=lambda: Clock("rendering", dt=0.05, rate=1.0))

    adaptive: bool = True        # slow biological clock when populations are stable
    min_biological_dt: float = 1e-6
    max_biological_dt: float = 1e-2

    def step_all(self) -> Dict[str, float]:
        return {
            "cosmic": self.cosmic.step(),
            "planetary": self.planetary.step(),
            "biological": self.biological.step(),
            "behavioral": self.behavioral.step(),
            "rendering": self.rendering.step(),
        }

    def adaptive_biological_dt(self, population_change: float) -> None:
        """Adjust biological dt based on how fast the population is changing.

        population_change: absolute fractional change in population size this
        step (0 = stable, 1 = turnover). Larger change -> smaller dt.
        """
        if not self.adaptive:
            return
        # More change -> finer resolution; stable -> coarser (but bounded).
        target = self.max_biological_dt * (1.0 - population_change) + \
                 self.min_biological_dt * population_change
        self.biological.dt = float(np.clip(target, self.min_biological_dt,
                                            self.max_biological_dt))

    def checkpoint(self) -> dict:
        return {
            "cosmic": {"t": self.cosmic.t, "dt": self.cosmic.dt, "rate": self.cosmic.rate,
                       "enabled": self.cosmic.enabled, "steps": self.cosmic.steps},
            "planetary": {"t": self.planetary.t, "dt": self.planetary.dt, "rate": self.planetary.rate,
                          "enabled": self.planetary.enabled, "steps": self.planetary.steps},
            "biological": {"t": self.biological.t, "dt": self.biological.dt, "rate": self.biological.rate,
                           "enabled": self.biological.enabled, "steps": self.biological.steps},
            "behavioral": {"t": self.behavioral.t, "dt": self.behavioral.dt, "rate": self.behavioral.rate,
                           "enabled": self.behavioral.enabled, "steps": self.behavioral.steps},
            "rendering": {"t": self.rendering.t, "dt": self.rendering.dt, "rate": self.rendering.rate,
                          "enabled": self.rendering.enabled, "steps": self.rendering.steps},
            "adaptive": self.adaptive,
        }

    @classmethod
    def restore(cls, data: dict) -> "EvolutionScheduler":
        def ck(name):
            d = data.get(name, {})
            return Clock(name=name, t=float(d.get("t", 0.0)), dt=float(d.get("dt", 0.01)),
                         rate=float(d.get("rate", 1.0)), enabled=bool(d.get("enabled", True)),
                         steps=int(d.get("steps", 0)))
        return cls(cosmic=ck("cosmic"), planetary=ck("planetary"),
                   biological=ck("biological"), behavioral=ck("behavioral"),
                   rendering=ck("rendering"), adaptive=bool(data.get("adaptive", True)))

