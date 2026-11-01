"""Open-ended genotype representation.

A Genome is a sequence of *functional modules* over an extensible alphabet.
Modules encode traits (metabolism, sensing, motility, structure, regulation,
reproduction, defence, coordination). The genome is the unit of heredity:
mutation, insertion, deletion, duplication, recombination and gene loss all
operate on it explicitly. Fitness is never assigned arbitrarily -- it emerges
from the phenotype-environment relationship.

Provenance: this is a synthetic genotype-to-phenotype substrate, not a model of
any specific terrestrial nucleic acid. Alphabet size, mutation rates and module
semantics are inputs with declared assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

# Extensible functional alphabet. Each symbol is a functional module code.
# Modules are interpreted by the DevelopmentalProgram into phenotypic traits.
ALPHABET: Tuple[str, ...] = (
    "A",  # metabolic core (baseline energy harvesting)
    "C",  # photosynthetic / light harvesting
    "G",  # chemotrophic / chemical energy harvesting
    "T",  # structural support / cell wall
    "M",  # membrane / transport
    "S",  # sensing module
    "L",  # motility / locomotion
    "D",  # defence / toxin resistance
    "R",  # reproduction / cell division
    "N",  # neural / signalling (nervous-system-like)
    "X",  # regulatory (modulates expression of neighbours)
    "Y",  # horizontal-transfer receptor
    "Z",  # neutral / structural spacer
)
ALPHABET_SET = frozenset(ALPHABET)
_ALPHABET_IDX = {a: i for i, a in enumerate(ALPHABET)}


@dataclass
class Genome:
    """A linear genotype: a tuple of module symbols plus regulatory marks.

    Parameters
    ----------
    modules:
        Ordered tuple of module symbols (each in ALPHABET).
    regulatory:
        Mapping module-position -> expression weight in (0, 1]. Position is the
        index in `modules`. Marks are inherited and subject to mutation.
    metadata:
        Provenance and annotation (biochemistry label, source, assumed parameters).
    """

    modules: Tuple[str, ...]
    regulatory: Dict[int, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for m in self.modules:
            if m not in ALPHABET_SET:
                raise ValueError(f"unknown module symbol {m!r}; allowed {sorted(ALPHABET_SET)}")
        for pos, w in self.regulatory.items():
            if not (0.0 < w <= 1.0):
                raise ValueError(f"regulatory weight at {pos} must be in (0, 1], got {w}")
        # Normalise regulatory keys to valid positions.
        self.regulatory = {p: w for p, w in self.regulatory.items() if 0 <= p < len(self.modules)}

    @property
    def length(self) -> int:
        return len(self.modules)

    @classmethod
    def random(cls, rng: np.random.Generator, min_len: int = 4, max_len: int = 24,
               metadata: Optional[Dict[str, object]] = None) -> "Genome":
        """Draw a random genome of length in [min_len, max_len]."""
        n = int(rng.integers(min_len, max_len + 1))
        modules = tuple(rng.choice(ALPHABET, size=n).tolist())
        reg = {i: float(rng.uniform(0.2, 1.0)) for i in range(n) if rng.random() < 0.3}
        return cls(modules=modules, regulatory=reg, metadata=dict(metadata or {}))

    @property
    def sequence(self) -> str:
        return "".join(self.modules)

    @sequence.setter
    def sequence(self, value: str) -> None:
        for ch in value:
            if ch not in ALPHABET_SET:
                raise ValueError(f"unknown module symbol {ch!r}")
        self.modules = tuple(value)
        self.regulatory = {p: w for p, w in self.regulatory.items() if p < len(self.modules)}

    def module_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {a: 0 for a in ALPHABET}
        for m in self.modules:
            counts[m] += 1
        return counts

    def copy(self) -> "Genome":
        return Genome(modules=tuple(self.modules),
                      regulatory=dict(self.regulatory),
                      metadata=dict(self.metadata))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Genome):
            return NotImplemented
        return (self.modules == other.modules and self.regulatory == other.regulatory)

    def __hash__(self) -> int:
        return hash((self.modules, tuple(sorted(self.regulatory.items()))))
