"""Genetic operators on Genomes.

All operators return a *new* Genome; the original is never mutated in place.
Every operator carries an explicit per-event cost model (metabolic/developmental
cost of the altered architecture) and returns the offspring genome together with
a description of the change for lineage tracking.

Provenance: operator rates and spectra are inputs with declared assumptions.
No operator is "beneficial by default" -- fitness effects emerge from the
phenotype-environment relationship.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .genome import Genome, ALPHABET, ALPHABET_SET


@dataclass
class MutationEvent:
    """A single heritable change applied to a genome."""
    operator: str              # substitution|insertion|deletion|duplication|recombination|regulation|loss|transfer
    position: int              # position in parent genome where the change occurred
    detail: str                # human/machine-readable description
    cost_delta: float = 0.0    # change in developmental/metabolic cost (dimensionless, relative)


@dataclass
class Genetics:
    """Configurable genetic operator rates and spectra.

    All rates are per-genome per-generation probabilities unless noted.
    """
    substitution: float = 1e-4       # per-site substitution probability
    insertion: float = 1e-5          # per-genome insertion probability
    deletion: float = 1e-5           # per-genome deletion probability
    duplication: float = 1e-5        # per-genome segment duplication probability
    recombination: float = 1e-6      # per-genome recombination probability (needs partner)
    regulation: float = 1e-4         # per-site regulatory-weight perturbation probability
    gene_loss: float = 1e-6          # per-genome functional-module loss probability
    horizontal_transfer: float = 1e-7  # per-genome probability (needs donor)

    # Regulatory weight perturbation magnitude.
    regulation_sigma: float = 0.2

    # Cost of maintaining each extra module / regulatory mark (relative).
    cost_per_module: float = 0.001
    cost_per_reg_mark: float = 0.0005


class GeneticOperators:
    """Applies mutation and structural operators to Genomes."""

    def __init__(self, rng: np.random.Generator, genetics: Genetics):
        self.rng = rng
        self.g = genetics

    # ---- point substitution ------------------------------------------------
    def substitute(self, genome: Genome) -> Tuple[Genome, List[MutationEvent]]:
        """Per-site substitution: each site mutates to a different module with prob `substitution`."""
        if genome.length == 0:
            return genome.copy(), []
        modules = list(genome.modules)
        events: List[MutationEvent] = []
        for i in range(len(modules)):
            if self.rng.random() < self.g.substitution:
                old = modules[i]
                choices = [a for a in ALPHABET if a != old]
                modules[i] = self.rng.choice(choices)
                events.append(MutationEvent("substitution", i,
                                            f"{old}->{modules[i]}",
                                            cost_delta=0.0))
        child = Genome(modules=tuple(modules), regulatory=dict(genome.regulatory),
                       metadata=dict(genome.metadata))
        return child, events

    # ---- insertion ---------------------------------------------------------
    def insert(self, genome: Genome) -> Tuple[Genome, List[MutationEvent]]:
        """Insert one random module at a random position."""
        if genome.length == 0:
            child = Genome(modules=(self.rng.choice(ALPHABET),), regulatory={},
                           metadata=dict(genome.metadata))
            return child, [MutationEvent("insertion", 0, f"insert {child.modules[0]}",
                                         cost_delta=self.g.cost_per_module)]
        pos = int(self.rng.integers(0, genome.length + 1))
        new_mod = self.rng.choice(ALPHABET)
        modules = list(genome.modules)
        modules.insert(pos, new_mod)
        # shift regulatory marks after insertion point
        reg = {}
        for p, w in genome.regulatory.items():
            q = p + 1 if p >= pos else p
            reg[q] = w
        child = Genome(modules=tuple(modules), regulatory=reg, metadata=dict(genome.metadata))
        return child, [MutationEvent("insertion", pos, f"insert {new_mod}",
                                     cost_delta=self.g.cost_per_module)]

    # ---- deletion ----------------------------------------------------------
    def delete(self, genome: Genome) -> Tuple[Genome, List[MutationEvent]]:
        """Delete one module at a random position."""
        if genome.length <= 1:
            return genome.copy(), []
        pos = int(self.rng.integers(0, genome.length))
        removed = genome.modules[pos]
        modules = list(genome.modules)
        modules.pop(pos)
        reg = {}
        for p, w in genome.regulatory.items():
            if p == pos:
                continue  # removed mark
            q = p - 1 if p > pos else p
            reg[q] = w
        child = Genome(modules=tuple(modules), regulatory=reg, metadata=dict(genome.metadata))
        return child, [MutationEvent("deletion", pos, f"delete {removed}",
                                     cost_delta=-self.g.cost_per_module)]

    # ---- duplication -------------------------------------------------------
    def duplicate(self, genome: Genome) -> Tuple[Genome, List[MutationEvent]]:
        """Duplicate a contiguous segment (length 1..min(4, genome length))."""
        if genome.length == 0:
            return genome.copy(), []
        seg_len = int(self.rng.integers(1, min(4, genome.length) + 1))
        start = int(self.rng.integers(0, genome.length - seg_len + 1))
        segment = genome.modules[start:start + seg_len]
        modules = list(genome.modules)
        modules[start:start] = list(segment)  # insert copy right after original
        # shift regulatory marks
        reg = {}
        for p, w in genome.regulatory.items():
            if p >= start:
                reg[p + seg_len] = w
            else:
                reg[p] = w
        # copy regulatory marks for duplicated segment positions
        for p, w in genome.regulatory.items():
            if start <= p < start + seg_len:
                reg[p + seg_len] = w
        child = Genome(modules=tuple(modules), regulatory=reg, metadata=dict(genome.metadata))
        cost = seg_len * self.g.cost_per_module
        return child, [MutationEvent("duplication", start,
                                     f"dup segment len={seg_len}", cost_delta=cost)]

    # ---- regulatory variation ---------------------------------------------
    def regulate(self, genome: Genome) -> Tuple[Genome, List[MutationEvent]]:
        """Perturb regulatory weights at each site with prob `regulation`."""
        if genome.length == 0:
            return genome.copy(), []
        reg = dict(genome.regulatory)
        events: List[MutationEvent] = []
        for i in range(genome.length):
            if self.rng.random() < self.g.regulation:
                old = reg.get(i, 1.0)
                new_w = float(np.clip(old + self.rng.normal(0.0, self.g.regulation_sigma), 0.05, 1.0))
                reg[i] = new_w
                events.append(MutationEvent("regulation", i,
                                            f"weight {old:.3f}->{new_w:.3f}",
                                            cost_delta=0.0))
        child = Genome(modules=tuple(genome.modules), regulatory=reg, metadata=dict(genome.metadata))
        return child, events

    # ---- gene/functional-module loss --------------------------------------
    def lose_gene(self, genome: Genome) -> Tuple[Genome, List[MutationEvent]]:
        """Remove one functional module (non-neutral, non-structural) if any."""
        functional = {m for m in ALPHABET if m != "Z"}
        if genome.length == 0:
            return genome.copy(), []
        candidates = [i for i, m in enumerate(genome.modules) if m in functional]
        if not candidates:
            return genome.copy(), []
        pos = int(self.rng.choice(candidates))
        removed = genome.modules[pos]
        modules = list(genome.modules)
        modules.pop(pos)
        reg = {}
        for p, w in genome.regulatory.items():
            if p == pos:
                continue
            q = p - 1 if p > pos else p
            reg[q] = w
        child = Genome(modules=tuple(modules), regulatory=reg, metadata=dict(genome.metadata))
        return child, [MutationEvent("gene_loss", pos, f"lose {removed}",
                                     cost_delta=-self.g.cost_per_module)]

    # ---- recombination -----------------------------------------------------
    def recombine(self, a: Genome, b: Genome) -> Tuple[Genome, List[MutationEvent]]:
        """Single-point crossover between two genomes. Returns child from a+b."""
        if a.length == 0 and b.length == 0:
            return a.copy(), []
        if a.length == 0:
            return b.copy(), [MutationEvent("recombination", 0, "adopt b", cost_delta=0.0)]
        if b.length == 0:
            return a.copy(), [MutationEvent("recombination", 0, "adopt a", cost_delta=0.0)]
        pa = int(self.rng.integers(0, a.length + 1))
        pb = int(self.rng.integers(0, b.length + 1))
        modules = tuple(list(a.modules[:pa]) + list(b.modules[pb:]))
        # combine regulatory marks
        reg: Dict[int, float] = {}
        for p, w in a.regulatory.items():
            if p < pa:
                reg[p] = w
        for p, w in b.regulatory.items():
            q = pa + p
            if q < pa + (b.length - pb):
                reg[q] = w
        child = Genome(modules=modules, regulatory=reg, metadata=dict(a.metadata))
        return child, [MutationEvent("recombination", pa, f"crossover a[{pa}]+b[{pb}]",
                                     cost_delta=0.0)]

    # ---- horizontal transfer -----------------------------------------------
    def transfer(self, recipient: Genome, donor: Genome) -> Tuple[Genome, List[MutationEvent]]:
        """Copy a random segment from donor into recipient at a random position."""
        if donor.length == 0:
            return recipient.copy(), []
        if recipient.length == 0:
            seg = donor.modules
            child = Genome(modules=tuple(seg), regulatory={},
                           metadata=dict(recipient.metadata))
            return child, [MutationEvent("horizontal_transfer", 0,
                                         f"acquire {len(seg)} modules from donor",
                                         cost_delta=len(seg) * self.g.cost_per_module)]
        seg_len = int(self.rng.integers(1, min(4, donor.length) + 1))
        start = int(self.rng.integers(0, donor.length - seg_len + 1))
        segment = donor.modules[start:start + seg_len]
        pos = int(self.rng.integers(0, recipient.length + 1))
        modules = list(recipient.modules)
        modules[pos:pos] = list(segment)
        reg = dict(recipient.regulatory)
        for p in list(reg.keys()):
            if p >= pos:
                reg[p + seg_len] = reg.pop(p)
        child = Genome(modules=tuple(modules), regulatory=reg, metadata=dict(recipient.metadata))
        return child, [MutationEvent("horizontal_transfer", pos,
                                     f"acquire {seg_len} modules from donor",
                                     cost_delta=seg_len * self.g.cost_per_module)]
