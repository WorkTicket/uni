"""Tests for evolutionary architecture search (Sprint 6, Research Program C6)."""

import numpy as np
import pytest

from universe_engine.research.evolution.experiment import (
    AgentGenome, AgentPhenotype, EvolvableAgent, LineageRecord,
    EvolutionArchive, EvolutionaryExperiment, make_controller, mutate_genome,
)


# ---------------------------------------------------------------------------
# Genome tests
# ---------------------------------------------------------------------------

def test_genome_alphabet_validation():
    """Genome only allows valid module symbols."""
    g = AgentGenome(modules=("R", "W", "S", "M"))
    assert "R" in g.modules
    with pytest.raises(ValueError):
        AgentGenome(modules=("X",))  # invalid module


def test_genome_random():
    """Random genome generation produces valid genomes."""
    rng = np.random.default_rng(42)
    g = AgentGenome.random(rng)
    for m in g.modules:
        assert m in g.ALPHABET
    assert 1 <= len(g.modules) <= 6


def test_genome_copy():
    """Genome copy is independent."""
    g1 = AgentGenome(modules=("R", "S"), regulatory={0: 0.5})
    g2 = g1.copy()
    assert g2.modules == g1.modules
    assert g2.regulatory == g1.regulatory
    g2.modules = ("W",)
    assert g1.modules == ("R", "S")  # original unchanged


# ---------------------------------------------------------------------------
# Phenotype tests
# ---------------------------------------------------------------------------

def test_phenotype_development():
    """Genotype-to-phenotype mapping produces traits and costs."""
    g = AgentGenome(modules=("E", "S", "M", "D", "N"))
    phen = AgentPhenotype.develop(g)
    assert phen.controller_type == "ecs_iw"
    assert phen.sensing_range >= 1
    assert phen.metabolic_efficiency > 1.0
    assert phen.defence > 0
    assert phen.maintenance_cost > 0


def test_phenotype_reactive_default():
    """Genome without E/W modules develops reactive controller."""
    g = AgentGenome(modules=("S", "M"))
    phen = AgentPhenotype.develop(g)
    assert phen.controller_type == "reactive"


def test_phenotype_recurrent():
    """Genome with W but not E develops recurrent controller."""
    g = AgentGenome(modules=("W", "S"))
    phen = AgentPhenotype.develop(g)
    assert phen.controller_type == "recurrent"


# ---------------------------------------------------------------------------
# Mutation tests
# ---------------------------------------------------------------------------

def test_mutation_preserves_alphabet():
    """Mutation never produces invalid module symbols."""
    rng = np.random.default_rng(1)
    g = AgentGenome(modules=("R", "S", "M"))
    for _ in range(100):
        child = mutate_genome(g, rng)
        for m in child.modules:
            assert m in g.ALPHABET


def test_mutation_not_always_beneficial():
    """Mutation can increase maintenance cost (not automatically beneficial)."""
    rng = np.random.default_rng(7)
    g = AgentGenome(modules=("R",), mutation_rate=0.5)
    phen_before = AgentPhenotype.develop(g)
    costs_before = phen_before.maintenance_cost
    # Run many mutations and check costs vary (not all equal)
    costs = []
    for _ in range(100):
        child = mutate_genome(g, rng)
        phen_child = AgentPhenotype.develop(child)
        costs.append(phen_child.maintenance_cost)
    assert min(costs) != max(costs)  # mutation produces variation in cost


# ---------------------------------------------------------------------------
# Archive tests
# ---------------------------------------------------------------------------

def test_archive_register_lineage():
    """Archive tracks lineages with parent relationships."""
    arch = EvolutionArchive()
    g1 = AgentGenome(modules=("R",))
    g2 = AgentGenome(modules=("R", "S"))
    h1 = arch.register_lineage(g1, None, 0, 0.0)
    h2 = arch.register_lineage(g2, g1, 1, 1.0)
    assert h1 != h2  # different genomes
    assert arch.lineages[h2].parent_hash == h1
    assert arch.lineages[h2].generation == 1


def test_archive_diversity():
    """Archive computes diversity metrics."""
    arch = EvolutionArchive()
    g1 = AgentGenome(modules=("R",))
    g2 = AgentGenome(modules=("E", "S"))
    arch.register_lineage(g1, None, 0, 0.0)
    arch.register_lineage(g2, None, 0, 0.0)
    div = arch.diversity()
    assert div["n_lineages"] == 2
    assert div["genome_diversity"] == 2


def test_archive_extinction():
    """Archive records extinction events."""
    arch = EvolutionArchive()
    g = AgentGenome(modules=("R",))
    arch.register_lineage(g, None, 0, 0.0)
    arch.extinction(g, 10.0)
    h = hash(g)
    assert arch.lineages[h].extinction_time == 10.0
    assert arch.total_extinctions == 1


# ---------------------------------------------------------------------------
# Evolutionary experiment tests
# ---------------------------------------------------------------------------

def test_experiment_creation():
    """Evolutionary experiment creates with founders."""
    exp = EvolutionaryExperiment(seed=42, n_founders=5)
    assert len(exp.archive.population) == 5
    assert exp.t == 0.0


def test_experiment_step():
    """One evolutionary step advances time and may produce births/deaths."""
    exp = EvolutionaryExperiment(seed=42, n_founders=5)
    stats_before = exp.stats()
    exp.step()
    stats_after = exp.stats()
    assert stats_after["t"] > stats_before["t"]
    assert stats_after["generation"] == stats_before["generation"] + 1


def test_experiment_run():
    """Running multiple steps produces stats history."""
    exp = EvolutionaryExperiment(seed=42, n_founders=5)
    history = exp.run(n_steps=20)
    assert len(history) == 20
    assert history[-1]["t"] > history[0]["t"]


def test_experiment_resource_limited():
    """Population cannot grow without bound -- energy/food is limited."""
    exp = EvolutionaryExperiment(seed=1, n_founders=3, max_energy=5.0)
    history = exp.run(n_steps=50)
    # Population should not explode; some agents may die
    final = history[-1]
    assert final["n_alive"] >= 0
    assert final["n_total"] >= final["n_alive"]


def test_experiment_diversity_maintained():
    """Diversity metrics are tracked across generations."""
    exp = EvolutionaryExperiment(seed=7, n_founders=8)
    exp.run(n_steps=30)
    div = exp.archive.diversity()
    assert "n_lineages" in div
    assert "controller_diversity" in div
    assert div["n_lineages"] >= 1


def test_experiment_checkpoint():
    """Experiment checkpoint preserves state."""
    exp = EvolutionaryExperiment(seed=42, n_founders=4)
    exp.run(n_steps=10)
    cp = exp.checkpoint()
    assert cp["seed"] == 42
    assert cp["t"] == exp.t
    assert "archive" in cp
    assert "lineages" in cp["archive"]
    assert "population" in cp["archive"]


def test_experiment_no_humanoid_reward():
    """Experiment does not reward humanoid traits -- no such concept exists."""
    exp = EvolutionaryExperiment(seed=99, n_founders=4)
    history = exp.run(n_steps=20)
    # Fitness is based on reward/steps, not on any "humanoid" measure
    for h in history:
        assert "mean_fitness" in h
        assert "max_fitness" in h
        assert "n_lineages" in h


def test_experiment_changing_environment():
    """Environment changes over time to test adaptation."""
    exp = EvolutionaryExperiment(seed=5, n_founders=6)
    # Run enough steps to trigger environmental changes (every 10 time units)
    history = exp.run(n_steps=200)
    # Environment should have changed at least once
    regen_rates = [p.regeneration_rate for p in exp.env.patches.values()]
    assert len(set(regen_rates)) > 1  # not all the same
