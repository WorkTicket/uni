"""Milestone D: heritable replication. Template-directed copying of sequences with an explicit per-site error rate, resource (monomer)
consumption with exact accounting, decay, serial transfer, genotype lineage tracking. The copying chemistry itself is a SUPPLIED mechanism
(not discovered): see provenance in population.py. No claim is made that such replicators arise spontaneously."""
from .population import Population, duplex_mechanism, mutate, ALPHABET
__all__ = ["Population", "duplex_mechanism", "mutate", "ALPHABET"]
