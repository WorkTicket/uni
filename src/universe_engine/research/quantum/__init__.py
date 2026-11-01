"""Quantum reality subsystem (Research Program B).

Small, verified quantum mechanics for research: states, density matrices,
operators, measurements, entangled systems, and open-system dynamics.

This is NOT a global wavefunction for every particle in the universe.
It supports Bell/CHSH experiments, dephasing, amplitude damping, and
Lindblad evolution for small isolated systems.

Scientific status: implemented, numerically verified (T1). Not observationally
validated (T3) against quantum hardware.
"""

from .states import (
    QuantumState, DensityMatrix, QuantumOperator, Hamiltonian,
    QuantumMeasurement, EntangledSystem, QuantumEnvironment,
    pauli_x, pauli_y, pauli_z,
    TOL, _is_hermitian, _is_positive, _is_trace_one,
)
from .chsh import (
    CHSHExperiment, CHSHResult, verify_chsh_limits,
    chsh_s, chsh_correlation, classical_chsh_bound, quantum_chsh_max,
    optimal_chsh_settings, measurement_in_basis, observable,
)

__all__ = [
    "QuantumState", "DensityMatrix", "QuantumOperator", "Hamiltonian",
    "QuantumMeasurement", "EntangledSystem", "QuantumEnvironment",
    "pauli_x", "pauli_y", "pauli_z",
    "CHSHExperiment", "CHSHResult", "verify_chsh_limits",
    "chsh_s", "chsh_correlation", "classical_chsh_bound", "quantum_chsh_max",
    "optimal_chsh_settings", "measurement_in_basis", "observable",
    "TOL", "_is_hermitian", "_is_positive", "_is_trace_one",
]
