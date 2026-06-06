"""Peak-bitstring solving: method selection, marginal attack, confidence-gated solve.

The heavy tensor-network backend (:mod:`quantum_hack.peak.tensor_network`) lazily
imports ``quimb``/``cotengra`` only when called, so importing this subpackage works
without them. The exact and MPS paths rely only on Qiskit and Qiskit Aer.
"""

from quantum_hack.peak.marginals import (
    exact_z_marginals,
    greedy_refine,
    least_confident_qubits,
    marginal_bitstring,
    z_marginals_from_counts,
)
from quantum_hack.peak.result import PeakResult, score_confidence
from quantum_hack.peak.selector import (
    CircuitProfile,
    Method,
    profile_circuit,
    select_method,
)
from quantum_hack.peak.solve import solve_peak

__all__ = [
    "PeakResult",
    "score_confidence",
    "Method",
    "CircuitProfile",
    "profile_circuit",
    "select_method",
    "exact_z_marginals",
    "z_marginals_from_counts",
    "marginal_bitstring",
    "least_confident_qubits",
    "greedy_refine",
    "solve_peak",
]
