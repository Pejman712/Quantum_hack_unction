"""Quantum hackathon toolkit: load QASM circuits, find peak bitstrings, strip gates."""

from quantum_hack.challenges import find_challenge, iter_challenges, load_circuit
from quantum_hack.metrics import circuit_stats, compare_circuits
from quantum_hack.results import (
    ChallengeStatus,
    SubmissionRecord,
    challenge_status,
    is_known_failure,
    load_results,
)
from quantum_hack.simulation import matrix_product_operators, statevector_simulation
from quantum_hack.strip import (
    strip,
    strip_rz_and_cx_from_start,
    strip_rz_from_end,
    strip_rz_from_start,
)
from quantum_hack.transform import snap_rotations_to_pi_over_4, transpile_to_basis
from quantum_hack.verification import circuits_equivalent, verify_equivalence
from quantum_hack.viz import (
    cx_only_circuit,
    save_circuit_drawing,
    save_counts_histogram,
    save_cx_drawing,
)

__all__ = [
    "statevector_simulation",
    "matrix_product_operators",
    "strip_rz_and_cx_from_start",
    "strip_rz_from_start",
    "strip_rz_from_end",
    "strip",
    "snap_rotations_to_pi_over_4",
    "transpile_to_basis",
    "save_circuit_drawing",
    "save_cx_drawing",
    "cx_only_circuit",
    "save_counts_histogram",
    "circuit_stats",
    "compare_circuits",
    "load_circuit",
    "find_challenge",
    "iter_challenges",
    "load_results",
    "challenge_status",
    "is_known_failure",
    "ChallengeStatus",
    "SubmissionRecord",
    "verify_equivalence",
    "circuits_equivalent",
]
