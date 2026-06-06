"""Quantum hackathon toolkit: load QASM circuits, find peak bitstrings, strip gates."""

from quantum_hack.batch import (
    CacheComparison,
    CircuitJob,
    compare_cache_to_results,
    failed_challenges,
    list_jobs,
    run_batch,
    solve_job,
    write_results_csvs,
)
from quantum_hack.challenges import find_challenge, iter_challenges, load_circuit
from quantum_hack.metrics import circuit_stats, compare_circuits
from quantum_hack.optimize import OptimizeResult, cost, optimize_circuit
from quantum_hack.peak import (
    CircuitProfile,
    Method,
    PeakResult,
    exact_z_marginals,
    greedy_refine,
    marginal_bitstring,
    profile_circuit,
    score_confidence,
    select_method,
    solve_peak,
    z_marginals_from_counts,
)
from quantum_hack.results import (
    ChallengeStatus,
    SubmissionRecord,
    challenge_status,
    is_known_failure,
    load_results,
)
from quantum_hack.simulation import (
    matrix_product_operators,
    mps_sample_counts,
    statevector_probability,
    statevector_simulation,
)
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
    # simulation
    "statevector_simulation",
    "matrix_product_operators",
    "mps_sample_counts",
    "statevector_probability",
    # peak solving
    "solve_peak",
    "PeakResult",
    "Method",
    "CircuitProfile",
    "profile_circuit",
    "select_method",
    "score_confidence",
    "exact_z_marginals",
    "z_marginals_from_counts",
    "marginal_bitstring",
    "greedy_refine",
    # batch
    "run_batch",
    "solve_job",
    "list_jobs",
    "write_results_csvs",
    "compare_cache_to_results",
    "failed_challenges",
    "CacheComparison",
    "CircuitJob",
    # optimization
    "optimize_circuit",
    "OptimizeResult",
    "cost",
    # strip / transform
    "strip_rz_and_cx_from_start",
    "strip_rz_from_start",
    "strip_rz_from_end",
    "strip",
    "snap_rotations_to_pi_over_4",
    "transpile_to_basis",
    # metrics / verification
    "circuit_stats",
    "compare_circuits",
    "verify_equivalence",
    "circuits_equivalent",
    # challenges / results
    "load_circuit",
    "find_challenge",
    "iter_challenges",
    "load_results",
    "challenge_status",
    "is_known_failure",
    "ChallengeStatus",
    "SubmissionRecord",
    # viz
    "save_circuit_drawing",
    "save_cx_drawing",
    "cx_only_circuit",
    "save_counts_histogram",
]
