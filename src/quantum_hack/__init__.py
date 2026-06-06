"""Quantum hackathon toolkit: load QASM circuits, find peak bitstrings, strip gates."""

from quantum_hack.challenges import iter_challenges, iter_challenges_gpu, load_circuit
from quantum_hack.gpu import GpuClusterConfig, cluster_info, count_gpus, gpu_available
from quantum_hack.metrics import circuit_stats, compare_circuits
from quantum_hack.simulation import (
    auto_simulation,
    gpu_mps_simulation,
    gpu_statevector_simulation,
    matrix_product_operators,
    statevector_simulation,
)
from quantum_hack.strip import strip_rz_and_cx_from_start, strip_rz_from_end
from quantum_hack.transform import snap_rotations_to_pi_over_4, transpile_to_basis
from quantum_hack.verification import circuits_equivalent, verify_equivalence
from quantum_hack.viz import (
    cx_only_circuit,
    save_circuit_drawing,
    save_counts_histogram,
    save_cx_drawing,
)

__all__ = [
    # CPU simulation
    "statevector_simulation",
    "matrix_product_operators",
    # GPU simulation
    "gpu_statevector_simulation",
    "gpu_mps_simulation",
    "auto_simulation",
    # GPU cluster management
    "GpuClusterConfig",
    "gpu_available",
    "count_gpus",
    "cluster_info",
    # Circuit transforms
    "strip_rz_and_cx_from_start",
    "strip_rz_from_end",
    "snap_rotations_to_pi_over_4",
    "transpile_to_basis",
    # Visualization
    "save_circuit_drawing",
    "save_cx_drawing",
    "cx_only_circuit",
    "save_counts_histogram",
    # Metrics
    "circuit_stats",
    "compare_circuits",
    # Challenges
    "load_circuit",
    "iter_challenges",
    "iter_challenges_gpu",
    # Verification
    "verify_equivalence",
    "circuits_equivalent",
]
