"""Peak-bitstring estimators for quantum circuits.

Three strategies are provided:

* :func:`statevector_simulation` — exact, dense statevector simulation on CPU.
  Accurate but memory scales as ``2**num_qubits``.
* :func:`matrix_product_operators` — approximate, shot-based sampling via the
  matrix-product-state (MPS) simulator on CPU.  Scales to many more qubits.
* :func:`gpu_statevector_simulation` — exact statevector on a GPU cluster via
  NVIDIA cuStateVec (single or multi-GPU).  Falls back to CPU statevector when
  no CUDA GPU is present.
* :func:`gpu_mps_simulation` — shot-based MPS on GPU.  Falls back to CPU MPS.
* :func:`auto_simulation` — pick the fastest available backend automatically
  based on qubit count and detected GPU resources.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# Qubit threshold: circuits at or below this size use exact statevector.
_SV_QUBIT_LIMIT = 20


# ---------------------------------------------------------------------------
# CPU backends (unchanged from original implementation)
# ---------------------------------------------------------------------------

def statevector_simulation(qc: QuantumCircuit, verbose: bool = False) -> tuple[str, float]:
    """Return the most likely measurement bitstring via exact CPU statevector simulation.

    Measurements are removed first because :class:`~qiskit.quantum_info.Statevector`
    operates on a unitary circuit acting on the all-zero state.

    Args:
        qc: Circuit to simulate.
        verbose: If ``True``, print the peak bitstring and probability.

    Returns:
        tuple[str, float]: The peak bitstring and its exact probability.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    assert qc_no_meas is not None

    sv = Statevector.from_instruction(qc_no_meas)
    probs = sv.probabilities_dict()
    peak_bitstring = max(probs, key=lambda b: probs[b])
    peak_prob = probs[peak_bitstring]

    if verbose:
        print("Most likely bitstring:", peak_bitstring)
        print("Peak probability:", peak_prob)
    return peak_bitstring, peak_prob


def matrix_product_operators(
    qc: QuantumCircuit,
    shots: int = 4096,
    bond_dim: int = 128,
    verbose: bool = False,
) -> tuple[str, float]:
    """Estimate the most likely bitstring via shot-based CPU MPS sampling.

    Args:
        qc: Circuit to sample.
        shots: Number of measurement shots to draw.
        bond_dim: Maximum MPS bond dimension.
        verbose: If ``True``, print the estimated peak bitstring and probability.

    Returns:
        tuple[str, float]: The estimated peak bitstring and its estimated probability.
    """
    qc_copy = qc.remove_final_measurements(inplace=False)
    assert qc_copy is not None
    qc_copy.measure_all()

    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
    )
    qc_t = transpile(qc_copy, basis_gates=["cx", "u", "measure", "reset"], optimization_level=0)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    peak_bitstring = max(counts, key=counts.get)
    peak_prob_est = counts[peak_bitstring] / shots

    if verbose:
        print("Estimated peak bitstring:", peak_bitstring)
        print("Estimated peak probability:", peak_prob_est)
    return peak_bitstring, peak_prob_est


# ---------------------------------------------------------------------------
# GPU backends
# ---------------------------------------------------------------------------

def gpu_statevector_simulation(
    qc: QuantumCircuit,
    config: "GpuClusterConfig | None" = None,
    shots: int = 8192,
    verbose: bool = False,
) -> tuple[str, float]:
    """Return the most likely bitstring via GPU-accelerated statevector simulation.

    Uses NVIDIA cuStateVec through Qiskit Aer's ``statevector`` method with
    ``device='GPU'``.  Multi-GPU nodes are handled automatically via Aer's
    blocking mode when ``config.num_gpus > 1``.  Falls back to CPU statevector
    transparently when no CUDA GPU is detected.

    The statevector is obtained by running the circuit with ``save_statevector``
    (zero-shot execution) so that exact probabilities are computed, not
    shot-sampled estimates.

    Args:
        qc: Circuit to simulate.
        config: GPU cluster config.  ``None`` triggers auto-detection.
        shots: Used only for the CPU-fallback path (ignored on GPU).
        verbose: If ``True``, print the peak bitstring and probability.

    Returns:
        tuple[str, float]: The peak bitstring and its exact (GPU) or estimated
        (CPU fallback) probability.
    """
    from quantum_hack.gpu import GpuClusterConfig, make_statevector_simulator

    if config is None:
        config = GpuClusterConfig.auto_detect()

    if not config.has_gpu:
        if verbose:
            print("[gpu_statevector_simulation] No GPU detected — falling back to CPU statevector.")
        return statevector_simulation(qc, verbose=verbose)

    qc_sv = qc.remove_final_measurements(inplace=False)
    assert qc_sv is not None
    qc_sv.save_statevector()

    sim = make_statevector_simulator(config)
    qc_t = transpile(qc_sv, basis_gates=["cx", "u", "save_statevector"], optimization_level=0)
    result = sim.run(qc_t, shots=0).result()

    import numpy as np
    sv_data = result.data(qc_t)["statevector"]
    amps = np.asarray(sv_data)
    probs = (amps.conj() * amps).real
    peak_idx = int(np.argmax(probs))
    peak_bitstring = format(peak_idx, f"0{qc.num_qubits}b")
    peak_prob = float(probs[peak_idx])

    if verbose:
        print(f"[GPU/{config}] Peak bitstring: {peak_bitstring}  ({peak_prob:.4%})")
    return peak_bitstring, peak_prob


def gpu_mps_simulation(
    qc: QuantumCircuit,
    config: "GpuClusterConfig | None" = None,
    shots: int = 4096,
    bond_dim: int = 128,
    verbose: bool = False,
) -> tuple[str, float]:
    """Estimate the most likely bitstring via GPU-accelerated MPS sampling.

    Uses Qiskit Aer's ``matrix_product_state`` method with ``device='GPU'``
    when a CUDA GPU is present; falls back to CPU MPS otherwise.

    Args:
        qc: Circuit to sample.
        config: GPU cluster config.  ``None`` triggers auto-detection.
        shots: Number of measurement shots to draw.
        bond_dim: Maximum MPS bond dimension.
        verbose: If ``True``, print the estimated peak bitstring and probability.

    Returns:
        tuple[str, float]: The estimated peak bitstring and its estimated probability.
    """
    from quantum_hack.gpu import GpuClusterConfig, make_mps_simulator

    if config is None:
        config = GpuClusterConfig.auto_detect()

    if not config.has_gpu and verbose:
        print("[gpu_mps_simulation] No GPU detected — falling back to CPU MPS.")

    qc_copy = qc.remove_final_measurements(inplace=False)
    assert qc_copy is not None
    qc_copy.measure_all()

    sim = make_mps_simulator(config, bond_dim=bond_dim)
    qc_t = transpile(qc_copy, basis_gates=["cx", "u", "measure", "reset"], optimization_level=0)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    peak_bitstring = max(counts, key=counts.get)
    peak_prob_est = counts[peak_bitstring] / shots

    if verbose:
        print(f"[GPU-MPS/{config}] Estimated peak bitstring: {peak_bitstring}  ({peak_prob_est:.4%})")
    return peak_bitstring, peak_prob_est


def auto_simulation(
    qc: QuantumCircuit,
    config: "GpuClusterConfig | None" = None,
    shots: int = 4096,
    bond_dim: int = 128,
    sv_qubit_limit: int = _SV_QUBIT_LIMIT,
    verbose: bool = False,
) -> tuple[str, float]:
    """Choose and run the best available simulation backend automatically.

    Selection logic (in priority order):

    1. **GPU statevector** — if a CUDA GPU is present *and* the circuit fits
       within ``sv_qubit_limit`` qubits.
    2. **GPU MPS** — if a CUDA GPU is present *and* the circuit is larger than
       ``sv_qubit_limit`` qubits.
    3. **CPU statevector** — if no GPU is available and the circuit is small.
    4. **CPU MPS** — if no GPU is available and the circuit is large.

    Args:
        qc: Circuit to simulate.
        config: GPU cluster config.  ``None`` triggers auto-detection.
        shots: Shot count for MPS paths.
        bond_dim: Bond dimension for MPS paths.
        sv_qubit_limit: Qubit threshold below (or equal to) which statevector
            is preferred over MPS.
        verbose: If ``True``, print the selected backend and result.

    Returns:
        tuple[str, float]: The peak bitstring and its probability.
    """
    from quantum_hack.gpu import GpuClusterConfig

    if config is None:
        config = GpuClusterConfig.auto_detect()

    small = qc.num_qubits <= sv_qubit_limit

    if config.has_gpu:
        if small:
            backend = "GPU-statevector"
            result = gpu_statevector_simulation(qc, config=config, verbose=verbose)
        else:
            backend = "GPU-MPS"
            result = gpu_mps_simulation(qc, config=config, shots=shots, bond_dim=bond_dim, verbose=verbose)
    else:
        if small:
            backend = "CPU-statevector"
            result = statevector_simulation(qc, verbose=verbose)
        else:
            backend = "CPU-MPS"
            result = matrix_product_operators(qc, shots=shots, bond_dim=bond_dim, verbose=verbose)

    if verbose:
        print(f"[auto_simulation] backend={backend}, qubits={qc.num_qubits}")
    return result
