"""Peak-bitstring estimators for quantum circuits.

Two strategies are provided:

* :func:`statevector_simulation` — exact, dense statevector simulation. Accurate
  but memory scales as ``2**num_qubits``, so it only works for small circuits.
* :func:`matrix_product_operators` — approximate, shot-based sampling via the matrix-product-state
  (MPS) simulator. Scales to many more qubits at the cost of being approximate.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def statevector_simulation(qc: QuantumCircuit, verbose: bool = False) -> tuple[str, float]:
    """Return the most likely measurement bitstring via exact statevector simulation.

    Measurements are removed first because :class:`~qiskit.quantum_info.Statevector`
    operates on a unitary circuit acting on the all-zero state. The input circuit is
    left untouched.

    Args:
        qc (QuantumCircuit): Circuit to simulate.
        verbose (bool): If ``True``, print the peak bitstring and probability.

    Returns:
        tuple[str, float]: The peak bitstring and its exact probability.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    assert qc_no_meas is not None  # inplace=False always returns a new circuit

    sv = Statevector.from_instruction(qc_no_meas)

    probs = sv.probabilities_dict()  # bitstring -> probability
    peak_bitstring = max(probs, key=lambda b: probs[b])
    peak_prob = probs[peak_bitstring]

    if verbose:
        print("Most likely bitstring:", peak_bitstring)
        print("Peak probability:", peak_prob)
    return peak_bitstring, peak_prob


def matrix_product_operators(
    qc: QuantumCircuit,
    shots: int = 4096,
    bond_dim: int = 64,
    verbose: bool = False,
) -> tuple[str, float]:
    """Estimate the most likely bitstring via shot-based MPS sampling.

    The input circuit is left untouched: any final measurements are removed and a
    full ``measure_all`` is added on a copy before sampling.

    Args:
        qc (QuantumCircuit): Circuit to sample.
        shots (int): Number of measurement shots to draw.
        bond_dim (int): Maximum MPS bond dimension.
        verbose (bool): If ``True``, print the estimated peak bitstring and probability.

    Returns:
        tuple[str, float]: The estimated peak bitstring and its estimated probability.
    """
    qc_copy = qc.remove_final_measurements(inplace=False)
    assert qc_copy is not None  # inplace=False always returns a new circuit
    qc_copy.measure_all()

    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
    )

    qc_t = transpile(qc_copy, sim)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    peak_bitstring = max(counts, key=counts.get)
    peak_count = counts[peak_bitstring]
    peak_prob_est = peak_count / shots

    if verbose:
        print("Estimated peak bitstring:", peak_bitstring)
        print("Estimated peak probability:", peak_prob_est)
    return peak_bitstring, peak_prob_est
