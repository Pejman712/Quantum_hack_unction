from qiskit import QuantumCircuit

from quantum_hack.peak.selector import (
    Method,
    profile_circuit,
    select_method,
)


def _circuit(num_qubits: int, two_qubit_gates: int) -> QuantumCircuit:
    """Build a circuit with a known qubit count and number of CX gates.

    Args:
        num_qubits (int): Number of qubits (must be >= 2 when two_qubit_gates > 0).
        two_qubit_gates (int): Number of CX gates to add.

    Returns:
        QuantumCircuit: The constructed circuit.
    """
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(0.3, i)
    for k in range(two_qubit_gates):
        qc.cx(k % num_qubits, (k + 1) % num_qubits)
    return qc


def test_profile_counts_qubits_two_qubit_gates_and_depth():
    profile = profile_circuit(_circuit(5, 4))
    assert profile.num_qubits == 5
    assert profile.num_two_qubit_gates == 4
    assert profile.depth > 0


def test_profile_ignores_final_measurements():
    qc = _circuit(3, 2)
    qc.measure_all()
    assert profile_circuit(qc).num_qubits == 3


def test_small_circuit_uses_mps_then_marginal():
    chain = select_method(profile_circuit(_circuit(28, 10)))
    assert chain[:3] == [Method.MPS, Method.MARGINAL, Method.GREEDY_REFINE]
    assert Method.TENSOR_NETWORK in chain  # 28 <= TN_MAX_QUBITS


def test_solver_never_uses_exact_statevector():
    # The peak is always found with MPS, never an exact statevector.
    assert not hasattr(Method, "EXACT_STATEVECTOR")
    for n in (8, 28, 40, 64):
        assert all(m != "statevector" for m in select_method(profile_circuit(_circuit(n, 20))))


def test_medium_shallow_circuit_uses_mps_first():
    chain = select_method(profile_circuit(_circuit(40, 50)))
    assert chain[0] == Method.MPS
    assert Method.MARGINAL in chain and Method.GREEDY_REFINE in chain


def test_large_deep_circuit_leads_with_marginal():
    chain = select_method(profile_circuit(_circuit(60, 1200)))
    assert chain[0] == Method.MARGINAL
    # Deep + wide + no GPU: no exact tensor-network amplitude in the chain.
    assert Method.TENSOR_NETWORK not in chain


def test_gpu_enables_tensor_network_for_large_circuit():
    chain = select_method(profile_circuit(_circuit(60, 1200)), gpu_available=True)
    assert Method.TENSOR_NETWORK in chain
