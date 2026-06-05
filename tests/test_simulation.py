from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from quantum_hack.simulation import matrix_product_operators, statevector_simulation

SAMPLE_QASM = Path("qasm_data/very_easy/challenge-8_1.qasm")


def _deterministic_circuit() -> QuantumCircuit:
    """3-qubit circuit whose only outcome is |001> (Qiskit's q2 q1 q0 ordering).

    Returns:
        QuantumCircuit: A 3-qubit circuit producing the deterministic ``|001>`` outcome.
    """
    qc = QuantumCircuit(3)
    qc.x(0)
    return qc


def test_statevector_returns_definite_bitstring():
    assert statevector_simulation(_deterministic_circuit()) == "001"


def test_statevector_does_not_mutate_input():
    qc = _deterministic_circuit()
    qc.measure_all()
    before = len(qc.data)
    statevector_simulation(qc)
    assert len(qc.data) == before  # remove_final_measurements(inplace=False)


def test_matrix_product_operators_returns_definite_bitstring():
    # measure_all prefixes the register name, e.g. "001" stays "001" here.
    assert matrix_product_operators(_deterministic_circuit(), shots=256) == "001"


def test_matrix_product_operators_does_not_mutate_input():
    qc = _deterministic_circuit()
    before = len(qc.data)
    matrix_product_operators(qc, shots=256)
    assert len(qc.data) == before  # measurements added on a copy, not the original


def test_statevector_and_matrix_product_operators_agree_on_deterministic_circuit():
    sv_peak = statevector_simulation(_deterministic_circuit())
    mpo_peak = matrix_product_operators(_deterministic_circuit(), shots=512)
    assert sv_peak == mpo_peak


@pytest.mark.skipif(not SAMPLE_QASM.exists(), reason="sample QASM not present")
def test_runs_on_real_qasm_circuit():
    qc = QuantumCircuit.from_qasm_file(str(SAMPLE_QASM))
    peak = statevector_simulation(qc)
    assert len(peak) == qc.num_qubits
    assert set(peak) <= {"0", "1"}
