from qiskit import QuantumCircuit

from quantum_hack.transform import transpile_to_basis
from quantum_hack.verification import circuits_equivalent, verify_equivalence


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def test_transpiled_circuit_is_equivalent():
    qc = _bell()
    optimized = transpile_to_basis(qc, optimization_level=2)
    assert circuits_equivalent(qc, optimized)


def test_identical_circuits_are_equivalent():
    assert circuits_equivalent(_bell(), _bell())


def test_different_circuits_are_not_equivalent():
    other = QuantumCircuit(2)
    other.x(0)
    assert not circuits_equivalent(_bell(), other)


def test_verify_equivalence_returns_criterion():
    result = verify_equivalence(_bell(), _bell())
    # The enum member exposes a .name; identical circuits are exactly equivalent.
    assert result.name == "equivalent"


def test_equivalent_up_to_global_phase_counts_as_equivalent():
    qc = _bell()
    optimized = transpile_to_basis(qc, optimization_level=2)
    # Transpilation typically introduces a global phase; bool helper must accept it.
    assert verify_equivalence(qc, optimized).name in {
        "equivalent",
        "equivalent_up_to_global_phase",
    }
    assert circuits_equivalent(qc, optimized)
