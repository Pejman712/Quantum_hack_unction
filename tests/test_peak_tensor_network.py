import pytest
from qiskit import QuantumCircuit

from quantum_hack.peak import tensor_network as tn
from quantum_hack.peak.marginals import exact_z_marginals
from quantum_hack.simulation import statevector_probability

requires_quimb = pytest.mark.skipif(
    not tn.quimb_available(), reason="quimb (the [tn] extra) is not installed"
)


def test_quimb_available_returns_bool():
    assert isinstance(tn.quimb_available(), bool)


@requires_quimb
def test_tn_probability_matches_statevector():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    for bitstring in ("100", "111", "000"):
        assert tn.probability(qc, bitstring) == pytest.approx(
            statevector_probability(qc, bitstring), abs=1e-9
        )


@requires_quimb
def test_tn_marginals_match_exact():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.ry(0.5, 1)
    assert tn.single_qubit_marginals_z(qc) == pytest.approx(exact_z_marginals(qc), abs=1e-9)
