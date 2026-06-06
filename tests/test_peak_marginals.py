import math

import pytest
from qiskit import QuantumCircuit

from quantum_hack.peak.marginals import (
    exact_z_marginals,
    greedy_refine,
    least_confident_qubits,
    marginal_bitstring,
    z_marginals_from_counts,
)
from quantum_hack.simulation import statevector_probability


def test_exact_marginals_sign_matches_definite_state():
    # |001>: qubit 0 is |1> (<Z> = -1), qubits 1 and 2 are |0> (<Z> = +1).
    qc = QuantumCircuit(3)
    qc.x(0)
    marginals = exact_z_marginals(qc)
    assert marginals[0] == pytest.approx(-1.0)
    assert marginals[1] == pytest.approx(1.0)
    assert marginals[2] == pytest.approx(1.0)


def test_marginal_bitstring_uses_qiskit_ordering():
    # qubit 0 -> '1', others -> '0' gives "001" (rightmost char is qubit 0).
    assert marginal_bitstring([-1.0, 1.0, 1.0]) == "001"


def test_exact_marginal_vote_recovers_definite_state():
    qc = QuantumCircuit(4)
    qc.x(0)
    qc.x(2)
    assert marginal_bitstring(exact_z_marginals(qc)) == "0101"


def test_z_marginals_from_counts_matches_sampling():
    # qubit 0 always 1, qubit 1 always 0 across the samples.
    counts = {"01": 100}  # bitstring "01": qubit0=1, qubit1=0
    marginals = z_marginals_from_counts(counts, 2)
    assert marginals[0] == pytest.approx(-1.0)
    assert marginals[1] == pytest.approx(1.0)


def test_z_marginals_from_empty_counts_is_zero():
    assert z_marginals_from_counts({}, 3) == [0.0, 0.0, 0.0]


def test_least_confident_qubits_sorted_by_abs_expectation():
    # |<Z>|: q0=0.9, q1=0.1, q2=0.5 -> least confident first: q1, q2.
    assert least_confident_qubits([0.9, -0.1, 0.5], 2) == [1, 2]


def test_greedy_refine_climbs_to_peak_from_wrong_seed():
    # Each qubit biased toward |1> (P(1) = sin^2(1) ~ 0.71); the peak is "11".
    qc = QuantumCircuit(2)
    qc.ry(2.0, 0)
    qc.ry(2.0, 1)
    refined, prob = greedy_refine(qc, "00", oracle=statevector_probability)
    assert refined == "11"
    assert prob == pytest.approx(math.sin(1.0) ** 4, rel=1e-6)


def test_greedy_refine_rejects_wrong_length_seed():
    qc = QuantumCircuit(3)
    with pytest.raises(ValueError, match="3 chars"):
        greedy_refine(qc, "00", oracle=statevector_probability)
