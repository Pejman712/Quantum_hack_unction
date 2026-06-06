import pytest
from qiskit import QuantumCircuit

from quantum_hack.peak.solve import solve_peak
from quantum_hack.results import SubmissionRecord


def _failed_record(challenge: str, bitstring: str, qubits: int) -> SubmissionRecord:
    """Build a 'failed' submission record for the known-failure tests.

    Args:
        challenge (str): Challenge name.
        bitstring (str): The rejected bitstring.
        qubits (int): Qubit count.

    Returns:
        SubmissionRecord: A record with status 'failed'.
    """
    return SubmissionRecord(
        challenge=challenge,
        qubits=qubits,
        method="MPS",
        bitstring=bitstring,
        probability=0.9,
        status="failed",
        difficulty="test",
    )


def test_solve_finds_definite_peak_via_mps():
    qc = QuantumCircuit(3)
    qc.x(0)
    result = solve_peak(qc)
    assert result.bitstring == "001"
    assert result.probability == pytest.approx(1.0)
    # MPS sample and marginal vote agree on the only outcome (no exact statevector is used).
    assert {"MPS", "marginal"}.issubset(result.agreeing_methods)
    assert result.confidence > 0.5
    # Without a tensor-network oracle the local-max check is skipped; with quimb it confirms True.
    assert result.is_local_max in (None, True)


def test_solve_reports_consensus_on_small_skewed_circuit():
    qc = QuantumCircuit(4)
    qc.x(0)
    qc.x(2)
    qc.ry(0.3, 1)  # qubit 1 stays most likely |0>
    result = solve_peak(qc)
    assert result.bitstring == "0101"
    assert {"MPS", "marginal"}.issubset(result.agreeing_methods)


def test_solve_avoids_a_known_failed_bitstring_when_alternatives_exist():
    # Clear peak "00" (~0.91) with a sampled runner-up "01" (~0.09).
    qc = QuantumCircuit(2)
    qc.ry(0.6, 0)
    assert solve_peak(qc).bitstring == "00"
    records = [_failed_record("ch", "00", 2)]
    switched = solve_peak(qc, challenge="ch", records=records)
    assert switched.bitstring == "01"
    assert "already failed" in switched.notes


def test_solve_notes_when_only_candidate_is_known_failure():
    qc = QuantumCircuit(3)
    qc.x(0)
    records = [_failed_record("ch", "001", 3)]
    result = solve_peak(qc, challenge="ch", records=records)
    assert "already failed" in result.notes


def test_solve_does_not_mutate_input_circuit():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.measure_all()
    before = len(qc.data)
    solve_peak(qc)
    assert len(qc.data) == before
