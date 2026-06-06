import math
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from quantum_hack.simulation import (
    matrix_product_operators,
    mps_sample_counts,
    statevector_probability,
    statevector_simulation,
)

SAMPLE_QASM = Path("qasm_data/very_easy/challenge-8_1.qasm")


def _deterministic_circuit() -> QuantumCircuit:
    """3-qubit circuit whose only outcome is |001> (Qiskit's q2 q1 q0 ordering).

    Returns:
        QuantumCircuit: A 3-qubit circuit producing the deterministic ``|001>`` outcome.
    """
    qc = QuantumCircuit(3)
    qc.x(0)
    return qc


def _skewed_circuit() -> QuantumCircuit:
    """1-qubit circuit with distinct outcome probabilities P("0")=0.75, P("1")=0.25.

    Returns:
        QuantumCircuit: A 1-qubit RY(pi/3) circuit with unequal, tie-free probabilities.
    """
    qc = QuantumCircuit(1)
    qc.ry(math.pi / 3, 0)  # P(0)=cos^2(pi/6)=0.75, P(1)=sin^2(pi/6)=0.25
    return qc


def test_statevector_returns_definite_bitstring():
    bitstring, prob = statevector_simulation(_deterministic_circuit())
    assert bitstring == "001"
    assert prob == pytest.approx(1.0)


def test_statevector_does_not_mutate_input():
    qc = _deterministic_circuit()
    qc.measure_all()
    before = len(qc.data)
    statevector_simulation(qc)
    assert len(qc.data) == before  # remove_final_measurements(inplace=False)


def test_matrix_product_operators_returns_definite_bitstring():
    # measure_all prefixes the register name, e.g. "001" stays "001" here.
    bitstring, prob = matrix_product_operators(_deterministic_circuit(), shots=256)
    assert bitstring == "001"
    assert prob == pytest.approx(1.0)


def test_matrix_product_operators_does_not_mutate_input():
    qc = _deterministic_circuit()
    before = len(qc.data)
    matrix_product_operators(qc, shots=256)
    assert len(qc.data) == before  # measurements added on a copy, not the original


def test_statevector_and_matrix_product_operators_agree_on_deterministic_circuit():
    sv_peak = statevector_simulation(_deterministic_circuit())
    mpo_peak = matrix_product_operators(_deterministic_circuit(), shots=512)
    assert sv_peak == mpo_peak


# --- top_n: return a ranked list of (bitstring, probability) pairs ---


def test_statevector_top_n_returns_ranking_sorted_descending():
    ranking = statevector_simulation(_skewed_circuit(), top_n=2)
    assert [b for b, _ in ranking] == ["0", "1"]
    assert ranking[0][1] == pytest.approx(0.75)
    assert ranking[1][1] == pytest.approx(0.25)


def test_statevector_top_n_one_returns_single_element_list_not_tuple():
    ranking = statevector_simulation(_skewed_circuit(), top_n=1)
    assert ranking == [("0", pytest.approx(0.75))]


def test_statevector_top_n_larger_than_outcomes_returns_all_available():
    # The deterministic circuit has a single nonzero outcome; asking for 5 yields 1.
    ranking = statevector_simulation(_deterministic_circuit(), top_n=5)
    assert ranking == [("001", pytest.approx(1.0))]


@pytest.mark.parametrize("bad", [0, -1])
def test_statevector_top_n_must_be_positive(bad):
    with pytest.raises(ValueError, match="positive integer"):
        statevector_simulation(_skewed_circuit(), top_n=bad)


def test_matrix_product_operators_top_n_returns_sorted_ranking():
    ranking = matrix_product_operators(_skewed_circuit(), shots=4096, top_n=2)
    assert len(ranking) == 2
    assert ranking[0][0] == "0"  # the more likely outcome ranks first
    assert ranking[0][1] >= ranking[1][1]
    assert sum(prob for _, prob in ranking) == pytest.approx(1.0)


def test_matrix_product_operators_top_n_one_returns_single_element_list():
    ranking = matrix_product_operators(_deterministic_circuit(), shots=256, top_n=1)
    assert ranking == [("001", pytest.approx(1.0))]


def test_matrix_product_operators_top_n_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        matrix_product_operators(_deterministic_circuit(), shots=64, top_n=0)


# --- verbose: print the result(s) to stdout ---


def test_statevector_verbose_prints_peak(capsys):
    statevector_simulation(_deterministic_circuit(), verbose=True)
    out = capsys.readouterr().out
    assert "Most likely bitstring:" in out
    assert "001" in out


def test_statevector_verbose_prints_ranking(capsys):
    statevector_simulation(_skewed_circuit(), verbose=True, top_n=2)
    out = capsys.readouterr().out
    assert "1. 0" in out  # rank 1 is bitstring "0"
    assert "2. 1" in out  # rank 2 is bitstring "1"


def test_matrix_product_operators_verbose_prints_peak(capsys):
    matrix_product_operators(_deterministic_circuit(), shots=256, verbose=True)
    out = capsys.readouterr().out
    assert "Estimated peak bitstring:" in out
    assert "001" in out


def test_matrix_product_operators_verbose_prints_ranking(capsys):
    matrix_product_operators(_deterministic_circuit(), shots=256, verbose=True, top_n=1)
    out = capsys.readouterr().out
    assert "1. 001" in out


def test_matrix_product_operators_handles_circuit_wider_than_default_backend():
    # 64 qubits exceeds the AerSimulator's memory-based default ceiling (63).
    # Transpiling against the backend target used to raise CircuitTooWideForTarget;
    # the MPS method itself handles widths well beyond that.
    qc = QuantumCircuit(64)
    qc.x(0)
    qc.x(63)
    bitstring, prob = matrix_product_operators(qc, shots=256)
    assert len(bitstring) == 64
    assert prob == pytest.approx(1.0)
    assert bitstring.count("1") == 2  # deterministic: exactly the two flipped qubits


def test_statevector_probability_matches_definite_outcome():
    assert statevector_probability(_deterministic_circuit(), "001") == pytest.approx(1.0)
    assert statevector_probability(_deterministic_circuit(), "000") == pytest.approx(0.0)


def test_statevector_probability_splits_skewed_circuit():
    assert statevector_probability(_skewed_circuit(), "0") == pytest.approx(0.75)
    assert statevector_probability(_skewed_circuit(), "1") == pytest.approx(0.25)


@pytest.mark.parametrize("bad", ["01", "012", "0z1"])
def test_statevector_probability_rejects_bad_bitstring(bad):
    with pytest.raises(ValueError, match="binary chars"):
        statevector_probability(_deterministic_circuit(), bad)


def test_mps_sample_counts_sum_to_shots_and_match_outcome():
    counts = mps_sample_counts(_deterministic_circuit(), shots=256)
    assert sum(counts.values()) == 256
    assert max(counts, key=lambda b: counts[b]) == "001"


@pytest.mark.skipif(not SAMPLE_QASM.exists(), reason="sample QASM not present")
def test_runs_on_real_qasm_circuit():
    qc = QuantumCircuit.from_qasm_file(str(SAMPLE_QASM))
    peak, prob = statevector_simulation(qc)
    assert len(peak) == qc.num_qubits
    assert set(peak) <= {"0", "1"}
    assert 0.0 <= prob <= 1.0
