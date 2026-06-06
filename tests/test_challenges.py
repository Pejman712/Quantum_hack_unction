import pytest
from qiskit import QuantumCircuit

from quantum_hack.challenges import find_challenge, iter_challenges, load_circuit

QASM_X = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
x q[0];
"""

QASM_ID = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
id q[0];
"""


def _write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def test_load_circuit_reads_qasm(tmp_path):
    f = tmp_path / "c.qasm"
    f.write_text(QASM_X)
    qc = load_circuit(f)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 1


def test_load_circuit_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_circuit(tmp_path / "does_not_exist.qasm")


def _make_data_dir(tmp_path):
    _write(tmp_path / "very_easy" / "a.qasm", QASM_X)
    _write(tmp_path / "easy" / "b.qasm", QASM_ID)
    return tmp_path


def test_find_challenge_locates_file_across_difficulties(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    assert find_challenge("b", data_dir=data_dir) == data_dir / "easy" / "b.qasm"


def test_find_challenge_missing_raises(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        find_challenge("nope", data_dir=data_dir)


def test_find_challenge_ambiguous_raises(tmp_path):
    _write(tmp_path / "very_easy" / "dup.qasm", QASM_X)
    _write(tmp_path / "easy" / "dup.qasm", QASM_X)
    with pytest.raises(FileNotFoundError, match="Multiple"):
        find_challenge("dup", data_dir=tmp_path)


def test_find_challenge_exact_stem_not_prefix(tmp_path):
    _write(tmp_path / "very_easy" / "c-1.qasm", QASM_X)
    _write(tmp_path / "very_easy" / "c-11.qasm", QASM_ID)
    assert find_challenge("c-1", data_dir=tmp_path) == tmp_path / "very_easy" / "c-1.qasm"


def test_iter_challenges_filters_by_difficulty(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    results = dict(iter_challenges("very_easy", data_dir=data_dir))
    assert list(results) == ["very_easy/a"]


def test_iter_challenges_walks_all_difficulties_sorted(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    names = [name for name, _ in iter_challenges(data_dir=data_dir, func=None)]
    assert names == ["easy/b", "very_easy/a"]


def test_iter_challenges_func_none_yields_circuits(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    _, value = next(iter(iter_challenges("very_easy", func=None, data_dir=data_dir)))
    assert isinstance(value, QuantumCircuit)


def test_iter_challenges_default_func_runs_matrix_product_operators(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    # Default func is matrix_product_operators, which returns (bitstring, probability);
    # the |1> circuit must yield bitstring "1".
    results = dict(iter_challenges("very_easy", data_dir=data_dir))
    bitstring, _ = results["very_easy/a"]
    assert bitstring == "1"


def test_iter_challenges_accepts_custom_func(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    results = dict(iter_challenges(data_dir=data_dir, func=lambda qc: qc.num_qubits))
    assert results == {"very_easy/a": 1, "easy/b": 1}
