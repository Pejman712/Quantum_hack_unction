import re

from qiskit import QuantumCircuit

from quantum_hack.viz import (
    cx_only_circuit,
    save_circuit_drawing,
    save_counts_histogram,
    save_cx_drawing,
)


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def test_saves_stable_filename_without_timestamp(tmp_path):
    path = save_circuit_drawing(
        _bell(), "bell", out_dir=tmp_path, timestamp=False
    )

    assert path == tmp_path / "bell.png"
    assert path.exists() and path.stat().st_size > 0


def test_timestamped_filename_matches_sortable_pattern(tmp_path):
    path = save_circuit_drawing(_bell(), "bell", out_dir=tmp_path)

    assert path.exists()
    assert re.fullmatch(r"bell_\d{8}_\d{6}\.png", path.name)


def test_creates_missing_output_directory(tmp_path):
    nested = tmp_path / "a" / "b"
    path = save_circuit_drawing(_bell(), out_dir=nested, timestamp=False)

    assert path == nested / "circuit.png"
    assert nested.is_dir()


def test_respects_image_format(tmp_path):
    path = save_circuit_drawing(
        _bell(), "bell", out_dir=tmp_path, timestamp=False, image_format="svg"
    )

    assert path.suffix == ".svg"
    assert path.exists()


def test_cx_only_circuit_keeps_only_cx_gates():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rz(0.5, 1)
    qc.cx(0, 1)
    qc.x(0)
    qc.cx(1, 0)
    qc.measure_all()

    cx_only = cx_only_circuit(qc)

    assert cx_only.num_qubits == 2
    names = [instr.operation.name for instr in cx_only.data]
    assert names == ["cx", "cx"]
    assert [cx_only.find_bit(q).index for q in cx_only.data[0].qubits] == [0, 1]
    assert [cx_only.find_bit(q).index for q in cx_only.data[1].qubits] == [1, 0]


def test_save_cx_drawing_writes_file(tmp_path):
    path = save_cx_drawing(_bell(), "bell_cx", out_dir=tmp_path, timestamp=False)

    assert path == tmp_path / "bell_cx.png"
    assert path.exists() and path.stat().st_size > 0


def test_save_counts_histogram_writes_file(tmp_path):
    counts = {"00": 512, "11": 488}
    path = save_counts_histogram(counts, "hist", out_dir=tmp_path, timestamp=False)

    assert path == tmp_path / "hist.png"
    assert path.exists() and path.stat().st_size > 0
