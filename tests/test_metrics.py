from qiskit import QuantumCircuit

from quantum_hack.metrics import circuit_stats, compare_circuits


def test_circuit_stats_reports_basic_metrics():
    # circuit_stats normalizes to the rz/rx/cx basis before counting, so the h is
    # decomposed into basis gates; assert on the normalized result.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    stats = circuit_stats(qc)

    assert stats["num_qubits"] == 2
    assert set(stats["ops"]) <= {"rz", "rx", "cx"}
    assert stats["ops"]["cx"] == 1
    assert stats["size"] > 2  # h expands into multiple basis gates
    assert stats["depth"] >= 2
    assert stats["cx_depth"] == 1  # exactly one CX layer on the critical path


def test_compare_circuits_reports_deltas():
    before = QuantumCircuit(2)
    before.cx(0, 1)
    before.cx(0, 1)
    before.cx(0, 1)  # 3 cx

    after = QuantumCircuit(2)
    after.cx(0, 1)  # 1 cx
    after.rx(0.5, 0)  # a basis gate only present after

    result = compare_circuits(before, after)

    assert result["ops"]["cx"] == {"before": 3, "after": 1, "delta": -2}
    assert result["ops"]["rx"] == {"before": 0, "after": 1, "delta": 1}
    assert result["size"] == {"before": 3, "after": 2, "delta": -1}
    assert result["cx_depth"] == {"before": 3, "after": 1, "delta": -2}


def test_compare_circuits_includes_gates_from_either_side():
    before = QuantumCircuit(1)
    before.rz(0.1, 0)
    after = QuantumCircuit(1)
    after.rx(0.1, 0)

    ops = compare_circuits(before, after)["ops"]

    assert ops["rz"]["delta"] == -1
    assert ops["rx"]["delta"] == 1
