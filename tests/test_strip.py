from qiskit import QuantumCircuit

from quantum_hack.strip import strip_rz_and_cx_from_start, strip_rz_from_end


def _gate_names(qc: QuantumCircuit) -> list[str]:
    return [instruction.operation.name for instruction in qc.data]


# --- strip_rz_and_cx_from_start: drop leading RZ/CX before each qubit's first RX ---


def test_start_drops_leading_rz_and_cx_keeps_after_first_rx():
    qc = QuantumCircuit(2)
    qc.rz(0.1, 0)  # dropped: before q0's first rx
    qc.cx(0, 1)  # dropped: before first rx on both q0 and q1
    qc.rx(0.5, 0)  # kept: q0 boundary
    qc.rx(0.5, 1)  # kept: q1 boundary
    qc.rz(0.2, 0)  # kept: after q0's rx
    qc.cx(0, 1)  # kept: after rx on both

    stripped = strip_rz_and_cx_from_start(qc)

    assert _gate_names(stripped) == ["rx", "rx", "rz", "cx"]


def test_start_keeps_cx_that_is_leading_on_only_one_qubit():
    # q0 has already seen an rx, so the cx is past the boundary on q0 and must stay,
    # even though q1 never sees an rx.
    qc = QuantumCircuit(2)
    qc.rx(0.5, 0)
    qc.cx(0, 1)

    stripped = strip_rz_and_cx_from_start(qc)

    assert _gate_names(stripped) == ["rx", "cx"]


def test_start_drops_everything_when_qubit_never_sees_rx():
    qc = QuantumCircuit(2)
    qc.rz(0.5, 0)
    qc.cx(0, 1)

    stripped = strip_rz_and_cx_from_start(qc)

    assert _gate_names(stripped) == []


def test_start_keeps_rz_on_qubit_entangled_before_its_first_rx():
    # q1 is taken off |0> by cx(q0 -> q1) once q0 is active, before q1's own rx, so
    # the rz on q1 is now a real relative phase (not a global phase) and must be kept.
    qc = QuantumCircuit(2)
    qc.rx(0.5, 0)  # q0 active
    qc.cx(0, 1)  # kept: active control entangles q1
    qc.rz(0.3, 1)  # must be kept: q1 is no longer |0>
    qc.rx(0.5, 1)  # q1's first rx, later

    stripped = strip_rz_and_cx_from_start(qc)

    assert _gate_names(stripped) == ["rx", "cx", "rz", "rx"]


def test_start_keeps_cx_whose_control_was_entangled_before_its_rx():
    # q1 is entangled by cx(q0 -> q1) before q1's rx; a later cx(q1 -> q2) must be kept
    # because q1 is already active, even though neither q1 nor q2 has seen an rx yet.
    qc = QuantumCircuit(3)
    qc.rx(0.5, 0)
    qc.cx(0, 1)  # entangles q1
    qc.cx(1, 2)  # must be kept: q1 (control) is active

    stripped = strip_rz_and_cx_from_start(qc)

    assert _gate_names(stripped) == ["rx", "cx", "cx"]


# --- strip_rz_from_end: drop trailing RZ after each qubit's last RX or CX ---


def test_end_drops_trailing_rz_after_last_rx_or_cx_per_qubit():
    qc = QuantumCircuit(2)
    qc.rz(0.1, 0)  # kept: before q0's boundary (the cx below)
    qc.cx(0, 1)  # boundary for both q0 and q1
    qc.rz(0.2, 0)  # dropped: trailing on q0
    qc.rz(0.3, 1)  # dropped: trailing on q1

    stripped = strip_rz_from_end(qc)

    assert _gate_names(stripped) == ["rz", "cx"]


def test_end_rx_counts_as_boundary():
    qc = QuantumCircuit(1)
    qc.rz(0.1, 0)  # kept: before the rx boundary
    qc.rx(0.5, 0)  # boundary
    qc.rz(0.2, 0)  # dropped: trailing

    stripped = strip_rz_from_end(qc)

    assert _gate_names(stripped) == ["rz", "rx"]


def test_end_drops_all_rz_when_qubit_has_no_rx_or_cx():
    qc = QuantumCircuit(1)
    qc.rz(0.1, 0)
    qc.rz(0.2, 0)

    stripped = strip_rz_from_end(qc)

    assert _gate_names(stripped) == []


def test_strip_preserves_qubit_count():
    qc = QuantumCircuit(4)
    qc.rz(0.1, 0)
    assert strip_rz_and_cx_from_start(qc).num_qubits == 4
    assert strip_rz_from_end(qc).num_qubits == 4
