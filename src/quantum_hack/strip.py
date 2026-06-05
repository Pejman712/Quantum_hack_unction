"""Helpers for stripping leading/trailing gates from a circuit, per qubit."""

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction


def _qubit_indices(qc: QuantumCircuit, instruction: CircuitInstruction) -> list[int]:
    """Return the qubit indices an instruction acts on, in order.

    Args:
        qc (QuantumCircuit): Circuit the instruction belongs to.
        instruction (CircuitInstruction): Instruction whose qubits to resolve.

    Returns:
        list[int]: The integer qubit indices the instruction acts on.
    """
    return [qc.find_bit(q).index for q in instruction.qubits]


def _first_rx_per_qubit(qc: QuantumCircuit) -> dict[int, int]:
    """Map each qubit to the data index of its first RX gate.

    Args:
        qc (QuantumCircuit): Circuit to scan.

    Returns:
        dict[int, int]: Mapping of qubit index to the data index of its first RX gate.
    """
    first: dict[int, int] = {}
    for i, instruction in enumerate(qc.data):
        if instruction.operation.name == "rx":
            for q in _qubit_indices(qc, instruction):
                first.setdefault(q, i)
    return first


def _last_rx_or_cx_per_qubit(qc: QuantumCircuit) -> dict[int, int]:
    """Map each qubit to the data index of its last RX or CX gate.

    Args:
        qc (QuantumCircuit): Circuit to scan.

    Returns:
        dict[int, int]: Mapping of qubit index to the data index of its last RX or CX gate.
    """
    last: dict[int, int] = {}
    for i, instruction in enumerate(qc.data):
        if instruction.operation.name in ("rx", "cx"):
            for q in _qubit_indices(qc, instruction):
                last[q] = i
    return last


def strip_rz_and_cx_from_start(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``qc`` with leading RZ/CX gates removed, per qubit.

    On each qubit, an RZ/CX gate that occurs before that qubit's first RX gate is
    considered "leading" and dropped. A two-qubit CX is dropped only when it
    precedes the first RX on *both* of its qubits; otherwise it lies after the
    boundary on at least one wire and is kept. If a qubit never sees an RX, every
    RZ/CX touching only that qubit is treated as leading.

    Args:
        qc (QuantumCircuit): Circuit to strip.

    Returns:
        QuantumCircuit: A new circuit with leading RZ/CX gates removed.
    """
    first_rx = _first_rx_per_qubit(qc)
    after_end = len(qc.data)  # sentinel "first RX index" for qubits with no RX
    new_qc = QuantumCircuit(qc.num_qubits)
    for i, instruction in enumerate(qc.data):
        op = instruction.operation
        indices = _qubit_indices(qc, instruction)
        if op.name == "rz" and i < first_rx.get(indices[0], after_end):
            continue
        if op.name == "cx" and all(i < first_rx.get(q, after_end) for q in indices):
            continue
        new_qc.append(op, indices)
    return new_qc


def strip_rz_from_end(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``qc`` with trailing RZ gates removed, per qubit.

    On each qubit, an RZ gate that occurs after that qubit's last RX or CX gate is
    considered "trailing" and dropped. If a qubit has no RX/CX gate, every RZ on it
    is treated as trailing.

    Args:
        qc (QuantumCircuit): Circuit to strip.

    Returns:
        QuantumCircuit: A new circuit with trailing RZ gates removed.
    """
    last_boundary = _last_rx_or_cx_per_qubit(qc)
    new_qc = QuantumCircuit(qc.num_qubits)
    for i, instruction in enumerate(qc.data):
        op = instruction.operation
        indices = _qubit_indices(qc, instruction)
        if op.name == "rz" and i > last_boundary.get(indices[0], -1):
            continue
        new_qc.append(op, indices)
    return new_qc
