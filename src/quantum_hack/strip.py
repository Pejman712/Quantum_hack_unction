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

    A gate is "leading" while every qubit it touches is still ``|0>`` (up to global
    phase): an RZ on such a qubit is only a global phase, and a CX whose control is
    still ``|0>`` is the identity, so both are unobservable and dropped. A qubit
    leaves ``|0>`` either at its first RX *or* when it becomes the target of a kept
    CX (one whose control is already active), which entangles it; RZ/CX after that
    point are kept. A single forward pass tracks which qubits are still ``|0>``.

    Args:
        qc (QuantumCircuit): Circuit to strip.

    Returns:
        QuantumCircuit: A new circuit with the leading RZ/CX gates removed, equivalent
        to ``qc`` up to global phase.
    """
    active: set[int] = set()  # qubits no longer guaranteed to be |0> (up to phase)
    new_qc = QuantumCircuit(qc.num_qubits)
    for instruction in qc.data:
        op = instruction.operation
        indices = _qubit_indices(qc, instruction)
        # On a still-|0> qubit, rz is a global phase and a cx (control |0>) is identity.
        if op.name == "rz" and indices[0] not in active:
            continue
        if op.name == "cx" and indices[0] not in active:
            continue
        new_qc.append(op, indices)
        if op.name == "rx":
            active.add(indices[0])  # rx takes its qubit off |0>
        elif op.name == "cx":
            active.add(indices[1])  # kept cx (active control) entangles its target
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
