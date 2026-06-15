"""
Parse QASM files into structured representations for the transformer.

Gate vocabulary covers all gate types found across the circuit dataset:
  rx, rz  — single-qubit rotations with a continuous angle parameter
  cx      — CNOT (2-qubit, no angle)
  swap    — SWAP (2-qubit, no angle)

encode_circuit() returns everything the model needs:
  - n_qubits          : int
  - gate_sequence     : list of [type_id, cos(θ), sin(θ), q0, q1]
                        q1 = n_qubits (padding sentinel) for 1-qubit gates
  - interaction_matrix: float32 (n_qubits × n_qubits) symmetric matrix
                        entry [i,j] = number of 2-qubit gates acting on (i,j)
                        used as additive attention bias in the transformer
"""

import math
import re

import numpy as np

GATE_VOCAB = {"rx": 0, "rz": 1, "cx": 2, "swap": 3}
GATE_VOCAB_SIZE = len(GATE_VOCAB)

_PI = math.pi

# matches: rx(3.14) q[2];  or  rx(pi) q[0];  or  rz(-1.234e-3) q[5];
_RE_1Q = re.compile(r"^(rx|rz)\(([^)]+)\)\s+q\[(\d+)\]")
_RE_CX = re.compile(r"^cx\s+q\[(\d+)\],q\[(\d+)\]")
_RE_SWAP = re.compile(r"^swap\s+q\[(\d+)\],q\[(\d+)\]")
_RE_QREG = re.compile(r"^qreg q\[(\d+)\]")


def _parse_angle(s):
    """Parse an angle string that may contain 'pi' as a literal."""
    s = s.strip().replace("pi", str(_PI))
    return float(eval(s))  # noqa: S307 — inputs are controlled QASM files, not user text


def parse_qasm(qasm_path):
    """
    Parse a QASM file into (n_qubits, gates).

    Each gate dict has:
        type     : str  — gate name
        type_id  : int  — index into GATE_VOCAB
        angle    : float — rotation angle (0.0 for cx/swap)
        qubits   : list[int]
    """
    n_qubits = None
    gates = []

    with open(qasm_path) as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line or line.startswith("//") or line.startswith("OPENQASM") or line.startswith("include"):
                continue

            m = _RE_QREG.match(line)
            if m:
                n_qubits = int(m.group(1))
                continue

            m = _RE_1Q.match(line)
            if m:
                gate_type = m.group(1)
                angle = _parse_angle(m.group(2))
                q0 = int(m.group(3))
                gates.append({"type": gate_type, "type_id": GATE_VOCAB[gate_type],
                               "angle": angle, "qubits": [q0]})
                continue

            m = _RE_CX.match(line)
            if m:
                q0, q1 = int(m.group(1)), int(m.group(2))
                gates.append({"type": "cx", "type_id": GATE_VOCAB["cx"],
                               "angle": 0.0, "qubits": [q0, q1]})
                continue

            m = _RE_SWAP.match(line)
            if m:
                q0, q1 = int(m.group(1)), int(m.group(2))
                gates.append({"type": "swap", "type_id": GATE_VOCAB["swap"],
                               "angle": 0.0, "qubits": [q0, q1]})
                continue

    return n_qubits, gates


def build_interaction_matrix(n_qubits, gates):
    """
    Symmetric (n_qubits × n_qubits) float32 matrix.
    Entry [i,j] = number of 2-qubit gates acting on the pair (i,j).
    Diagonal is always zero.
    """
    m = np.zeros((n_qubits, n_qubits), dtype=np.float32)
    for gate in gates:
        if len(gate["qubits"]) == 2:
            i, j = gate["qubits"]
            m[i, j] += 1.0
            m[j, i] += 1.0
    return m


def build_typed_matrices(n_qubits, gates):
    """
    Split the interaction matrix into separate cx and swap counts.

    Returns
    -------
    cx_matrix   : np.ndarray (n_qubits, n_qubits), float32
    swap_matrix : np.ndarray (n_qubits, n_qubits), float32
    """
    cx_m   = np.zeros((n_qubits, n_qubits), dtype=np.float32)
    swap_m = np.zeros((n_qubits, n_qubits), dtype=np.float32)
    for gate in gates:
        if len(gate["qubits"]) == 2:
            i, j = gate["qubits"]
            if gate["type"] == "cx":
                cx_m[i, j] += 1.0
                cx_m[j, i] += 1.0
            elif gate["type"] == "swap":
                swap_m[i, j] += 1.0
                swap_m[j, i] += 1.0
    return cx_m, swap_m


def _build_gate_sequence(gates, n_qubits):
    seq = []
    for gate in gates:
        q0 = gate["qubits"][0]
        q1 = gate["qubits"][1] if len(gate["qubits"]) > 1 else n_qubits
        theta = gate["angle"]
        seq.append([gate["type_id"], math.cos(theta), math.sin(theta), q0, q1])
    return seq


def encode_circuit(qasm_path):
    """
    Full encoding of a QASM file.

    Returns
    -------
    n_qubits : int
    gate_sequence : list of [type_id, cos(θ), sin(θ), q0, q1]
        q1 == n_qubits acts as the padding sentinel for 1-qubit gates.
    interaction_matrix : np.ndarray, shape (n_qubits, n_qubits), dtype float32
    """
    n_qubits, gates = parse_qasm(qasm_path)
    interaction = build_interaction_matrix(n_qubits, gates)
    return n_qubits, _build_gate_sequence(gates, n_qubits), interaction


def encode_circuit_typed(qasm_path):
    """
    Like encode_circuit(), but returns separate cx and swap interaction matrices.

    This enables the CircuitConditioner to weight the two gate types independently,
    since cx creates entanglement while swap only relocates it.

    Returns
    -------
    n_qubits         : int
    gate_sequence    : list of [type_id, cos(θ), sin(θ), q0, q1]
    cx_matrix        : np.ndarray (n_qubits, n_qubits), float32 — CNOT coupling counts
    swap_matrix      : np.ndarray (n_qubits, n_qubits), float32 — SWAP coupling counts
    interaction_matrix : np.ndarray (n_qubits, n_qubits), float32 — cx + swap
    """
    n_qubits, gates = parse_qasm(qasm_path)
    cx_m, swap_m = build_typed_matrices(n_qubits, gates)
    return n_qubits, _build_gate_sequence(gates, n_qubits), cx_m, swap_m, cx_m + swap_m
