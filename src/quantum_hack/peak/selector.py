"""Profile a circuit and pick an ordered chain of peak-finding methods to try.

The peak is always found with the matrix-product-state (MPS) simulator and the marginal
"Z-expectation" vote derived from the same sampling — never an exact statevector (it does not
scale and offers no benefit here). The selector is deliberately cheap (no simulation): it inspects
qubit and two-qubit-gate counts and returns a cheap-to-expensive chain, optionally ending with the
tensor-network amplitude backend (quimb) as a refinement/verification oracle when it is feasible.
"""

from enum import StrEnum

from qiskit import QuantumCircuit

# The tensor-network amplitude oracle (quimb) is appended as a refinement step when feasible: with
# a GPU, or for circuits up to this many qubits on CPU.
TN_MAX_QUBITS = 44
# MPS sampling is reliable until roughly this many two-qubit gates; beyond it the marginal
# vote (per-qubit majority) leads and MPS is only a cross-check.
MPS_RELIABLE_MAX_2Q = 1000

TWO_QUBIT_GATE_NAMES = frozenset({"cx", "cz", "cy", "swap", "iswap", "ecr", "rzz", "rxx", "ryy"})


class Method(StrEnum):
    """A peak-finding method. Values double as the ``method`` column in result CSVs."""

    MPS = "MPS"
    MARGINAL = "marginal"
    GREEDY_REFINE = "greedy_refine"
    TENSOR_NETWORK = "tensor_network"


class CircuitProfile:
    """Cheap structural summary of a circuit, used to choose methods.

    Attributes:
        num_qubits (int): Number of qubits.
        num_two_qubit_gates (int): Count of two-qubit gates (cx, swap, cz, ...).
        num_gates (int): Total instruction count (excluding final measurements/barriers).
        depth (int): Circuit depth.
    """

    __slots__ = ("num_qubits", "num_two_qubit_gates", "num_gates", "depth")

    def __init__(
        self, num_qubits: int, num_two_qubit_gates: int, num_gates: int, depth: int
    ) -> None:
        """Initialise the profile.

        Args:
            num_qubits (int): Number of qubits.
            num_two_qubit_gates (int): Count of two-qubit gates.
            num_gates (int): Total instruction count.
            depth (int): Circuit depth.
        """
        self.num_qubits = num_qubits
        self.num_two_qubit_gates = num_two_qubit_gates
        self.num_gates = num_gates
        self.depth = depth

    def __repr__(self) -> str:
        """Return a debug representation.

        Returns:
            str: A readable summary of all fields.
        """
        return (
            f"CircuitProfile(num_qubits={self.num_qubits}, "
            f"num_two_qubit_gates={self.num_two_qubit_gates}, "
            f"num_gates={self.num_gates}, depth={self.depth})"
        )


def profile_circuit(qc: QuantumCircuit) -> CircuitProfile:
    """Summarise ``qc`` for method selection without simulating it.

    Final measurements and barriers are ignored so the counts reflect the unitary.

    Args:
        qc (QuantumCircuit): Circuit to profile.

    Returns:
        CircuitProfile: Structural summary (qubits, two-qubit-gate count, gate count, depth).
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    assert qc_no_meas is not None  # inplace=False always returns a new circuit
    ops = qc_no_meas.count_ops()
    two_q = sum(count for name, count in ops.items() if name in TWO_QUBIT_GATE_NAMES)
    num_gates = sum(count for name, count in ops.items() if name not in ("barrier",))
    return CircuitProfile(
        num_qubits=qc_no_meas.num_qubits,
        num_two_qubit_gates=two_q,
        num_gates=num_gates,
        depth=qc_no_meas.depth(),
    )


def select_method(profile: CircuitProfile, *, gpu_available: bool = False) -> list[Method]:
    """Return an ordered, cheap-to-expensive chain of methods to try for this circuit.

    Always MPS-led (never exact statevector). For deep circuits the per-qubit marginal vote leads,
    since MPS sampling is unreliable there. The tensor-network amplitude oracle is appended for
    refinement/verification when feasible.

    Args:
        profile (CircuitProfile): The circuit's structural summary.
        gpu_available (bool): Whether GPU Aer (LUMI ``standard-g``) is available, which enables
            GPU-accelerated MPS and the tensor-network oracle for large circuits.

    Returns:
        list[Method]: Methods to attempt in order. The orchestrator runs them until it has a
        trustworthy result, also using cheaper methods for cross-validation.
    """
    n = profile.num_qubits
    g2 = profile.num_two_qubit_gates

    if g2 <= MPS_RELIABLE_MAX_2Q:
        chain = [Method.MPS, Method.MARGINAL, Method.GREEDY_REFINE]
    else:
        # Deep circuits: MPS sampling is unreliable, so the per-qubit marginal vote leads.
        chain = [Method.MARGINAL, Method.MPS, Method.GREEDY_REFINE]

    # The tensor-network amplitude oracle (quimb) refines/verifies when feasible.
    if gpu_available or n <= TN_MAX_QUBITS:
        chain.append(Method.TENSOR_NETWORK)
    return chain
