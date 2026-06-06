"""Tensor-network amplitude/marginal backend via ``quimb`` + ``cotengra`` (lazy-imported).

This is the exact scale path for large circuits: contract the tensor network to read off the
amplitude of a *specific* candidate bitstring (the oracle for :func:`greedy_refine`) or the
single-qubit ``<Z_i>`` marginals — without ever building the full ``2**n`` statevector.

``quimb`` and ``cotengra`` are optional and imported only inside the functions, so importing
``quantum_hack`` works without them (mirroring the ``mqt.qcec`` pattern in ``verification``).
Install them into the LUMI Qiskit container with ``pip install quimb cotengra`` (extra
``quantum-hack-unction[tn]``). These functions are exercised on the cluster, not in local CI.

Bit-order note: ``quimb`` orders qubits ``q0..q_{n-1}`` (leftmost char is qubit 0); Qiskit
orders ``q_{n-1}..q0`` (rightmost char is qubit 0). Every public function here takes and returns
**Qiskit-ordered** bitstrings and reverses once at the ``quimb`` boundary.
"""

import numpy as np
from qiskit import QuantumCircuit


def quimb_available() -> bool:
    """Return whether the ``quimb`` tensor-network backend can be imported.

    Returns:
        bool: ``True`` if ``import quimb`` succeeds, else ``False``.
    """
    try:
        import quimb.tensor  # noqa: F401
    except ImportError:
        return False
    return True


def torch_gpu_available() -> bool:
    """Return ``True`` if PyTorch with CUDA is importable and has a visible GPU.

    Used to decide whether to move circuit tensors to CUDA before contraction.

    Returns:
        bool: ``True`` only when ``import torch`` succeeds and ``torch.cuda.is_available()``
        is ``True``.
    """
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


def _move_to_gpu(circ) -> None:
    """Move every tensor array inside ``circ`` to CUDA in-place (complex128).

    ``circ`` must be a ``quimb.tensor.TensorNetwork`` subclass (e.g. ``Circuit``).
    Keeping complex128 (float64) avoids numerical drift in amplitude oracles; the gate
    tensors are small so the memory cost is negligible.

    Args:
        circ: A ``quimb.tensor.TensorNetwork`` whose arrays to move to CUDA.

    Returns:
        None
    """
    import torch
    circ.apply_to_arrays(
        lambda x: torch.as_tensor(np.asarray(x), device="cuda", dtype=torch.complex128)
    )


def _to_quimb_circuit(qc: QuantumCircuit):  # noqa: ANN202 (quimb type unavailable at import)
    """Convert a Qiskit circuit to a ``quimb`` tensor-network circuit via OpenQASM 2.

    Final measurements are removed first (the contraction needs a unitary). Requires ``quimb``.

    Args:
        qc (QuantumCircuit): Circuit to convert.

    Returns:
        quimb.tensor.Circuit: The equivalent ``quimb`` circuit.
    """
    import quimb.tensor as qtn
    from qiskit.qasm2 import dumps

    qc_no_meas = qc.remove_final_measurements(inplace=False)
    assert qc_no_meas is not None  # inplace=False always returns a new circuit
    return qtn.Circuit.from_openqasm2_str(dumps(qc_no_meas))


def amplitude(
    qc: QuantumCircuit, bitstring: str, *, optimize: str = "auto-hq", device: str = "CPU"
) -> complex:
    """Return the amplitude ``<bitstring|U|0...0>`` via tensor-network contraction.

    Args:
        qc (QuantumCircuit): Circuit whose amplitude to evaluate.
        bitstring (str): Target bitstring in Qiskit ordering (rightmost char is qubit 0).
        optimize (str): ``cotengra``/``opt_einsum`` contraction-path preset (e.g. ``"auto-hq"``).
        device (str): ``"GPU"`` moves all tensors to CUDA and contracts with PyTorch (requires
            ``torch`` with CUDA). Any other value uses the default numpy backend on CPU.

    Returns:
        complex: The complex amplitude of ``bitstring``.
    """
    circ = _to_quimb_circuit(qc)
    quimb_bitstring = bitstring[::-1]  # Qiskit q_{n-1}..q0 -> quimb q0..q_{n-1}
    if device.upper() == "GPU" and torch_gpu_available():
        _move_to_gpu(circ)
        return complex(circ.amplitude(quimb_bitstring, optimize=optimize, backend="torch"))
    return complex(circ.amplitude(quimb_bitstring, optimize=optimize))


def probability(
    qc: QuantumCircuit, bitstring: str, *, optimize: str = "auto-hq", device: str = "CPU"
) -> float:
    """Return the probability of ``bitstring`` via tensor-network contraction.

    Suitable as the :func:`~quantum_hack.peak.marginals.greedy_refine` oracle for circuits too
    large for an exact statevector.

    Args:
        qc (QuantumCircuit): Circuit whose probability to evaluate.
        bitstring (str): Target bitstring in Qiskit ordering (rightmost char is qubit 0).
        optimize (str): Contraction-path preset.
        device (str): ``"GPU"`` runs the contraction on CUDA via PyTorch; CPU otherwise.

    Returns:
        float: ``|amplitude|**2`` in ``[0, 1]``.
    """
    amp = amplitude(qc, bitstring, optimize=optimize, device=device)
    return float((amp.conjugate() * amp).real)


def single_qubit_marginals_z(
    qc: QuantumCircuit, *, optimize: str = "auto-hq", device: str = "CPU"
) -> list[float]:
    """Return single-qubit ``<Z_i>`` expectations via tensor-network contraction.

    Args:
        qc (QuantumCircuit): Circuit to analyse.
        optimize (str): Contraction-path preset.
        device (str): ``"GPU"`` runs contractions on CUDA via PyTorch; CPU otherwise.

    Returns:
        list[float]: ``<Z_i>`` in ``[-1, 1]`` per qubit, index 0 is qubit 0 (Qiskit ordering),
        matching :func:`~quantum_hack.peak.marginals.exact_z_marginals`.
    """
    import quimb as qu

    circ = _to_quimb_circuit(qc)
    if device.upper() == "GPU" and torch_gpu_available():
        import torch
        _move_to_gpu(circ)
        z = torch.as_tensor(np.asarray(qu.pauli("Z")), device="cuda", dtype=torch.complex128)
        return [
            float(circ.local_expectation(z, i, optimize=optimize, backend="torch").real)
            for i in range(qc.num_qubits)
        ]
    z = qu.pauli("Z")
    return [
        float(circ.local_expectation(z, i, optimize=optimize).real)
        for i in range(qc.num_qubits)
    ]


def contraction_width(qc: QuantumCircuit, *, optimize: str = "auto-hq") -> float:
    """Return the log2 width of the largest intermediate tensor for an amplitude contraction.

    A proxy for tensor-network difficulty: a width above ~30 means a single amplitude needs more
    than ~2**30 complex numbers in memory, so contraction should be sliced or avoided. Best-effort
    against the ``quimb`` rehearsal API; returns ``inf`` if the width cannot be determined.

    Args:
        qc (QuantumCircuit): Circuit to analyse.
        optimize (str): Contraction-path preset.

    Returns:
        float: ``log2`` of the largest intermediate tensor size, or ``inf`` if unknown.
    """
    circ = _to_quimb_circuit(qc)
    zeros = "0" * qc.num_qubits
    try:
        info = circ.amplitude_rehearse(zeros, optimize=optimize)
        tree = info["tree"]
        return float(tree.contraction_width())
    except (KeyError, AttributeError, TypeError):
        return float("inf")
