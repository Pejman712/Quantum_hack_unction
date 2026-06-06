"""The marginal "Z-expectation" attack and greedy bit-flip refinement.

For a peaked circuit the per-qubit single-bit marginals usually point at the peak even when
argmax-over-samples does not (the peak can be under-sampled while each qubit's *marginal* is
still biased the right way). For qubit ``i`` we take the sign of ``<Z_i>``: a negative
expectation means qubit ``i`` is more likely ``1``. The resulting candidate is then polished by
:func:`greedy_refine`, a steepest-ascent hill-climb over single-bit flips that calls an
amplitude oracle (exact statevector for small circuits, a tensor-network contraction for large
ones).
"""

from collections.abc import Callable, Iterable, Mapping

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_hack.simulation import statevector_probability

# An amplitude oracle: given a circuit and a Qiskit-ordered bitstring, return its probability.
Oracle = Callable[[QuantumCircuit, str], float]


def exact_z_marginals(qc: QuantumCircuit) -> list[float]:
    """Return the exact single-qubit ``<Z_i>`` expectation for every qubit.

    Computed from an exact statevector, so this is only practical for small circuits. The
    input circuit is left untouched.

    Args:
        qc (QuantumCircuit): Circuit to analyse.

    Returns:
        list[float]: ``<Z_i>`` in ``[-1, 1]`` for each qubit ``i`` (index 0 is qubit 0). A
        positive value means qubit ``i`` is more likely ``0``, negative means more likely ``1``.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    assert qc_no_meas is not None  # inplace=False always returns a new circuit
    n = qc_no_meas.num_qubits
    # Reshape the probability vector to (2,)*n and marginalise in one vectorised pass; this is far
    # faster than n separate Statevector.probabilities([i]) calls on a wide circuit. Reshape axis 0
    # is the most-significant qubit (n-1), so qubit i lives on axis (n-1-i).
    probs = Statevector.from_instruction(qc_no_meas).probabilities().reshape((2,) * n)
    marginals: list[float] = []
    for i in range(n):
        axis = n - 1 - i
        other = tuple(a for a in range(n) if a != axis)
        p1 = float(probs.sum(axis=other)[1]) if other else float(probs[1])
        marginals.append(1.0 - 2.0 * p1)
    return marginals


def z_marginals_from_counts(counts: Mapping[str, int], num_qubits: int) -> list[float]:
    """Estimate single-qubit ``<Z_i>`` from sampled measurement counts.

    Lets the marginal attack reuse the same MPS sampling used to find the most-frequent
    bitstring, so it works on circuits too large for an exact statevector.

    Args:
        counts (Mapping[str, int]): Bitstring-to-count mapping (Qiskit ordering), e.g. from
            :func:`~quantum_hack.simulation.mps_sample_counts`.
        num_qubits (int): Number of qubits (bitstring width).

    Returns:
        list[float]: Estimated ``<Z_i>`` in ``[-1, 1]`` per qubit (index 0 is qubit 0). Returns
        all zeros when ``counts`` is empty.
    """
    total = sum(counts.values())
    if total == 0:
        return [0.0] * num_qubits
    ones = [0] * num_qubits
    for bitstring, count in counts.items():
        for i in range(num_qubits):
            if bitstring[num_qubits - 1 - i] == "1":
                ones[i] += count
    return [1.0 - 2.0 * (ones[i] / total) for i in range(num_qubits)]


def marginal_bitstring(marginals: Iterable[float]) -> str:
    """Build the most-likely bitstring from per-qubit ``<Z_i>`` expectations.

    Args:
        marginals (Iterable[float]): ``<Z_i>`` per qubit, index 0 is qubit 0 (as returned by
            :func:`exact_z_marginals` or :func:`z_marginals_from_counts`).

    Returns:
        str: The candidate bitstring in Qiskit ordering (rightmost char is qubit 0); qubit
        ``i`` is ``1`` when ``<Z_i> < 0``, else ``0``.
    """
    bits = ["1" if z < 0 else "0" for z in marginals]
    return "".join(reversed(bits))


def least_confident_qubits(marginals: Iterable[float], k: int) -> list[int]:
    """Return the indices of the ``k`` qubits whose ``<Z_i>`` is closest to zero.

    These are the bits the marginal vote is least sure about, so refinement should flip them
    first; restricting refinement to them caps oracle calls on wide circuits.

    Args:
        marginals (Iterable[float]): ``<Z_i>`` per qubit (index 0 is qubit 0).
        k (int): Number of qubit indices to return.

    Returns:
        list[int]: Up to ``k`` qubit indices sorted by ascending ``|<Z_i>|``.
    """
    values = list(marginals)
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    return order[: max(0, k)]


def _flip_bit(bitstring: str, qubit: int, num_qubits: int) -> str:
    """Return ``bitstring`` with the bit for ``qubit`` flipped.

    Args:
        bitstring (str): Qiskit-ordered bitstring.
        qubit (int): Qubit index (0 is the rightmost char).
        num_qubits (int): Bitstring width.

    Returns:
        str: A new bitstring with that one bit toggled.
    """
    pos = num_qubits - 1 - qubit
    flipped = "1" if bitstring[pos] == "0" else "0"
    return f"{bitstring[:pos]}{flipped}{bitstring[pos + 1:]}"


def greedy_refine(
    qc: QuantumCircuit,
    seed: str,
    *,
    oracle: Oracle = statevector_probability,
    candidate_qubits: Iterable[int] | None = None,
    max_iters: int = 256,
) -> tuple[str, float]:
    """Hill-climb from ``seed`` to a local probability maximum by flipping single bits.

    Steepest ascent: each sweep evaluates every single-bit-flip neighbour (restricted to
    ``candidate_qubits`` if given) and moves to the most probable improvement, stopping at a
    local maximum or after ``max_iters`` moves.

    Args:
        qc (QuantumCircuit): Circuit whose peak is sought.
        seed (str): Starting bitstring in Qiskit ordering; its length must equal the qubit count.
        oracle (Oracle): Probability oracle ``(qc, bitstring) -> float``; defaults to the exact
            :func:`~quantum_hack.simulation.statevector_probability`. Supply a tensor-network
            oracle for large circuits.
        candidate_qubits (Iterable[int] | None): Qubit indices allowed to flip; ``None`` (default)
            considers all qubits. Restrict to the least-confident qubits to cap oracle calls.
        max_iters (int): Maximum number of improving moves before stopping.

    Returns:
        tuple[str, float]: The refined bitstring and its oracle probability (a local maximum
        over the considered flips).

    Raises:
        ValueError: If ``seed`` length does not match the circuit's qubit count.
    """
    n = qc.num_qubits
    if len(seed) != n:
        raise ValueError(f"seed must be {n} chars, got {len(seed)}")
    qubits = list(range(n)) if candidate_qubits is None else list(candidate_qubits)

    best, best_p = seed, oracle(qc, seed)
    for _ in range(max_iters):
        winner, winner_p = None, best_p
        for i in qubits:
            candidate = _flip_bit(best, i, n)
            p = oracle(qc, candidate)
            if p > winner_p:
                winner, winner_p = candidate, p
        if winner is None:
            break
        best, best_p = winner, winner_p
    return best, best_p
