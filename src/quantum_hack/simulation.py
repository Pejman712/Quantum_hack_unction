"""Peak-bitstring estimators for quantum circuits.

Two strategies are provided:

* :func:`statevector_simulation` — exact, dense statevector simulation. Accurate
  but memory scales as ``2**num_qubits``, so it only works for small circuits.
* :func:`matrix_product_operators` — approximate, shot-based sampling via the matrix-product-state
  (MPS) simulator. Scales to many more qubits at the cost of being approximate.
"""

from typing import overload

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def _ranked(probs: dict[str, float], top_n: int) -> list[tuple[str, float]]:
    """Return the ``top_n`` highest-probability ``(bitstring, probability)`` pairs.

    Args:
        probs (dict[str, float]): Mapping of bitstring to probability.
        top_n (int): Number of pairs to return; must be a positive integer.

    Returns:
        list[tuple[str, float]]: Up to ``top_n`` pairs sorted by descending probability.

    Raises:
        ValueError: If ``top_n`` is less than 1.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n}")
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def _print_ranking(ranking: list[tuple[str, float]]) -> None:
    """Print a ranked list of ``(bitstring, probability)`` pairs, highest first.

    Args:
        ranking (list[tuple[str, float]]): Bitstring/probability pairs to print.

    Returns:
        None
    """
    for rank, (bitstring, prob) in enumerate(ranking, start=1):
        print(f"{rank}. {bitstring}  {prob:.4f}")


@overload
def statevector_simulation(
    qc: QuantumCircuit, verbose: bool = ..., *, top_n: None = ...
) -> tuple[str, float]: ...
@overload
def statevector_simulation(
    qc: QuantumCircuit, verbose: bool = ..., *, top_n: int
) -> list[tuple[str, float]]: ...
def statevector_simulation(
    qc: QuantumCircuit, verbose: bool = False, *, top_n: int | None = None
) -> tuple[str, float] | list[tuple[str, float]]:
    """Return the most likely measurement bitstring via exact statevector simulation.

    Measurements are removed first because :class:`~qiskit.quantum_info.Statevector`
    operates on a unitary circuit acting on the all-zero state. The input circuit is
    left untouched.

    Args:
        qc (QuantumCircuit): Circuit to simulate.
        verbose (bool): If ``True``, print the result(s).
        top_n (int | None): If ``None`` (default), return only the single peak. If a
            positive integer, return the ``top_n`` most likely bitstrings instead.

    Returns:
        tuple[str, float] | list[tuple[str, float]]: When ``top_n`` is ``None``, the
        peak bitstring and its exact probability. When ``top_n`` is an int, a list of
        up to ``top_n`` ``(bitstring, probability)`` pairs sorted by descending probability.

    Raises:
        ValueError: If ``top_n`` is given and is less than 1.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    assert qc_no_meas is not None  # inplace=False always returns a new circuit

    sv = Statevector.from_instruction(qc_no_meas)

    probs = sv.probabilities_dict()  # bitstring -> probability

    if top_n is not None:
        ranking = _ranked(probs, top_n)
        if verbose:
            _print_ranking(ranking)
        return ranking

    peak_bitstring = max(probs, key=lambda b: probs[b])
    peak_prob = probs[peak_bitstring]

    if verbose:
        print("Most likely bitstring:", peak_bitstring)
        print("Peak probability:", peak_prob)
    return peak_bitstring, peak_prob


@overload
def matrix_product_operators(
    qc: QuantumCircuit,
    shots: int = ...,
    bond_dim: int = ...,
    verbose: bool = ...,
    *,
    top_n: None = ...,
) -> tuple[str, float]: ...
@overload
def matrix_product_operators(
    qc: QuantumCircuit,
    shots: int = ...,
    bond_dim: int = ...,
    verbose: bool = ...,
    *,
    top_n: int,
) -> list[tuple[str, float]]: ...
def matrix_product_operators(
    qc: QuantumCircuit,
    shots: int = 4096,
    bond_dim: int = 64,
    verbose: bool = False,
    *,
    top_n: int | None = None,
) -> tuple[str, float] | list[tuple[str, float]]:
    """Estimate the most likely bitstring via shot-based MPS sampling.

    The input circuit is left untouched: any final measurements are removed and a
    full ``measure_all`` is added on a copy before sampling.

    Args:
        qc (QuantumCircuit): Circuit to sample.
        shots (int): Number of measurement shots to draw.
        bond_dim (int): Maximum MPS bond dimension.
        verbose (bool): If ``True``, print the result(s).
        top_n (int | None): If ``None`` (default), return only the single peak. If a
            positive integer, return the ``top_n`` most sampled bitstrings instead.

    Returns:
        tuple[str, float] | list[tuple[str, float]]: When ``top_n`` is ``None``, the
        estimated peak bitstring and its estimated probability. When ``top_n`` is an
        int, a list of up to ``top_n`` ``(bitstring, probability)`` pairs sorted by
        descending estimated probability.

    Raises:
        ValueError: If ``top_n`` is given and is less than 1.
    """
    qc_copy = qc.remove_final_measurements(inplace=False)
    assert qc_copy is not None  # inplace=False always returns a new circuit
    qc_copy.measure_all()

    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
    )

    qc_t = transpile(qc_copy, sim)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()
    probs = {bitstring: count / shots for bitstring, count in counts.items()}

    if top_n is not None:
        ranking = _ranked(probs, top_n)
        if verbose:
            _print_ranking(ranking)
        return ranking

    peak_bitstring = max(probs, key=lambda b: probs[b])
    peak_prob_est = probs[peak_bitstring]

    if verbose:
        print("Estimated peak bitstring:", peak_bitstring)
        print("Estimated peak probability:", peak_prob_est)
    return peak_bitstring, peak_prob_est
