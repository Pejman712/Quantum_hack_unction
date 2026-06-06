"""Peak-bitstring estimators for quantum circuits.

Two strategies are provided:

* :func:`statevector_simulation` — exact, dense statevector simulation. Accurate
  but memory scales as ``2**num_qubits``, so it only works for small circuits.
* :func:`matrix_product_operators` — approximate, shot-based sampling via the matrix-product-state
  (MPS) simulator. Scales to many more qubits at the cost of being approximate.
"""

from typing import overload

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def probe_gpu() -> bool:
    """Return ``True`` if AerSimulator can actually execute a circuit on GPU.

    ``available_devices()`` only reflects CUDA hardware presence; this function runs a
    real 1-qubit job so it catches CPU-only Aer wheels that list the GPU but cannot use it.
    The result is not cached — callers should call this once at startup.

    Returns:
        bool: ``True`` if GPU simulation works end-to-end, ``False`` otherwise.
    """
    try:
        from qiskit import QuantumCircuit as _QC

        sim = AerSimulator(method="statevector", device="GPU")
        qc = _QC(1)
        qc.measure_all()
        qc_t = transpile(qc, basis_gates=_simulator_basis_gates(sim))
        sim.run(qc_t, shots=1).result()
        return True
    except Exception:
        return False


def _simulator_basis_gates(sim: AerSimulator) -> list[str]:
    """Return the simulator's natively supported standard gate names.

    Filters the backend's advertised basis down to gates the transpiler accepts
    as ``basis_gates`` (dropping non-standard entries such as ``kraus`` and
    ``quantum_channel``). Transpiling against this basis decomposes unsupported
    gates without binding to the backend target, whose memory-based qubit ceiling
    is meaningless for the MPS method and otherwise rejects wide circuits.

    Args:
        sim (AerSimulator): The simulator whose basis to inspect.

    Returns:
        list[str]: Standard gate names supported by ``sim``.
    """
    standard = set(get_standard_gate_name_mapping())
    return [gate for gate in sim.configuration().basis_gates if gate in standard]


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
    qc: QuantumCircuit, verbose: bool = ..., *, top_n: None = ..., device: str = ...
) -> tuple[str, float]: ...
@overload
def statevector_simulation(
    qc: QuantumCircuit, verbose: bool = ..., *, top_n: int, device: str = ...
) -> list[tuple[str, float]]: ...
def statevector_simulation(
    qc: QuantumCircuit, verbose: bool = False, *, top_n: int | None = None, device: str = "CPU"
) -> tuple[str, float] | list[tuple[str, float]]:
    """Return the most likely measurement bitstring via exact statevector simulation.

    Measurements are removed first because the simulation operates on a unitary circuit
    acting on the all-zero state. The input circuit is left untouched.

    Args:
        qc (QuantumCircuit): Circuit to simulate.
        verbose (bool): If ``True``, print the result(s).
        top_n (int | None): If ``None`` (default), return only the single peak. If a
            positive integer, return the ``top_n`` most likely bitstrings instead.
        device (str): ``"CPU"`` (default) uses the exact in-process
            :class:`~qiskit.quantum_info.Statevector`. Any other value (e.g. ``"GPU"``) runs
            Qiskit Aer's statevector simulator on that device (used on LUMI's GPUs); Aer must
            be built with support for it or the call raises.

    Returns:
        tuple[str, float] | list[tuple[str, float]]: When ``top_n`` is ``None``, the
        peak bitstring and its exact probability. When ``top_n`` is an int, a list of
        up to ``top_n`` ``(bitstring, probability)`` pairs sorted by descending probability.

    Raises:
        ValueError: If ``top_n`` is given and is less than 1.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    assert qc_no_meas is not None  # inplace=False always returns a new circuit

    if device == "CPU":
        sv = Statevector.from_instruction(qc_no_meas)
    else:
        sim = AerSimulator(method="statevector", device=device)
        circ = qc_no_meas.copy()
        circ.save_statevector()
        qc_t = transpile(circ, basis_gates=_simulator_basis_gates(sim))
        sv = Statevector(sim.run(qc_t).result().get_statevector())

    # Work on the raw probability array (index == basis-state integer, qubit 0 is the LSB) rather
    # than probabilities_dict(): the dict materialises a string label for every one of the 2**n
    # basis states (~4 GiB at 24 qubits), which is what limited exact simulation to tiny circuits.
    prob_array = sv.probabilities()
    num_qubits = qc_no_meas.num_qubits

    if top_n is not None:
        if top_n < 1:
            raise ValueError(f"top_n must be a positive integer, got {top_n}")
        k = min(top_n, prob_array.size)
        top_indices = np.argpartition(prob_array, -k)[-k:]
        top_indices = top_indices[np.argsort(prob_array[top_indices])[::-1]]
        ranking = [
            (format(int(i), f"0{num_qubits}b"), float(prob_array[i]))
            for i in top_indices
            if prob_array[i] > 0.0
        ]
        if verbose:
            _print_ranking(ranking)
        return ranking

    peak_index = int(np.argmax(prob_array))
    peak_bitstring = format(peak_index, f"0{num_qubits}b")
    peak_prob = float(prob_array[peak_index])

    if verbose:
        print("Most likely bitstring:", peak_bitstring)
        print("Peak probability:", peak_prob)
    return peak_bitstring, peak_prob


def mps_sample_counts(
    qc: QuantumCircuit,
    shots: int = 4096,
    bond_dim: int = 64,
    *,
    device: str = "CPU",
) -> dict[str, int]:
    """Sample measurement counts via the matrix-product-state (MPS) simulator.

    The input circuit is left untouched: any final measurements are removed and a full
    ``measure_all`` is added on a copy before sampling. Returning raw counts lets callers
    compute both the most-frequent bitstring and per-qubit marginals from one run.

    Args:
        qc (QuantumCircuit): Circuit to sample.
        shots (int): Number of measurement shots to draw.
        bond_dim (int): Maximum MPS bond dimension.
        device (str): Aer device, ``"CPU"`` (default) or ``"GPU"`` (LUMI ``standard-g``).

    Returns:
        dict[str, int]: Mapping of measured bitstring to shot count (Qiskit ordering).
    """
    qc_copy = qc.remove_final_measurements(inplace=False)
    assert qc_copy is not None  # inplace=False always returns a new circuit
    qc_copy.measure_all()

    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
        device=device,
    )

    qc_t = transpile(qc_copy, basis_gates=_simulator_basis_gates(sim))
    try:
        return sim.run(qc_t, shots=shots).result().get_counts()
    except RuntimeError as exc:
        if device == "CPU":
            raise
        import warnings
        warnings.warn(
            f"GPU MPS failed ({exc}); the installed qiskit-aer wheel is CPU-only. "
            "Retrying on CPU. Build qiskit-aer from source with CUDA to use the GPU.",
            stacklevel=2,
        )
        sim_cpu = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=bond_dim,
        )
        qc_t_cpu = transpile(qc_copy, basis_gates=_simulator_basis_gates(sim_cpu))
        return sim_cpu.run(qc_t_cpu, shots=shots).result().get_counts()


def statevector_probability(qc: QuantumCircuit, bitstring: str) -> float:
    """Return the exact probability of measuring ``bitstring`` via statevector simulation.

    Used as the amplitude oracle for local-max checks and greedy refinement on circuits small
    enough for an exact statevector. The input circuit is left untouched.

    Args:
        qc (QuantumCircuit): Circuit to simulate.
        bitstring (str): Target bitstring in Qiskit ordering (rightmost char is qubit 0). Its
            length must equal the circuit's qubit count.

    Returns:
        float: The probability of ``bitstring`` in ``[0, 1]``.

    Raises:
        ValueError: If ``bitstring`` is not the circuit's width or contains non-binary chars.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    assert qc_no_meas is not None  # inplace=False always returns a new circuit
    n = qc_no_meas.num_qubits
    if len(bitstring) != n or set(bitstring) - {"0", "1"}:
        raise ValueError(f"bitstring must be {n} binary chars, got {bitstring!r}")
    sv = Statevector.from_instruction(qc_no_meas)
    amp = sv.data[int(bitstring, 2)]
    return float((amp.conjugate() * amp).real)


@overload
def matrix_product_operators(
    qc: QuantumCircuit,
    shots: int = ...,
    bond_dim: int = ...,
    verbose: bool = ...,
    *,
    top_n: None = ...,
    device: str = ...,
) -> tuple[str, float]: ...
@overload
def matrix_product_operators(
    qc: QuantumCircuit,
    shots: int = ...,
    bond_dim: int = ...,
    verbose: bool = ...,
    *,
    top_n: int,
    device: str = ...,
) -> list[tuple[str, float]]: ...
def matrix_product_operators(
    qc: QuantumCircuit,
    shots: int = 4096,
    bond_dim: int = 64,
    verbose: bool = False,
    *,
    top_n: int | None = None,
    device: str = "CPU",
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
        device (str): Aer device, ``"CPU"`` (default) or ``"GPU"`` (LUMI ``standard-g``).

    Returns:
        tuple[str, float] | list[tuple[str, float]]: When ``top_n`` is ``None``, the
        estimated peak bitstring and its estimated probability. When ``top_n`` is an
        int, a list of up to ``top_n`` ``(bitstring, probability)`` pairs sorted by
        descending estimated probability.

    Raises:
        ValueError: If ``top_n`` is given and is less than 1.
    """
    counts = mps_sample_counts(qc, shots=shots, bond_dim=bond_dim, device=device)
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
