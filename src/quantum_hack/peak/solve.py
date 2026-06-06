"""The peak-finding orchestrator: pick methods, cross-validate, and score confidence.

:func:`solve_peak` profiles a circuit, runs an escalation chain of methods, and returns a
:class:`~quantum_hack.peak.result.PeakResult` carrying the evidence needed to judge it. The peak is
always found with the matrix-product-state (MPS) simulator and the marginal vote derived from the
same sampling — never an exact statevector. The guiding rules, since the portal hides the answer:

* **Consensus is the strongest heuristic** — when the most-frequent MPS sample and the per-qubit
  marginal vote agree, trust them.
* **A weak sampled peak defers to the marginal vote** — the marginal vote recovers peaks that are
  under-sampled (exactly the low-probability ``failed`` rows in the current result CSVs).
* **A tensor-network amplitude oracle refines/verifies** when quimb is available (large circuits).
* **Never re-propose a known failure** — if a candidate was already submitted and rejected, fall
  back to the next distinct candidate.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from qiskit import QuantumCircuit

from quantum_hack.peak.marginals import (
    Oracle,
    greedy_refine,
    least_confident_qubits,
    marginal_bitstring,
    z_marginals_from_counts,
)
from quantum_hack.peak.result import PeakResult, score_confidence
from quantum_hack.peak.selector import Method, profile_circuit, select_method
from quantum_hack.peak.tensor_network import probability as tn_probability
from quantum_hack.peak.tensor_network import quimb_available
from quantum_hack.results import DEFAULT_RESULTS_DIR, SubmissionRecord, is_known_failure
from quantum_hack.simulation import mps_sample_counts

# A sampled MPS peak at or above this probability is treated as concentrated enough to trust
# over a disagreeing marginal vote; below it the marginal vote leads.
CONCENTRATED_THRESHOLD = 0.15
# When refining a wide circuit with an expensive oracle, only flip this many least-confident bits.
REFINE_QUBIT_BUDGET = 14
# How many top candidates to register from the MPS sampling, so the known-failure fallback has
# alternatives to switch to when the portal rejects the top guess.
MPS_TOP_N = 5


def _marginal_independence_prob(marginals: Sequence[float], bitstring: str) -> float:
    """Estimate a bitstring's probability as the product of its per-qubit marginals.

    Exact only if qubits were independent, but a useful estimate for a candidate that sampling
    under-counted (it stays meaningful where the sampled frequency is ~0).

    Args:
        marginals (Sequence[float]): ``<Z_i>`` per qubit (index 0 is qubit 0).
        bitstring (str): Candidate bitstring in Qiskit ordering.

    Returns:
        float: Product over qubits of ``P(bit_i)`` derived from ``<Z_i>``, in ``[0, 1]``.
    """
    n = len(marginals)
    p = 1.0
    for i in range(n):
        bit = bitstring[n - 1 - i]
        p1 = (1.0 - marginals[i]) / 2.0
        p *= p1 if bit == "1" else (1.0 - p1)
    return p


def _local_max_check(
    qc: QuantumCircuit, bitstring: str, oracle: Oracle
) -> tuple[bool, float, float]:
    """Check whether ``bitstring`` beats all single-bit-flip neighbours under ``oracle``.

    Args:
        qc (QuantumCircuit): Circuit being solved.
        bitstring (str): Candidate peak bitstring (Qiskit ordering).
        oracle (Oracle): Probability oracle ``(qc, bitstring) -> float``.

    Returns:
        tuple[bool, float, float]: ``(is_local_max, neighbor_gap, probability)`` where
        ``neighbor_gap`` is the candidate's probability minus the best neighbour's (positive
        confirms a local maximum).
    """
    n = qc.num_qubits
    p = oracle(qc, bitstring)
    best_neighbor = 0.0
    for i in range(n):
        pos = n - 1 - i
        flipped = "1" if bitstring[pos] == "0" else "0"
        neighbor = f"{bitstring[:pos]}{flipped}{bitstring[pos + 1:]}"
        best_neighbor = max(best_neighbor, oracle(qc, neighbor))
    return p >= best_neighbor, p - best_neighbor, p


def solve_peak(
    qc: QuantumCircuit,
    *,
    gpu_available: bool = False,
    device: str = "CPU",
    shots: int = 8192,
    bond_dim: int = 128,
    challenge: str | None = None,
    records: Iterable[SubmissionRecord] | None = None,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    verify_local_max: bool = True,
) -> PeakResult:
    """Find the peak bitstring of ``qc`` and return it with confidence evidence.

    Args:
        qc (QuantumCircuit): Circuit to solve (final measurements, if any, are ignored).
        gpu_available (bool): Whether GPU Aer is available (LUMI); enables the tensor-network
            oracle for large circuits. Aer MPS sampling uses ``device``.
        device (str): Aer device for statevector/MPS runs (``"CPU"`` or ``"GPU"``).
        shots (int): MPS sampling shots.
        bond_dim (int): MPS bond dimension.
        challenge (str | None): Challenge name used to skip already-rejected bitstrings via
            :func:`~quantum_hack.results.is_known_failure`. ``None`` disables the check.
        records (Iterable[SubmissionRecord] | None): Pre-loaded result records for the known-
            failure check; ``None`` reads from ``results_dir`` on demand.
        results_dir (Path | str): Directory of result CSVs for the known-failure check.
        verify_local_max (bool): If ``True`` and an amplitude oracle is available, verify the
            winner beats every single-bit-flip neighbour.

    Returns:
        PeakResult: The chosen bitstring, its (estimated) probability, the method, and the
        confidence/local-max/consensus evidence.
    """
    profile = profile_circuit(qc)
    n = profile.num_qubits
    chain = select_method(profile, gpu_available=gpu_available)

    have_quimb = quimb_available()
    # The peak is found via MPS sampling and the marginal vote; the only amplitude oracle (for
    # refinement / local-max verification) is the tensor network, when quimb is available.
    oracle: Oracle | None = tn_probability if have_quimb else None

    methods_for: dict[str, set[str]] = defaultdict(set)
    prob_of: dict[str, float] = {}
    notes: list[str] = []
    mps_counts: dict[str, int] | None = None
    marginals: list[float] | None = None

    def register(bitstring: str, method: str, probability: float) -> None:
        methods_for[bitstring].add(method)
        # Keep the most informative (largest) probability estimate seen for a candidate.
        prob_of[bitstring] = max(prob_of.get(bitstring, 0.0), probability)

    for method in chain:
        if method == Method.MPS:
            if mps_counts is None:
                mps_counts = mps_sample_counts(qc, shots=shots, bond_dim=bond_dim, device=device)
            ranked = sorted(mps_counts, key=lambda b: mps_counts[b], reverse=True)
            for bitstring in ranked[:MPS_TOP_N]:
                register(bitstring, Method.MPS, mps_counts[bitstring] / shots)
            notes.append(f"MPS top p={mps_counts[ranked[0]] / shots:.3f} (bond={bond_dim})")
        elif method == Method.MARGINAL:
            if mps_counts is None:
                mps_counts = mps_sample_counts(qc, shots=shots, bond_dim=bond_dim, device=device)
            marginals = z_marginals_from_counts(mps_counts, n)
            cand = marginal_bitstring(marginals)
            register(cand, Method.MARGINAL, _marginal_independence_prob(marginals, cand))
            notes.append("marginal vote")
        elif method == Method.GREEDY_REFINE and oracle is not None and prob_of:
            seed = max(prob_of, key=lambda b: prob_of[b])
            budget = (
                least_confident_qubits(marginals, REFINE_QUBIT_BUDGET)
                if marginals is not None
                else None
            )
            refined, p = greedy_refine(qc, seed, oracle=oracle, candidate_qubits=budget)
            register(refined, Method.GREEDY_REFINE, p)
            notes.append("greedy refine")
        elif method == Method.TENSOR_NETWORK and have_quimb:
            # Re-score existing candidates with exact TN amplitudes for a decisive comparison.
            for cand in list(prob_of):
                register(cand, Method.TENSOR_NETWORK, tn_probability(qc, cand))
            notes.append("tensor-network amplitudes")

    winner, method_label = _choose_winner(
        have_oracle=oracle is not None,
        mps_counts=mps_counts,
        shots=shots,
        marginals=marginals,
        prob_of=prob_of,
        methods_for=methods_for,
        notes=notes,
    )

    winner = _avoid_known_failure(
        winner, prob_of, challenge, records, results_dir, notes
    )

    probability = prob_of.get(winner, 0.0)
    is_local_max: bool | None = None
    neighbor_gap: float | None = None
    if verify_local_max and oracle is not None:
        is_local_max, neighbor_gap, probability = _local_max_check(qc, winner, oracle)
        prob_of[winner] = max(prob_of.get(winner, 0.0), probability)

    agreeing = tuple(sorted(methods_for.get(winner, {method_label})))
    confidence = score_confidence(
        probability=probability,
        num_qubits=n,
        is_local_max=is_local_max,
        agreeing_methods=len(agreeing),
        neighbor_gap=neighbor_gap,
    )
    return PeakResult(
        bitstring=winner,
        probability=probability,
        method=method_label,
        num_qubits=n,
        confidence=confidence,
        is_local_max=is_local_max,
        neighbor_gap=neighbor_gap,
        agreeing_methods=agreeing,
        notes="; ".join(notes),
    )


def _choose_winner(
    *,
    have_oracle: bool,
    mps_counts: dict[str, int] | None,
    shots: int,
    marginals: list[float] | None,
    prob_of: dict[str, float],
    methods_for: dict[str, set[str]],
    notes: list[str],
) -> tuple[str, str]:
    """Pick the winning bitstring and a method label from the gathered candidates.

    Args:
        have_oracle (bool): Whether a tensor-network amplitude oracle was available to score
            candidates with real amplitudes.
        mps_counts (dict[str, int] | None): MPS sample counts, if sampled.
        shots (int): MPS shot count (to convert counts to frequencies).
        marginals (list[float] | None): Per-qubit ``<Z_i>``, if computed.
        prob_of (dict[str, float]): Best probability estimate per candidate.
        methods_for (dict[str, set[str]]): Methods that produced each candidate.
        notes (list[str]): Mutable note list appended to with the decision rationale.

    Returns:
        tuple[str, str]: ``(winner_bitstring, method_label)``.
    """
    if not prob_of:
        raise RuntimeError("no candidate bitstrings were produced")

    # Tensor-network-scored regime: the highest exact amplitude is decisive.
    if have_oracle:
        winner = max(prob_of, key=lambda b: prob_of[b])
        return winner, "+".join(sorted(methods_for[winner]))

    # MPS-only regime: consensus, then concentration, then defer to the marginal vote.
    mps_top = max(mps_counts, key=lambda b: mps_counts[b]) if mps_counts else None
    mps_p = (mps_counts[mps_top] / shots) if (mps_counts and mps_top) else 0.0
    marg = marginal_bitstring(marginals) if marginals is not None else None

    if mps_top is not None and mps_top == marg:
        notes.append("MPS and marginal agree")
        return mps_top, "MPS+marginal"
    if mps_top is not None and mps_p >= CONCENTRATED_THRESHOLD:
        return mps_top, "MPS"
    if marg is not None:
        notes.append(f"MPS peak weak (p={mps_p:.3f}); using marginal vote")
        return marg, "marginal"
    return mps_top, "MPS"  # pragma: no cover - marg is always set when MPS path runs


def _avoid_known_failure(
    winner: str,
    prob_of: dict[str, float],
    challenge: str | None,
    records: Iterable[SubmissionRecord] | None,
    results_dir: Path | str,
    notes: list[str],
) -> str:
    """Swap the winner for the next-best candidate if it was already submitted and rejected.

    Args:
        winner (str): The currently chosen bitstring.
        prob_of (dict[str, float]): Best probability estimate per candidate.
        challenge (str | None): Challenge name for the known-failure lookup; ``None`` disables it.
        records (Iterable[SubmissionRecord] | None): Pre-loaded records, or ``None`` to read disk.
        results_dir (Path | str): Directory of result CSVs.
        notes (list[str]): Mutable note list appended to when a swap happens.

    Returns:
        str: The winner to use (unchanged if it is not a known failure or no alternative exists).
    """
    if challenge is None:
        return winner
    pool = list(records) if records is not None else None
    ranked = sorted(prob_of, key=lambda b: prob_of[b], reverse=True)
    for candidate in [winner, *(b for b in ranked if b != winner)]:
        if not is_known_failure(challenge, candidate, results_dir=results_dir, records=pool):
            if candidate != winner:
                notes.append(f"{winner} already failed; switched to {candidate}")
            return candidate
    notes.append("all candidates already failed")
    return winner
