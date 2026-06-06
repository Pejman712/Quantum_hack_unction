"""Light, peak-preserving circuit reduction.

Simplification is *not* scored (only the bitstring is), so this stays deliberately small: it
shrinks a circuit just enough to help simulation and to tell a clean before/after story in the
presentation. It composes the existing Qiskit-only transforms — :func:`~quantum_hack.strip.strip`,
:func:`~quantum_hack.transform.snap_rotations_to_pi_over_4`, and
:func:`~quantum_hack.transform.transpile_to_basis` — greedily, keeping a step only if it does not
grow a weighted cost and does not change the peak bitstring.

The relevant invariant for peaked circuits is the **measurement-outcome distribution** (hence the
peak bitstring), not full unitary equivalence. ``strip`` and ``transpile`` preserve outcomes *by
construction* (``strip`` only drops leading gates on ``|0>`` and trailing diagonal RZ rotations,
which are invisible to a computational-basis measurement; ``transpile`` preserves the unitary up to
global phase). Angle snapping changes the distribution, so it is applied only when the exact peak
can be checked (small circuits) and is confirmed unchanged; on large circuits it is skipped. Note
this is a weaker guarantee than MQT QCEC's unitary equivalence — and the correct one here, since
QCEC would reject the (outcome-preserving) trailing-RZ strip.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from qiskit import QuantumCircuit

from quantum_hack.metrics import circuit_stats, compare_circuits
from quantum_hack.simulation import statevector_simulation
from quantum_hack.strip import strip
from quantum_hack.transform import snap_rotations_to_pi_over_4, transpile_to_basis

# Outcome guarantee attached to a reduced circuit.
BY_CONSTRUCTION = "by_construction"  # only strip/transpile: computational-basis outcomes preserved
PEAK_VERIFIED = "peak_verified"  # a lossy step ran; exact peak bitstring confirmed unchanged
UNVERIFIED = "unverified"  # peak could not be confirmed unchanged (do not trust)

# Peak preservation is checked with an exact statevector, so only up to this width.
PEAK_CHECK_MAX_QUBITS = 30


@dataclass
class OptimizeResult:
    """The outcome of :func:`optimize_circuit`.

    Attributes:
        original (QuantumCircuit): The input circuit (unchanged).
        optimized (QuantumCircuit): The reduced circuit.
        comparison (dict): :func:`~quantum_hack.metrics.compare_circuits` of original vs optimized.
        equivalence (str): One of :data:`BY_CONSTRUCTION`, :data:`PEAK_VERIFIED`,
            :data:`UNVERIFIED`.
        steps (list[str]): Names of the pipeline steps that were accepted, in order.
    """

    original: QuantumCircuit
    optimized: QuantumCircuit
    comparison: dict
    equivalence: str
    steps: list[str] = field(default_factory=list)


def cost(qc: QuantumCircuit, *, cx_weight: float = 3.0) -> float:
    """Return a scalar reduction cost (lower is better): weighted CX count plus depth.

    Uses :func:`~quantum_hack.metrics.circuit_stats`, which normalises to the ``rz``/``rx``/``cx``
    basis first so counts are comparable across transforms.

    Args:
        qc (QuantumCircuit): Circuit to measure.
        cx_weight (float): Weight applied to the CX count relative to depth.

    Returns:
        float: ``cx_weight * num_cx + depth``.
    """
    stats = circuit_stats(qc)
    return cx_weight * stats["ops"].get("cx", 0) + stats["depth"]


def default_pipeline() -> list[tuple[str, Callable[[QuantumCircuit], QuantumCircuit], bool]]:
    """Return the ordered reduction pipeline.

    Each entry is ``(name, transform, is_lossy)`` where ``is_lossy`` marks a step that can change
    the measurement distribution (only angle snapping) and therefore requires a peak check.

    Returns:
        list[tuple[str, Callable[[QuantumCircuit], QuantumCircuit], bool]]: The pipeline steps.
    """
    return [
        ("strip", strip, False),
        ("snap", snap_rotations_to_pi_over_4, True),
        ("transpile_opt3", lambda qc: transpile_to_basis(qc, optimization_level=3), False),
    ]


def optimize_circuit(
    qc: QuantumCircuit,
    *,
    cx_weight: float = 3.0,
    peak_check_max_qubits: int = PEAK_CHECK_MAX_QUBITS,
    verify: bool = True,
) -> OptimizeResult:
    """Greedily reduce ``qc`` while preserving its peak bitstring, rolling back unhelpful steps.

    Each step is applied to the best circuit so far and kept only if it does not increase
    :func:`cost`; the one lossy step (angle snapping) is additionally kept only when the exact peak
    can be checked and is unchanged. The input circuit is never mutated.

    Args:
        qc (QuantumCircuit): Circuit to reduce.
        cx_weight (float): CX weight passed to :func:`cost`.
        peak_check_max_qubits (int): Maximum width at which the exact peak is checked; wider
            circuits skip the lossy step and rely only on outcome-preserving transforms.
        verify (bool): If ``False``, skip the peak check (and thus the lossy step), trusting only
            the by-construction-safe transforms.

    Returns:
        OptimizeResult: The reduced circuit, the before/after comparison, the outcome guarantee,
        and the accepted step names.
    """
    can_verify = verify and qc.num_qubits <= peak_check_max_qubits
    original_peak = statevector_simulation(qc)[0] if can_verify else None

    best = qc
    best_cost = cost(best, cx_weight=cx_weight)
    accepted: list[str] = []
    used_lossy = False

    for name, transform, is_lossy in default_pipeline():
        try:
            candidate = transform(best)
        except Exception:  # noqa: BLE001 - a transform failing must not abort the pipeline
            continue
        if cost(candidate, cx_weight=cx_weight) > best_cost:
            continue
        if is_lossy:
            if not can_verify or statevector_simulation(candidate)[0] != original_peak:
                continue  # cannot confirm the peak survives this step -> skip it
            used_lossy = True
        best = candidate
        best_cost = cost(best, cx_weight=cx_weight)
        accepted.append(name)

    equivalence = _outcome_label(
        best, original_peak=original_peak, can_verify=can_verify, used_lossy=used_lossy
    )
    return OptimizeResult(
        original=qc,
        optimized=best,
        comparison=compare_circuits(qc, best),
        equivalence=equivalence,
        steps=accepted,
    )


def _outcome_label(
    optimized: QuantumCircuit,
    *,
    original_peak: str | None,
    can_verify: bool,
    used_lossy: bool,
) -> str:
    """Decide the outcome guarantee for a reduced circuit.

    Args:
        optimized (QuantumCircuit): The reduced circuit.
        original_peak (str | None): The original circuit's exact peak bitstring, or ``None`` if it
            was not computed (large circuit / verification disabled).
        can_verify (bool): Whether the exact peak could be checked for this width.
        used_lossy (bool): Whether a lossy step (angle snapping) was applied.

    Returns:
        str: :data:`PEAK_VERIFIED` if a lossy step ran and the final peak still matches,
        :data:`BY_CONSTRUCTION` if only outcome-preserving transforms ran, else :data:`UNVERIFIED`.
    """
    if not used_lossy:
        return BY_CONSTRUCTION
    if can_verify and statevector_simulation(optimized)[0] == original_peak:
        return PEAK_VERIFIED
    return UNVERIFIED
