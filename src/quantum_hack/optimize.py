"""Light, peak-preserving circuit reduction.

Simplification is *not* scored (only the bitstring is), so this stays deliberately small: it
shrinks a circuit just enough to help simulation and to tell a clean before/after story in the
presentation. It composes the available circuit transforms — :func:`~quantum_hack.strip.strip`,
:func:`~quantum_hack.transform.snap_rotations_to_pi_over_8`,
:func:`~quantum_hack.transform.transpile_to_basis`, and the optional ZX-calculus reduction
:func:`~quantum_hack.transform.optimize_with_pyzx` — greedily, keeping a step only if it does not
grow a weighted cost and does not change the peak bitstring.

The ZX (pyzx) step is by-construction outcome-preserving (``full_reduce`` keeps the unitary up to
global phase, like ``transpile``) but is tuned for Clifford+T circuits: on the arbitrary-angle
peaked circuits here it usually *grows* the gate count and is therefore rolled back by the cost
guard. It is wired in regardless so it helps when it can; it is skipped silently when pyzx is not
installed (the optional ``[zx]`` extra) and on circuits wider than :data:`PYZX_MAX_QUBITS`, where
ZX extraction is too slow to run inline.

The relevant invariant for peaked circuits is the **measurement-outcome distribution** (hence the
peak bitstring), not full unitary equivalence. ``strip`` and ``transpile`` preserve outcomes *by
construction* (``strip`` only drops leading gates on ``|0>`` and trailing diagonal RZ rotations,
which are invisible to a computational-basis measurement; ``transpile`` preserves the unitary up to
global phase). Angle snapping changes the distribution, so it is applied only when the exact peak
can be checked (small circuits) and is confirmed unchanged; on large circuits it is skipped. Note
this is a weaker guarantee than MQT QCEC's unitary equivalence — and the correct one here, since
QCEC would reject the (outcome-preserving) trailing-RZ strip.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from qiskit import QuantumCircuit

from quantum_hack.metrics import circuit_stats, compare_circuits
from quantum_hack.simulation import statevector_simulation
from quantum_hack.strip import strip
from quantum_hack.transform import (
    optimize_with_pyzx,
    snap_rotations_to_pi_over_8,
    transpile_to_basis,
)

# Outcome guarantee attached to a reduced circuit.
BY_CONSTRUCTION = "by_construction"  # only strip/transpile/pyzx: computational outcomes preserved
PEAK_VERIFIED = "peak_verified"  # a lossy step ran; exact peak bitstring confirmed unchanged
UNVERIFIED = "unverified"  # peak could not be confirmed unchanged (do not trust)

# Peak preservation is checked with an exact statevector, so only up to this width.
PEAK_CHECK_MAX_QUBITS = 30

# The pyzx step is skipped above this width: ZX extraction can be slow on wide circuits and is not
# worth running inline (this is a runtime bound, not a correctness one — pyzx is outcome-preserving
# at any width).
PYZX_MAX_QUBITS = 50


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


def _pyzx_reduce(qc: QuantumCircuit) -> QuantumCircuit:
    """Run :func:`~quantum_hack.transform.optimize_with_pyzx`, but only on narrow circuits.

    Args:
        qc (QuantumCircuit): Circuit to reduce.

    Returns:
        QuantumCircuit: The pyzx-reduced circuit.

    Raises:
        ValueError: If ``qc`` is wider than :data:`PYZX_MAX_QUBITS`. The pipeline catches this and
            skips the step, since ZX extraction is too slow to run inline on wide circuits.
        ImportError: If pyzx is not installed (also caught and skipped by the pipeline).
    """
    if qc.num_qubits > PYZX_MAX_QUBITS:
        raise ValueError(f"pyzx step skipped: {qc.num_qubits} qubits exceeds {PYZX_MAX_QUBITS}")
    return optimize_with_pyzx(qc)


def default_pipeline() -> list[tuple[str, Callable[[QuantumCircuit], QuantumCircuit], bool]]:
    """Return the ordered reduction pipeline.

    Each entry is ``(name, transform, is_lossy)`` where ``is_lossy`` marks a step that can change
    the measurement distribution (only angle snapping) and therefore requires a peak check. The
    final ``pyzx`` step is outcome-preserving by construction but optional: it is skipped (via the
    pipeline's catch-and-continue) when pyzx is not installed or the circuit is too wide.

    Returns:
        list[tuple[str, Callable[[QuantumCircuit], QuantumCircuit], bool]]: The pipeline steps.
    """
    return [
        ("strip", strip, False),
        ("snap", snap_rotations_to_pi_over_8, True),
        ("transpile_opt3", lambda qc: transpile_to_basis(qc, optimization_level=3), False),
        ("pyzx", _pyzx_reduce, False),
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


def _reduction_pct(before: int, after: int) -> float:
    """Percentage by which a metric shrank (positive means it got smaller).

    When ``before`` is ``0`` this returns ``0.0`` regardless of ``after`` (a percentage of zero is
    undefined), so growth from zero (``0 -> N``) reads as ``0%`` and is indistinguishable from no
    change here; the raw ``before -> after`` shown alongside it still reveals the growth.

    Args:
        before (int): Metric value before optimization.
        after (int): Metric value after optimization.

    Returns:
        float: ``100 * (before - after) / before``, or ``0.0`` when ``before`` is ``0``.
    """
    return 0.0 if before == 0 else 100.0 * (before - after) / before


@dataclass(frozen=True)
class OptimizationStats:
    """Before/after size metrics for one circuit run through :func:`optimize_circuit`.

    The headline metric is CX depth (the two-qubit critical-path length), not total circuit depth:
    two-qubit gates dominate runtime and error on hardware, so CX depth is the cost that matters.

    Attributes:
        name (str): Identifier for the circuit (e.g. ``"very_easy/challenge-8_1"``).
        num_qubits (int): Circuit width.
        cx_depth_before (int): CX depth before optimization.
        cx_depth_after (int): CX depth after optimization.
        size_before (int): Gate count before optimization.
        size_after (int): Gate count after optimization.
        cx_before (int): CX count before optimization.
        cx_after (int): CX count after optimization.
        steps (tuple[str, ...]): Names of the pipeline steps that were accepted, in order.
        equivalence (str): The outcome guarantee (:data:`BY_CONSTRUCTION`, :data:`PEAK_VERIFIED`,
            or :data:`UNVERIFIED`).
    """

    name: str
    num_qubits: int
    cx_depth_before: int
    cx_depth_after: int
    size_before: int
    size_after: int
    cx_before: int
    cx_after: int
    steps: tuple[str, ...]
    equivalence: str

    @property
    def cx_depth_reduction_pct(self) -> float:
        """Percentage CX-depth reduction.

        Returns:
            float: Percentage by which the CX depth shrank (positive means smaller).
        """
        return _reduction_pct(self.cx_depth_before, self.cx_depth_after)

    @property
    def size_reduction_pct(self) -> float:
        """Percentage gate-count reduction.

        Returns:
            float: Percentage by which the gate count shrank (positive means smaller).
        """
        return _reduction_pct(self.size_before, self.size_after)

    @property
    def cx_reduction_pct(self) -> float:
        """Percentage CX-count reduction.

        Returns:
            float: Percentage by which the CX count shrank (positive means smaller).
        """
        return _reduction_pct(self.cx_before, self.cx_after)


def optimization_stats(name: str, result: OptimizeResult) -> OptimizationStats:
    """Pull the before/after size metrics out of an :class:`OptimizeResult`.

    Args:
        name (str): Identifier for the circuit.
        result (OptimizeResult): The result returned by :func:`optimize_circuit`.

    Returns:
        OptimizationStats: The before/after CX-depth/size/CX metrics, accepted steps, and guarantee.
    """
    comparison = result.comparison
    cx = comparison["ops"].get("cx", {"before": 0, "after": 0})
    return OptimizationStats(
        name=name,
        num_qubits=result.original.num_qubits,
        cx_depth_before=comparison["cx_depth"]["before"],
        cx_depth_after=comparison["cx_depth"]["after"],
        size_before=comparison["size"]["before"],
        size_after=comparison["size"]["after"],
        cx_before=cx["before"],
        cx_after=cx["after"],
        steps=tuple(result.steps),
        equivalence=result.equivalence,
    )


def optimize_circuits(
    named_circuits: Iterable[tuple[str, QuantumCircuit]],
    *,
    cx_weight: float = 3.0,
    peak_check_max_qubits: int = PEAK_CHECK_MAX_QUBITS,
    verify: bool = True,
) -> list[OptimizationStats]:
    """Run :func:`optimize_circuit` over many named circuits and collect their reduction stats.

    Pairs cleanly with :func:`quantum_hack.challenges.iter_challenges` called with ``func=None``,
    which yields the ``(name, circuit)`` tuples this consumes.

    Args:
        named_circuits (Iterable[tuple[str, QuantumCircuit]]): ``(name, circuit)`` pairs to
            optimize.
        cx_weight (float): CX weight forwarded to :func:`optimize_circuit`.
        peak_check_max_qubits (int): Maximum width for the exact peak check, forwarded to
            :func:`optimize_circuit`. Lowering it skips the lossy step (and its costly statevector)
            on wide circuits.
        verify (bool): Whether to run the peak check, forwarded to :func:`optimize_circuit`.

    Returns:
        list[OptimizationStats]: One :class:`OptimizationStats` per input circuit, in input order.
    """
    return [
        optimization_stats(
            name,
            optimize_circuit(
                qc,
                cx_weight=cx_weight,
                peak_check_max_qubits=peak_check_max_qubits,
                verify=verify,
            ),
        )
        for name, qc in named_circuits
    ]


@dataclass(frozen=True)
class OptimizationSummary:
    """Aggregate before/after reduction across a set of :class:`OptimizationStats`.

    Attributes:
        count (int): Number of circuits summarized.
        cx_depth_before (int): Total CX depth before optimization.
        cx_depth_after (int): Total CX depth after optimization.
        size_before (int): Total gate count before optimization.
        size_after (int): Total gate count after optimization.
        cx_before (int): Total CX count before optimization.
        cx_after (int): Total CX count after optimization.
        improved (int): Number of circuits whose gate count strictly shrank.
        equivalence_counts (tuple[tuple[str, int], ...]): ``(guarantee, count)`` pairs sorted by
            guarantee. A tuple (not a dict) so the frozen summary stays immutable and hashable,
            matching :attr:`OptimizationStats.steps`; call ``dict(...)`` on it for lookups.
    """

    count: int
    cx_depth_before: int
    cx_depth_after: int
    size_before: int
    size_after: int
    cx_before: int
    cx_after: int
    improved: int
    equivalence_counts: tuple[tuple[str, int], ...]

    @classmethod
    def from_stats(cls, stats: Iterable[OptimizationStats]) -> "OptimizationSummary":
        """Aggregate a collection of per-circuit stats into totals.

        Args:
            stats (Iterable[OptimizationStats]): Per-circuit reduction stats to aggregate.

        Returns:
            OptimizationSummary: The summed metrics, improved count, and guarantee breakdown.
        """
        stats = list(stats)
        counts: dict[str, int] = {}
        for s in stats:
            counts[s.equivalence] = counts.get(s.equivalence, 0) + 1
        return cls(
            count=len(stats),
            cx_depth_before=sum(s.cx_depth_before for s in stats),
            cx_depth_after=sum(s.cx_depth_after for s in stats),
            size_before=sum(s.size_before for s in stats),
            size_after=sum(s.size_after for s in stats),
            cx_before=sum(s.cx_before for s in stats),
            cx_after=sum(s.cx_after for s in stats),
            improved=sum(1 for s in stats if s.size_after < s.size_before),
            equivalence_counts=tuple(sorted(counts.items())),
        )

    @property
    def cx_depth_reduction_pct(self) -> float:
        """Percentage reduction in total CX depth.

        Returns:
            float: Percentage by which summed CX depth shrank.
        """
        return _reduction_pct(self.cx_depth_before, self.cx_depth_after)

    @property
    def size_reduction_pct(self) -> float:
        """Percentage reduction in total gate count.

        Returns:
            float: Percentage by which the summed gate count shrank.
        """
        return _reduction_pct(self.size_before, self.size_after)

    @property
    def cx_reduction_pct(self) -> float:
        """Percentage reduction in total CX count.

        Returns:
            float: Percentage by which the summed CX count shrank.
        """
        return _reduction_pct(self.cx_before, self.cx_after)


def format_optimization_table(stats: Iterable[OptimizationStats]) -> str:
    """Render per-circuit reduction stats and their aggregate as a fixed-width text table.

    Args:
        stats (Iterable[OptimizationStats]): Per-circuit reduction stats to render.

    Returns:
        str: A multi-line table with one row per circuit, a totals row, and a guarantee breakdown.
    """
    stats = list(stats)
    summary = OptimizationSummary.from_stats(stats)

    def _cell(before: int, after: int) -> str:
        """Format a ``before->after (+pct%)`` cell (``+`` percent means it shrank).

        Args:
            before (int): Metric value before optimization.
            after (int): Metric value after optimization.

        Returns:
            str: A ``before->after (+pct%)`` string, e.g. ``"105->100 (+5%)"``.
        """
        return f"{before}->{after} ({_reduction_pct(before, after):+.0f}%)"

    headers = ["circuit", "n", "cx_depth", "size", "cx", "guarantee", "steps"]
    aligns = ["<", ">", ">", ">", ">", "<", "<"]
    body = [
        [
            s.name,
            str(s.num_qubits),
            _cell(s.cx_depth_before, s.cx_depth_after),
            _cell(s.size_before, s.size_after),
            _cell(s.cx_before, s.cx_after),
            s.equivalence,
            "+".join(s.steps) or "none",
        ]
        for s in stats
    ]
    total = [
        f"TOTAL ({summary.count})",
        "",
        _cell(summary.cx_depth_before, summary.cx_depth_after),
        _cell(summary.size_before, summary.size_after),
        _cell(summary.cx_before, summary.cx_after),
        "",
        "",
    ]

    # Size every column to its widest cell (header included) so nothing ever overflows the grid.
    widths = [max(len(row[c]) for row in [headers, *body, total]) for c in range(len(headers))]

    def _fmt(row: list[str]) -> str:
        """Pad a row's cells to the computed column widths.

        Args:
            row (list[str]): One cell string per column.

        Returns:
            str: The row rendered with two-space gutters between aligned, fixed-width columns.
        """
        return "  ".join(f"{cell:{aligns[c]}{widths[c]}}" for c, cell in enumerate(row))

    rule = "-" * (sum(widths) + 2 * (len(widths) - 1))
    lines = [_fmt(headers), rule, *(_fmt(row) for row in body), rule, _fmt(total)]
    breakdown = ", ".join(f"{k}={v}" for k, v in summary.equivalence_counts)
    lines.append(f"improved {summary.improved}/{summary.count} circuits; guarantees: {breakdown}")
    return "\n".join(lines)
