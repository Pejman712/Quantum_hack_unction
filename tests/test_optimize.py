import math

import pytest
from qiskit import QuantumCircuit

from quantum_hack.optimize import (
    BY_CONSTRUCTION,
    PEAK_VERIFIED,
    PYZX_MAX_QUBITS,
    UNVERIFIED,
    OptimizationStats,
    OptimizationSummary,
    cost,
    default_pipeline,
    format_optimization_table,
    optimization_stats,
    optimize_circuit,
    optimize_circuits,
)
from quantum_hack.simulation import statevector_simulation


def _reducible_circuit() -> QuantumCircuit:
    """Circuit with a strippable leading RZ on |0> and a trailing RZ.

    Returns:
        QuantumCircuit: A 2-qubit circuit that strip/transpile can shrink while preserving the peak.
    """
    qc = QuantumCircuit(2)
    qc.rz(0.5, 0)  # leading RZ on |0> -> only global phase, strippable
    qc.rx(0.7, 0)
    qc.cx(0, 1)
    qc.rz(0.5, 1)  # trailing RZ (diagonal, invisible to measurement) -> strippable
    return qc


def _peak(qc: QuantumCircuit) -> str:
    """Return the exact peak bitstring of a circuit.

    Args:
        qc (QuantumCircuit): Circuit to simulate.

    Returns:
        str: The most likely measurement bitstring.
    """
    return statevector_simulation(qc)[0]


def test_optimize_reduces_cost_and_preserves_peak():
    qc = _reducible_circuit()
    result = optimize_circuit(qc)
    assert cost(result.optimized) <= cost(qc)
    assert result.comparison["size"]["delta"] <= 0
    assert _peak(result.optimized) == _peak(qc)
    assert result.equivalence in (PEAK_VERIFIED, BY_CONSTRUCTION)


def test_optimize_does_not_mutate_input():
    qc = _reducible_circuit()
    before = len(qc.data)
    optimize_circuit(qc)
    assert len(qc.data) == before


def test_optimize_preserves_peak_through_qubit_permuting_circuit():
    # A SWAP that transpile_opt3 elides into a layout permutation. transpile_opt3 is is_lossy=False
    # (no per-step peak check), so if the permutation were not undone the pipeline would silently
    # return a relabelled (wrong) peak as BY_CONSTRUCTION. Guards the pipeline-level invariant.
    qc = QuantumCircuit(3)
    qc.rx(0.9, 0)
    qc.rx(0.4, 1)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.swap(0, 2)
    result = optimize_circuit(qc)
    assert _peak(result.optimized) == _peak(qc)
    assert result.equivalence in (PEAK_VERIFIED, BY_CONSTRUCTION)  # never silently UNVERIFIED


def test_snap_accepted_when_angles_already_on_grid_gives_peak_verified():
    qc = QuantumCircuit(2)
    qc.rx(math.pi / 4, 0)  # pi/4 = 2*(pi/8), on the grid -> snapping leaves the peak unchanged
    qc.cx(0, 1)
    result = optimize_circuit(qc)
    assert "snap" in result.steps
    assert result.equivalence == PEAK_VERIFIED
    assert _peak(result.optimized) == _peak(qc)


def test_lossy_step_skipped_when_verification_disabled():
    qc = _reducible_circuit()
    result = optimize_circuit(qc, peak_check_max_qubits=0)
    assert "snap" not in result.steps
    assert result.equivalence == BY_CONSTRUCTION
    assert _peak(result.optimized) == _peak(qc)


def test_pipeline_includes_pyzx_step():
    names = [name for name, _, is_lossy in default_pipeline() if not is_lossy]
    assert "pyzx" in names


def test_pyzx_skipped_on_wide_circuit():
    # Wider than PYZX_MAX_QUBITS, so _pyzx_reduce raises and the pipeline skips it. The circuit is
    # trivial and above the peak-check width, so no statevector is built.
    qc = QuantumCircuit(PYZX_MAX_QUBITS + 1)
    qc.x(0)
    result = optimize_circuit(qc)
    assert "pyzx" not in result.steps


def test_optimization_stats_pulls_before_after_metrics():
    qc = _reducible_circuit()
    result = optimize_circuit(qc)
    stats = optimization_stats("demo", result)

    assert stats.name == "demo"
    assert stats.num_qubits == qc.num_qubits
    # The comparison the result carries is the source of truth for the stats.
    assert stats.cx_depth_before == result.comparison["cx_depth"]["before"]
    assert stats.cx_depth_after == result.comparison["cx_depth"]["after"]
    assert stats.size_before == result.comparison["size"]["before"]
    assert stats.size_after == result.comparison["size"]["after"]
    assert stats.equivalence == result.equivalence
    assert stats.steps == tuple(result.steps)
    # _reducible_circuit shrinks, so reductions are non-negative.
    assert stats.size_reduction_pct >= 0
    assert stats.cx_reduction_pct >= 0


def test_optimization_stats_handles_circuit_without_cx():
    qc = QuantumCircuit(1)
    qc.rx(0.7, 0)
    stats = optimization_stats("no-cx", optimize_circuit(qc))

    assert stats.cx_before == 0
    assert stats.cx_after == 0
    assert stats.cx_reduction_pct == 0.0  # guards against division by zero


def test_optimize_circuits_returns_stats_per_circuit():
    named = [("a", _reducible_circuit()), ("b", _reducible_circuit())]
    stats = optimize_circuits(named)

    assert [s.name for s in stats] == ["a", "b"]
    assert all(isinstance(s, OptimizationStats) for s in stats)


def test_optimization_summary_aggregates_totals_and_guarantees():
    # Positional fields: name, n, cx_depth_before, cx_depth_after, size_before, size_after,
    # cx_before, cx_after, steps, equivalence.
    stats = [
        OptimizationStats("a", 4, 10, 8, 20, 15, 6, 4, ("strip",), BY_CONSTRUCTION),
        OptimizationStats("b", 4, 10, 10, 20, 20, 6, 6, (), BY_CONSTRUCTION),
    ]
    summary = OptimizationSummary.from_stats(stats)

    assert summary.count == 2
    assert summary.cx_depth_before == 20 and summary.cx_depth_after == 18
    assert summary.size_before == 40 and summary.size_after == 35
    assert summary.cx_before == 12 and summary.cx_after == 10
    assert summary.improved == 1  # only "a" shrank
    assert summary.equivalence_counts == ((BY_CONSTRUCTION, 2),)
    assert summary.size_reduction_pct == pytest.approx(100 * 5 / 40)
    assert summary.cx_depth_reduction_pct == pytest.approx(100 * 2 / 20)


def test_optimization_summary_is_frozen_and_hashable():
    # equivalence_counts is a tuple, not a dict, so the frozen summary stays immutable + hashable.
    summary = OptimizationSummary.from_stats(
        [OptimizationStats("a", 4, 10, 8, 20, 15, 6, 4, ("strip",), BY_CONSTRUCTION)]
    )
    assert isinstance(hash(summary), int)


def test_format_optimization_table_includes_rows_and_total():
    stats = optimize_circuits([("demo", _reducible_circuit())])
    table = format_optimization_table(stats)

    assert "demo" in table
    assert "TOTAL (1)" in table
    assert "guarantees:" in table


def test_format_optimization_table_columns_stay_aligned():
    # A long name and a circuit that grows must not break the grid. Columns are sized to the data,
    # so every grid line (header, rule, body rows, total) is exactly the same width; only the
    # trailing "improved ..." summary line differs.
    stats = [
        OptimizationStats(
            "very_easy/challenge-64_10", 64, 100, 90, 1000, 900, 200, 150, ("strip",),
            BY_CONSTRUCTION,
        ),
        OptimizationStats("x" * 60, 8, 5, 9, 4, 12, 0, 3, ("snap",), UNVERIFIED),  # grows (-pct)
    ]
    lines = format_optimization_table(stats).splitlines()

    grid_widths = {len(line) for line in lines[:-1]}
    assert len(grid_widths) == 1  # header + rules + both body rows + total all equal width
    assert "x" * 60 in lines[3]  # long name rendered in full, not truncated
    assert "(-200%)" in lines[3]  # 4->12 growth shown as a negative reduction
