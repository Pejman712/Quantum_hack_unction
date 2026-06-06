import math

from qiskit import QuantumCircuit

from quantum_hack.optimize import (
    BY_CONSTRUCTION,
    PEAK_VERIFIED,
    cost,
    optimize_circuit,
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


def test_snap_accepted_when_angles_already_on_grid_gives_peak_verified():
    qc = QuantumCircuit(2)
    qc.rx(math.pi / 4, 0)  # already a multiple of pi/4 -> snapping leaves the peak unchanged
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
