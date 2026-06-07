import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from quantum_hack.simulation import statevector_simulation
from quantum_hack.transform import (
    DEFAULT_BASIS_GATES,
    PI_OVER_8,
    optimize_with_pyzx,
    snap_rotations_to_pi_over_8,
    transpile_to_basis,
)


def _ops(qc: QuantumCircuit) -> set[str]:
    return set(qc.count_ops())


def test_transpile_to_basis_uses_default_rx_rz_cx():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.t(1)
    qc.cx(0, 1)

    out = transpile_to_basis(qc)

    assert DEFAULT_BASIS_GATES == ["rx", "rz", "cx"]
    assert _ops(out) <= set(DEFAULT_BASIS_GATES)


def test_transpile_to_basis_respects_custom_basis():
    qc = QuantumCircuit(1)
    qc.x(0)

    out = transpile_to_basis(qc, basis_gates=["u", "cx"])

    assert _ops(out) <= {"u", "cx"}


def test_transpile_to_basis_returns_new_circuit():
    qc = QuantumCircuit(1)
    qc.h(0)

    out = transpile_to_basis(qc)

    assert out is not qc
    assert _ops(qc) == {"h"}  # original untouched


def test_transpile_to_basis_preserves_peak_through_swap_elision():
    # opt3 elides the SWAP into a layout permutation, which (without the fix) relabels the peak
    # bitstring. preserve_qubit_order=True must restore the input order without re-adding the SWAP.
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.swap(0, 2)
    peak = statevector_simulation(qc)[0]

    restored = transpile_to_basis(qc, optimization_level=3)
    raw = transpile_to_basis(qc, optimization_level=3, preserve_qubit_order=False)

    assert statevector_simulation(restored)[0] == peak  # order restored -> same peak
    assert statevector_simulation(raw)[0] != peak  # the bug is real: raw output permutes the peak
    # Relabelling renames wires only; it must not change gate counts (nothing is undone).
    assert dict(restored.count_ops()).get("cx", 0) == dict(raw.count_ops()).get("cx", 0)


def test_transpile_to_basis_restores_order_when_routing_adds_ancillas():
    # With a coupling map, routing adds ancilla qubits and permutes the active ones. The full-width
    # relabel must still restore the logical peak (ancilla wires sit on the high indices as |0>).
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.swap(0, 2)
    qc.cx(0, 1)
    peak = statevector_simulation(qc)[0]

    restored = transpile_to_basis(
        qc, optimization_level=3, coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]], seed_transpiler=1
    )

    assert restored.num_qubits == 5  # two ancillas were added by routing
    assert statevector_simulation(restored)[0].endswith(peak)  # logical peak preserved on wires 0-2


def _angles(qc: QuantumCircuit, name: str) -> list[float]:
    return [
        float(instr.operation.params[0])
        for instr in qc.data
        if instr.operation.name == name
    ]


def test_snaps_near_multiple_to_exact_multiple():
    qc = QuantumCircuit(1)
    qc.rx(PI_OVER_8 + 0.003, 0)  # within 1% of pi/8
    qc.rz(2 * PI_OVER_8 - 0.002, 0)  # within 1% of pi/4

    out = snap_rotations_to_pi_over_8(qc)

    assert _angles(out, "rx")[0] == pytest.approx(PI_OVER_8)
    assert _angles(out, "rz")[0] == pytest.approx(2 * PI_OVER_8)


def test_snaps_near_zero_to_zero():
    qc = QuantumCircuit(1)
    qc.rx(0.002, 0)

    out = snap_rotations_to_pi_over_8(qc)

    assert _angles(out, "rx")[0] == 0.0


def test_rz_never_snaps_to_zero():
    # An RZ that would snap to 0 (or to a multiple of 2*pi, which wraps to 0) is left
    # untouched -- zeroing it would discard a real phase. RX still snaps to 0 (above).
    qc = QuantumCircuit(2)
    qc.rz(0.002, 0)  # within tolerance of 0
    qc.rz(2 * math.pi + 0.002, 1)  # within tolerance of 2*pi, which wraps to 0

    out = snap_rotations_to_pi_over_8(qc)

    assert _angles(out, "rz") == [0.002, 2 * math.pi + 0.002]  # both unchanged


def test_rz_snaps_to_zero_when_opted_in():
    # With snap_rz_to_zero=True, near-zero RZ angles snap to 0 (wrapped into (-pi, pi]).
    qc = QuantumCircuit(2)
    qc.rz(0.002, 0)  # within tolerance of 0
    qc.rz(2 * math.pi + 0.002, 1)  # within tolerance of 2*pi, which wraps to 0

    out = snap_rotations_to_pi_over_8(qc, snap_rz_to_zero=True)

    assert _angles(out, "rz")[0] == pytest.approx(0.0, abs=1e-12)
    assert _angles(out, "rz")[1] == pytest.approx(0.0, abs=1e-12)


def test_leaves_angle_far_from_grid_untouched():
    qc = QuantumCircuit(1)
    qc.rx(PI_OVER_8 / 2, 0)  # exactly halfway between 0 and pi/8

    out = snap_rotations_to_pi_over_8(qc)

    assert _angles(out, "rx")[0] == pytest.approx(PI_OVER_8 / 2)


def test_threshold_is_adjustable():
    qc = QuantumCircuit(1)
    qc.rz(PI_OVER_8 + 0.02, 0)  # ~5% off — outside default 1% but inside 10%

    default = snap_rotations_to_pi_over_8(qc)
    loose = snap_rotations_to_pi_over_8(qc, threshold=0.1)

    assert _angles(default, "rz")[0] == pytest.approx(PI_OVER_8 + 0.02)
    assert _angles(loose, "rz")[0] == pytest.approx(PI_OVER_8)


def test_other_gates_and_structure_preserved():
    qc = QuantumCircuit(2, 2)
    qc.rx(PI_OVER_8 + 0.002, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    out = snap_rotations_to_pi_over_8(qc)

    names = [instr.operation.name for instr in out.data]
    assert names == ["rx", "cx", "measure", "measure"]
    assert out.num_qubits == 2
    assert out.num_clbits == 2


def test_does_not_mutate_input():
    qc = QuantumCircuit(1)
    qc.rx(PI_OVER_8 + 0.002, 0)

    snap_rotations_to_pi_over_8(qc)

    assert float(qc.data[0].operation.params[0]) == PI_OVER_8 + 0.002


def test_negative_angle_snaps():
    qc = QuantumCircuit(1)
    qc.rz(-PI_OVER_8 + 0.001, 0)

    out = snap_rotations_to_pi_over_8(qc)

    assert _angles(out, "rz")[0] == pytest.approx(-PI_OVER_8)


def test_snapped_result_wrapped_into_minus_pi_pi():
    qc = QuantumCircuit(3)
    qc.rx(2 * math.pi - 0.002, 0)  # ~2*pi -> snaps to 2*pi, wraps to 0
    qc.rz(3 * math.pi / 2 - 0.002, 1)  # ~3*pi/2 -> snaps, wraps to -pi/2
    qc.rz(-math.pi + 0.001, 2)  # ~-pi -> snaps to -pi, wraps to +pi

    out = snap_rotations_to_pi_over_8(qc)

    assert _angles(out, "rx")[0] == pytest.approx(0.0, abs=1e-12)
    assert _angles(out, "rz")[0] == pytest.approx(-math.pi / 2)
    assert _angles(out, "rz")[1] == pytest.approx(math.pi)


def test_unbound_parameter_angle_left_untouched():
    # A symbolic (unbound) angle can't be cast to float, so snapping must skip it and
    # leave the gate exactly as-is rather than raising.
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)

    out = snap_rotations_to_pi_over_8(qc)

    assert out.data[0].operation.params[0] == theta


def test_all_snapped_angles_are_within_minus_pi_pi():
    qc = QuantumCircuit(1)
    for k in range(-10, 11):
        qc.rx(k * PI_OVER_8 + 0.001, 0)

    out = snap_rotations_to_pi_over_8(qc)

    for angle in _angles(out, "rx"):
        assert -math.pi < angle <= math.pi + 1e-12


def _pyzx_test_circuit() -> QuantumCircuit:
    """Small entangling circuit with arbitrary angles for the pyzx round-trip tests.

    Returns:
        QuantumCircuit: A 3-qubit circuit mixing rx/rz rotations and CX gates.
    """
    qc = QuantumCircuit(3)
    qc.rx(0.7, 0)
    qc.cx(0, 1)
    qc.rz(1.1, 1)
    qc.cx(1, 2)
    qc.rx(2.4, 2)
    qc.cx(0, 2)
    return qc


def test_optimize_with_pyzx_preserves_peak_and_returns_basis():
    pytest.importorskip("pyzx")
    qc = _pyzx_test_circuit()

    out = optimize_with_pyzx(qc)

    # full_reduce preserves the unitary up to global phase, so the peak bitstring is unchanged.
    assert statevector_simulation(out)[0] == statevector_simulation(qc)[0]
    assert _ops(out) <= set(DEFAULT_BASIS_GATES)


def test_optimize_with_pyzx_respects_custom_basis():
    pytest.importorskip("pyzx")
    qc = _pyzx_test_circuit()

    out = optimize_with_pyzx(qc, basis_gates=["u", "cx"])

    assert _ops(out) <= {"u", "cx"}


def test_optimize_with_pyzx_does_not_mutate_input():
    pytest.importorskip("pyzx")
    qc = _pyzx_test_circuit()
    before = len(qc.data)

    optimize_with_pyzx(qc)

    assert len(qc.data) == before


def test_optimize_with_pyzx_drops_final_measurements():
    pytest.importorskip("pyzx")
    qc = _pyzx_test_circuit()
    qc.measure_all()

    out = optimize_with_pyzx(qc)

    assert out.num_clbits == 0
