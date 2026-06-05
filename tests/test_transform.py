import math

import pytest
from qiskit import QuantumCircuit

from quantum_hack.transform import (
    DEFAULT_BASIS_GATES,
    PI_OVER_4,
    snap_rotations_to_pi_over_4,
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


def _angles(qc: QuantumCircuit, name: str) -> list[float]:
    return [
        float(instr.operation.params[0])
        for instr in qc.data
        if instr.operation.name == name
    ]


def test_snaps_near_multiple_to_exact_multiple():
    qc = QuantumCircuit(1)
    qc.rx(PI_OVER_4 + 0.003, 0)  # within 1% of pi/4
    qc.rz(2 * PI_OVER_4 - 0.004, 0)  # within 1% of pi/2

    out = snap_rotations_to_pi_over_4(qc)

    assert _angles(out, "rx")[0] == pytest.approx(PI_OVER_4)
    assert _angles(out, "rz")[0] == pytest.approx(2 * PI_OVER_4)


def test_snaps_near_zero_to_zero():
    qc = QuantumCircuit(1)
    qc.rx(0.002, 0)

    out = snap_rotations_to_pi_over_4(qc)

    assert _angles(out, "rx")[0] == 0.0


def test_leaves_angle_far_from_grid_untouched():
    qc = QuantumCircuit(1)
    qc.rx(PI_OVER_4 / 2, 0)  # exactly halfway between 0 and pi/4

    out = snap_rotations_to_pi_over_4(qc)

    assert _angles(out, "rx")[0] == pytest.approx(PI_OVER_4 / 2)


def test_threshold_is_adjustable():
    qc = QuantumCircuit(1)
    qc.rz(PI_OVER_4 + 0.05, 0)  # ~6.4% off — outside default 1%

    default = snap_rotations_to_pi_over_4(qc)
    loose = snap_rotations_to_pi_over_4(qc, threshold=0.1)

    assert _angles(default, "rz")[0] == pytest.approx(PI_OVER_4 + 0.05)
    assert _angles(loose, "rz")[0] == pytest.approx(PI_OVER_4)


def test_other_gates_and_structure_preserved():
    qc = QuantumCircuit(2, 2)
    qc.rx(PI_OVER_4 + 0.002, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    out = snap_rotations_to_pi_over_4(qc)

    names = [instr.operation.name for instr in out.data]
    assert names == ["rx", "cx", "measure", "measure"]
    assert out.num_qubits == 2
    assert out.num_clbits == 2


def test_does_not_mutate_input():
    qc = QuantumCircuit(1)
    qc.rx(PI_OVER_4 + 0.002, 0)

    snap_rotations_to_pi_over_4(qc)

    assert float(qc.data[0].operation.params[0]) == PI_OVER_4 + 0.002


def test_negative_angle_snaps():
    qc = QuantumCircuit(1)
    qc.rz(-PI_OVER_4 + 0.001, 0)

    out = snap_rotations_to_pi_over_4(qc)

    assert _angles(out, "rz")[0] == pytest.approx(-PI_OVER_4)


def test_snapped_result_wrapped_into_minus_pi_pi():
    qc = QuantumCircuit(3)
    qc.rx(2 * math.pi - 0.002, 0)  # ~2*pi -> snaps to 2*pi, wraps to 0
    qc.rz(3 * math.pi / 2 - 0.002, 1)  # ~3*pi/2 -> snaps, wraps to -pi/2
    qc.rz(-math.pi + 0.001, 2)  # ~-pi -> snaps to -pi, wraps to +pi

    out = snap_rotations_to_pi_over_4(qc)

    assert _angles(out, "rx")[0] == pytest.approx(0.0, abs=1e-12)
    assert _angles(out, "rz")[0] == pytest.approx(-math.pi / 2)
    assert _angles(out, "rz")[1] == pytest.approx(math.pi)


def test_all_snapped_angles_are_within_minus_pi_pi():
    qc = QuantumCircuit(1)
    for k in range(-10, 11):
        qc.rx(k * PI_OVER_4 + 0.001, 0)

    out = snap_rotations_to_pi_over_4(qc)

    for angle in _angles(out, "rx"):
        assert -math.pi < angle <= math.pi + 1e-12
