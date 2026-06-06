"""Circuit transforms: basis transpilation and angle snapping."""

import math

from qiskit import QuantumCircuit, transpile

PI_OVER_4 = math.pi / 4
TWO_PI = 2 * math.pi
DEFAULT_BASIS_GATES = ["rx", "rz", "cx"]


def transpile_to_basis(
    qc: QuantumCircuit,
    basis_gates: list[str] | None = None,
    optimization_level: int = 0,
    **transpile_kwargs: object,
) -> QuantumCircuit:
    """Transpile ``qc`` to a basis gate set (default ``rx``, ``rz``, ``cx``).

    ``optimization_level`` is forwarded to :func:`qiskit.transpile` (default ``1``,
    matching Qiskit's own default); use ``0`` to only rewrite into the basis without
    optimizing, or ``2``/``3`` for more aggressive optimization. Any further keyword
    arguments are passed through to :func:`qiskit.transpile`. Returns a new circuit.

    Args:
        qc (QuantumCircuit): Circuit to transpile.
        basis_gates (list[str] | None): Target basis gates; ``None`` uses the default set.
        optimization_level (int): Optimization level forwarded to :func:`qiskit.transpile`.
        **transpile_kwargs (object): Extra keyword arguments forwarded to :func:`qiskit.transpile`.

    Returns:
        QuantumCircuit: A new circuit rewritten into the basis gate set.
    """
    if basis_gates is None:
        basis_gates = DEFAULT_BASIS_GATES
    return transpile(
        qc,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        **transpile_kwargs,
    )


def _wrap_to_pi(angle: float) -> float:
    """Wrap ``angle`` into the half-open interval (-pi, pi].

    Args:
        angle (float): Angle in radians.

    Returns:
        float: The angle wrapped into (-pi, pi].
    """
    return math.pi - ((math.pi - angle) % TWO_PI)


def snap_rotations_to_pi_over_4(
    qc: QuantumCircuit, threshold: float = 0.01
) -> QuantumCircuit:
    """Return a copy of ``qc`` with near-grid RZ/RX angles snapped to a multiple of pi/4.

    For every RZ and RX gate, if its angle is within ``threshold`` of the nearest
    multiple of pi/4 it is replaced by that exact multiple, wrapped into (-pi, pi].
    ``threshold`` is a fraction of the pi/4 grid spacing, so the absolute tolerance
    is ``threshold * pi/4`` (default ``0.01`` -> 1% of pi/4). Because the pi/4 grid
    repeats every 2*pi, the wrap does not change which multiple is nearest.

    An RZ is never snapped to 0 (a multiple of 2*pi): zeroing it would discard a
    real, if small, phase, so such an RZ keeps its original angle. RX may still
    snap to 0.

    Gates other than RZ/RX, and any RZ/RX whose angle is a symbolic (unbound)
    parameter, are left untouched. Qubits, clbits, and measurements are preserved.

    Args:
        qc (QuantumCircuit): Circuit whose rotation angles should be snapped.
        threshold (float): Snap tolerance as a fraction of the pi/4 grid spacing.

    Returns:
        QuantumCircuit: A new circuit with near-grid RZ/RX angles snapped to multiples of pi/4.
    """
    tolerance = threshold * PI_OVER_4
    new_qc = qc.copy_empty_like()
    for instruction in qc.data:
        op = instruction.operation
        if op.name in ("rz", "rx") and len(op.params) == 1:
            try:
                angle = float(op.params[0])
            except (TypeError, ValueError):
                pass  # symbolic / unbound parameter — leave it alone
            else:
                steps = round(angle / PI_OVER_4)
                snapped = steps * PI_OVER_4
                # Never snap an RZ to 0 (steps a multiple of 8 -> a multiple of 2*pi):
                # zeroing an RZ would discard a real phase. RX may still snap to 0.
                snaps_rz_to_zero = op.name == "rz" and steps % 8 == 0
                if abs(angle - snapped) <= tolerance and not snaps_rz_to_zero:
                    op = op.copy()
                    op.params = [_wrap_to_pi(snapped)]
        new_qc.append(op, instruction.qubits, instruction.clbits)
    return new_qc
