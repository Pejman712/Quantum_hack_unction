"""Circuit transforms: basis transpilation, angle snapping, and ZX-calculus reduction."""

import math

from qiskit import QuantumCircuit, qasm2, transpile

PI_OVER_8 = math.pi / 8
TWO_PI = 2 * math.pi
DEFAULT_BASIS_GATES = ["rx", "rz", "cx"]
# Number of pi/8 steps in a full turn; an RZ that is a multiple of 2*pi has a step count
# divisible by this (round(TWO_PI / PI_OVER_8) == 16).
_STEPS_PER_TURN = round(TWO_PI / PI_OVER_8)


def _restore_qubit_order(transpiled: QuantumCircuit) -> QuantumCircuit:
    """Relabel ``transpiled``'s qubits so its measurement outcomes match the pre-transpile order.

    At ``optimization_level >= 1`` the transpiler may permute qubits — chiefly by *eliding* SWAP
    gates (a real CX saving) and recording the net permutation in ``transpiled.layout`` instead of
    leaving it in the gate list. The circuit then computes the same unitary but on relabelled wires,
    so a computational-basis measurement returns the peak bitstring with its bits permuted. This
    rebuilds the circuit on the original wire labels: gate counts and depth are unchanged (the SWAP
    elision is kept, not undone), only the wire each instruction sits on is renamed back.

    Only measurement-free circuits are relabelled. When clbits are present the transpiler already
    reorders them to keep counts in the original basis, so the permutation is not observable and the
    circuit is returned untouched. ``optimization_level == 0`` has no layout and is also a no-op.

    Args:
        transpiled (QuantumCircuit): A circuit returned by :func:`qiskit.transpile`.

    Returns:
        QuantumCircuit: ``transpiled`` with qubits relabelled to the original order, or
        ``transpiled`` unchanged when there is no observable permutation to undo.
    """
    layout = transpiled.layout
    if layout is None or transpiled.num_clbits:
        return transpiled
    # final_index_layout()[i] is the physical wire that virtual qubit i ends up on.
    # filter_ancillas=False keeps any routing ancillas in the list, so ``final`` is always a full
    # permutation of range(num_qubits) and ``phys_to_orig`` indexing stays in bounds; without a
    # coupling map (this codebase's only use) no ancillas are added and it equals the qubit count.
    final = layout.final_index_layout(filter_ancillas=False)
    if len(final) != transpiled.num_qubits or list(final) == list(range(transpiled.num_qubits)):
        return transpiled
    phys_to_orig = [0] * len(final)
    for orig, phys in enumerate(final):
        phys_to_orig[phys] = orig
    restored = QuantumCircuit(transpiled.num_qubits)
    for instruction in transpiled.data:
        qubits = [phys_to_orig[transpiled.find_bit(q).index] for q in instruction.qubits]
        restored.append(instruction.operation, qubits)
    return restored


def transpile_to_basis(
    qc: QuantumCircuit,
    basis_gates: list[str] | None = None,
    optimization_level: int = 0,
    preserve_qubit_order: bool = True,
    **transpile_kwargs: object,
) -> QuantumCircuit:
    """Transpile ``qc`` to a basis gate set (default ``rx``, ``rz``, ``cx``).

    ``optimization_level`` is forwarded to :func:`qiskit.transpile` (default ``1``,
    matching Qiskit's own default); use ``0`` to only rewrite into the basis without
    optimizing, or ``2``/``3`` for more aggressive optimization. Any further keyword
    arguments are passed through to :func:`qiskit.transpile`. Returns a new circuit.

    By default (``preserve_qubit_order=True``) any qubit permutation the transpiler introduces (e.g.
    by eliding SWAP gates at ``optimization_level >= 1``) is undone via :func:`_restore_qubit_order`
    so the circuit's measurement outcomes — including the peak bitstring — stay in the input qubit
    order. This keeps the gate savings while preventing a relabelled (and therefore wrong) peak
    bitstring; set it to ``False`` to return the raw transpiler output.

    Args:
        qc (QuantumCircuit): Circuit to transpile.
        basis_gates (list[str] | None): Target basis gates; ``None`` uses the default set.
        optimization_level (int): Optimization level forwarded to :func:`qiskit.transpile`.
        preserve_qubit_order (bool): If ``True`` (default), relabel qubits back to the input order
            when the transpiler permutes them (measurement-free circuits only).
        **transpile_kwargs (object): Extra keyword arguments forwarded to :func:`qiskit.transpile`.

    Returns:
        QuantumCircuit: A new circuit rewritten into the basis gate set.
    """
    if basis_gates is None:
        basis_gates = DEFAULT_BASIS_GATES
    transpiled = transpile(
        qc,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        **transpile_kwargs,
    )
    if preserve_qubit_order:
        transpiled = _restore_qubit_order(transpiled)
    return transpiled


def _wrap_to_pi(angle: float) -> float:
    """Wrap ``angle`` into the half-open interval (-pi, pi].

    Args:
        angle (float): Angle in radians.

    Returns:
        float: The angle wrapped into (-pi, pi].
    """
    return math.pi - ((math.pi - angle) % TWO_PI)


def snap_rotations_to_pi_over_8(
    qc: QuantumCircuit, threshold: float = 0.01, snap_rz_to_zero: bool = False
) -> QuantumCircuit:
    """Return a copy of ``qc`` with near-grid RZ/RX angles snapped to a multiple of pi/8.

    For every RZ and RX gate, if its angle is within ``threshold`` of the nearest
    multiple of pi/8 it is replaced by that exact multiple, wrapped into (-pi, pi].
    ``threshold`` is a fraction of the pi/8 grid spacing, so the absolute tolerance
    is ``threshold * pi/8`` (default ``0.01`` -> 1% of pi/8). Because the pi/8 grid
    repeats every 2*pi, the wrap does not change which multiple is nearest.

    By default an RZ is never snapped to 0 (a multiple of 2*pi): zeroing it would
    discard a real, if small, phase, so such an RZ keeps its original angle. Set
    ``snap_rz_to_zero`` to True to allow RZ angles near a multiple of 2*pi to snap
    to 0 as well. RX always snaps to 0 regardless of this flag.

    Gates other than RZ/RX, and any RZ/RX whose angle is a symbolic (unbound)
    parameter, are left untouched. Qubits, clbits, and measurements are preserved.

    Args:
        qc (QuantumCircuit): Circuit whose rotation angles should be snapped.
        threshold (float): Snap tolerance as a fraction of the pi/8 grid spacing.
        snap_rz_to_zero (bool): If True, allow near-zero RZ angles to snap to 0.
            If False (default), RZ angles are never snapped to 0.

    Returns:
        QuantumCircuit: A new circuit with near-grid RZ/RX angles snapped to multiples of pi/8.
    """
    tolerance = threshold * PI_OVER_8
    new_qc = qc.copy_empty_like()
    for instruction in qc.data:
        op = instruction.operation
        if op.name in ("rz", "rx") and len(op.params) == 1:
            try:
                angle = float(op.params[0])
            except (TypeError, ValueError):
                pass  # symbolic / unbound parameter — leave it alone
            else:
                steps = round(angle / PI_OVER_8)
                snapped = steps * PI_OVER_8
                # Unless snap_rz_to_zero is set, never snap an RZ to 0 (steps a multiple of
                # _STEPS_PER_TURN -> a multiple of 2*pi): zeroing an RZ would discard a real
                # phase. RX may always snap to 0.
                snaps_rz_to_zero = (
                    op.name == "rz" and steps % _STEPS_PER_TURN == 0 and not snap_rz_to_zero
                )
                if abs(angle - snapped) <= tolerance and not snaps_rz_to_zero:
                    op = op.copy()
                    op.params = [_wrap_to_pi(snapped)]
        new_qc.append(op, instruction.qubits, instruction.clbits)
    return new_qc


def optimize_with_pyzx(
    qc: QuantumCircuit,
    basis_gates: list[str] | None = None,
    basic_optimization: bool = True,
) -> QuantumCircuit:
    """Reduce ``qc`` with PyZX's ZX-calculus simplification and return a Qiskit circuit.

    The circuit is normalised into the ``rx``/``rz``/``cx`` basis and round-tripped through PyZX as
    OpenQASM 2.0: it is turned into a ZX-graph, simplified with :func:`pyzx.simplify.full_reduce`,
    re-extracted with :func:`pyzx.extract_circuit`, and (optionally) cleaned up with
    :func:`pyzx.optimize.basic_optimization`, before being transpiled back into the basis. Because
    ``full_reduce`` preserves the circuit's linear map (its unitary up to global phase), every
    measurement outcome — including the peak bitstring — is preserved by construction, the same
    guarantee as :func:`transpile_to_basis`.

    Final measurements are removed before the round trip (PyZX operates on a pure unitary), so the
    returned circuit carries no classical bits, matching :func:`~quantum_hack.strip.strip`. Note
    that ZX extraction is tuned for Clifford+T circuits: on circuits dominated by arbitrary
    continuous rotations (like the peaked challenge circuits) it preserves the unitary but often
    *grows* the gate count, so a caller wanting a guaranteed improvement should compare costs and
    keep the result only if it shrank (which is what :func:`~quantum_hack.optimize.optimize_circuit`
    does).

    Args:
        qc (QuantumCircuit): Circuit to reduce.
        basis_gates (list[str] | None): Basis to transpile the result back into; ``None`` (default)
            uses :data:`DEFAULT_BASIS_GATES` (``rx``/``rz``/``cx``).
        basic_optimization (bool): If ``True`` (default), run
            :func:`pyzx.optimize.basic_optimization` as a final peephole pass; it is best-effort
            and skipped if it raises.

    Returns:
        QuantumCircuit: A new circuit implementing the same unitary as ``qc`` up to global phase.

    Raises:
        ImportError: If PyZX is not installed (it is the optional ``[zx]`` extra).
    """
    try:
        import pyzx as zx
    except ImportError as exc:  # optional dependency, declared as the [zx] extra
        raise ImportError(
            "optimize_with_pyzx requires pyzx; install it with "
            "'pip install \"quantum-hack-unction[zx]\"' (or 'pip install pyzx')."
        ) from exc

    unitary = qc.remove_final_measurements(inplace=False)
    assert unitary is not None  # inplace=False always returns a new circuit
    unitary = transpile_to_basis(unitary, optimization_level=0)

    graph = zx.Circuit.from_qasm(qasm2.dumps(unitary)).to_graph()
    zx.simplify.full_reduce(graph)
    graph.normalize()
    reduced = zx.extract_circuit(graph.copy()).to_basic_gates()
    if basic_optimization:
        try:
            reduced = zx.optimize.basic_optimization(reduced).to_basic_gates()
        except Exception:  # noqa: BLE001 - peephole pass is best-effort; keep the extracted circuit
            pass

    result = qasm2.loads(reduced.to_qasm())
    return transpile_to_basis(result, basis_gates=basis_gates, optimization_level=0)
