"""Circuit metrics and before/after comparison helpers."""

from qiskit import QuantumCircuit, transpile


def circuit_stats(qc: QuantumCircuit) -> dict:
    """Return basic size metrics for ``qc``: qubits, depth, gate count, and op counts.

    Args:
        qc (QuantumCircuit): Circuit to measure.

    Returns:
        dict: Metrics with keys ``num_qubits`` (int), ``depth`` (int), ``size`` (int),
        and ``ops`` (dict[str, int]).
    """

    qc_transpiled = transpile(qc, basis_gates=["rz", "rx", "cx"], optimization_level=0)
    return {
        "num_qubits": qc_transpiled.num_qubits,
        "depth": qc_transpiled.depth(),
        "size": qc_transpiled.size(),
        "ops": dict(qc_transpiled.count_ops()),
    }


def compare_circuits(before: QuantumCircuit, after: QuantumCircuit) -> dict:
    """Compare two circuits and return before/after/delta for depth, size, and each gate.

    Useful for measuring whether a transpilation/optimization pass helped, e.g.
    ``compare_circuits(qc, optimized_qc)["ops"]["cx"]`` -> ``{"before": 42, "after": 18,
    "delta": -24}``. A negative delta means the metric shrank.

    Args:
        before (QuantumCircuit): Circuit before the transformation.
        after (QuantumCircuit): Circuit after the transformation.

    Returns:
        dict: Keys ``depth``, ``size``, and ``ops`` mapping to
        ``{"before": int, "after": int, "delta": int}`` entries (``ops`` keyed by gate name).
    """
    b, a = circuit_stats(before), circuit_stats(after)

    def _entry(x: int, y: int) -> dict:
        """Build a before/after/delta entry for a single metric.

        Args:
            x (int): Value before the transformation.
            y (int): Value after the transformation.

        Returns:
            dict: ``{"before": int, "after": int, "delta": int}`` where ``delta`` is ``y - x``.
        """
        return {"before": x, "after": y, "delta": y - x}

    gates = sorted(set(b["ops"]) | set(a["ops"]))
    return {
        "depth": _entry(b["depth"], a["depth"]),
        "size": _entry(b["size"], a["size"]),
        "ops": {g: _entry(b["ops"].get(g, 0), a["ops"].get(g, 0)) for g in gates},
    }
