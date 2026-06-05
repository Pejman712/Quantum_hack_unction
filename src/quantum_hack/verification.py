"""Functional equivalence checking of two circuits via MQT QCEC."""

from typing import TYPE_CHECKING

from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from mqt.qcec import EquivalenceCriterion

# Criteria that mean the circuits implement the same operation. Global and relative
# phase are not observable, so those count as equivalent.
EQUIVALENT_CRITERIA = frozenset(
    {"equivalent", "equivalent_up_to_global_phase", "equivalent_up_to_phase"}
)


def verify_equivalence(
    qc1: QuantumCircuit, qc2: QuantumCircuit, **verify_kwargs: object
) -> "EquivalenceCriterion":
    """Return the QCEC ``EquivalenceCriterion`` for two circuits.

    Thin wrapper over :func:`mqt.qcec.verify`; extra keyword arguments are forwarded
    to it. ``mqt.qcec`` is imported lazily so that importing ``quantum_hack`` does not
    require the (compiled) QCEC package until this function is actually called.

    Args:
        qc1 (QuantumCircuit): First circuit to compare.
        qc2 (QuantumCircuit): Second circuit to compare.
        **verify_kwargs (object): Extra keyword arguments forwarded to :func:`mqt.qcec.verify`.

    Returns:
        EquivalenceCriterion: The equivalence criterion reported by QCEC.
    """
    from mqt.qcec import verify

    return verify(qc1, qc2, **verify_kwargs).equivalence


def circuits_equivalent(
    qc1: QuantumCircuit, qc2: QuantumCircuit, **verify_kwargs: object
) -> bool:
    """Return ``True`` if QCEC proves the two circuits are functionally equivalent.

    Equivalence up to global/relative phase counts as equivalent (phase is not
    observable), so a transpiled/optimized circuit compares equal to its original.
    A non-definitive ``probably_equivalent`` result returns ``False``; use
    :func:`verify_equivalence` when you need the full criterion.

    Args:
        qc1 (QuantumCircuit): First circuit to compare.
        qc2 (QuantumCircuit): Second circuit to compare.
        **verify_kwargs (object): Extra keyword arguments forwarded to :func:`verify_equivalence`.

    Returns:
        bool: ``True`` if the circuits are functionally equivalent, else ``False``.
    """
    return verify_equivalence(qc1, qc2, **verify_kwargs).name in EQUIVALENT_CRITERIA
