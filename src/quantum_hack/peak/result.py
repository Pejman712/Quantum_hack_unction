"""The :class:`PeakResult` record and a confidence score for trusting a peak.

The challenge portal hides the correct answer, so a found bitstring carries no external
verification. :func:`score_confidence` combines three *internal* signals into a single
``[0, 1]`` score so the batch runner can flag which circuits to re-run before submitting:

* **magnitude** — the peak probability should be far above the uniform ``2**-n``.
* **local-max** — a true peak must beat every single-bit-flip neighbour; if a neighbour
  wins, it is provably not the peak.
* **consensus** — independent agreement of two or more methods is the strongest signal we
  have without ground truth.
"""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PeakResult:
    """A peak-bitstring estimate plus the evidence used to judge its trustworthiness.

    Attributes:
        bitstring (str): The estimated peak bitstring, in Qiskit ordering (rightmost char
            is qubit 0).
        probability (float): Estimated probability of ``bitstring`` (may be approximate).
        method (str): Name of the method that produced ``bitstring`` (e.g. ``"statevector"``,
            ``"MPS"``, ``"marginal"``, ``"greedy_refine"``).
        num_qubits (int): Number of qubits in the circuit.
        confidence (float): Composite confidence in ``[0, 1]`` (see :func:`score_confidence`).
        is_local_max (bool | None): ``True`` if ``bitstring`` beats every single-bit-flip
            neighbour, ``False`` if some neighbour is more probable, ``None`` if not tested.
        neighbor_gap (float | None): ``probability`` minus the best neighbour probability
            (positive confirms a local max), or ``None`` if not tested.
        agreeing_methods (tuple[str, ...]): Methods that independently produced ``bitstring``.
        notes (str): Free-text explanation for the presentation/writeup.
    """

    bitstring: str
    probability: float
    method: str
    num_qubits: int
    confidence: float = 0.0
    is_local_max: bool | None = None
    neighbor_gap: float | None = None
    agreeing_methods: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serialisable mapping of this result.

        Returns:
            dict: All fields, with ``agreeing_methods`` rendered as a ``list``.
        """
        data = asdict(self)
        data["agreeing_methods"] = list(self.agreeing_methods)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PeakResult":
        """Rebuild a :class:`PeakResult` from a mapping produced by :meth:`to_dict`.

        Args:
            data (dict): Mapping with the fields of :class:`PeakResult`.

        Returns:
            PeakResult: The reconstructed result.
        """
        fields = {**data, "agreeing_methods": tuple(data.get("agreeing_methods", ()))}
        return cls(**fields)


def _magnitude_signal(probability: float, num_qubits: int) -> float:
    """Score how far ``probability`` sits above the uniform ``2**-num_qubits`` baseline.

    Args:
        probability (float): Estimated peak probability.
        num_qubits (int): Number of qubits.

    Returns:
        float: ``0.0`` at (or below) the uniform baseline, ``1.0`` at probability ``1``.
    """
    if probability <= 0.0 or num_qubits <= 0:
        return 0.0
    # (log2(p) - log2(2**-n)) / n = (log2(p) + n) / n, clamped to [0, 1].
    return max(0.0, min(1.0, (math.log2(probability) + num_qubits) / num_qubits))


def score_confidence(
    *,
    probability: float,
    num_qubits: int,
    is_local_max: bool | None,
    agreeing_methods: int,
    neighbor_gap: float | None = None,
) -> float:
    """Combine magnitude, local-max, and consensus signals into a confidence in ``[0, 1]``.

    The three signals are combined as a geometric mean so that any single weak signal pulls
    the score down. A confirmed non-local-max yields ``0.0`` (it cannot be the peak).

    Args:
        probability (float): Estimated peak probability.
        num_qubits (int): Number of qubits.
        is_local_max (bool | None): ``True`` if it beats every 1-bit-flip neighbour, ``False``
            if some neighbour wins, ``None`` if untested.
        agreeing_methods (int): Number of methods that independently produced the bitstring
            (``1`` means a single method).
        neighbor_gap (float | None): Unused in the score directly; accepted for symmetry with
            :class:`PeakResult` and future tuning.

    Returns:
        float: Composite confidence in ``[0, 1]``.
    """
    magnitude = _magnitude_signal(probability, num_qubits)
    if is_local_max is True:
        local = 1.0
    elif is_local_max is False:
        local = 0.0
    else:
        local = 0.5
    consensus = min(1.0, 0.5 + 0.5 * (max(1, agreeing_methods) - 1))
    return (magnitude * local * consensus) ** (1.0 / 3.0)
