"""Reading submission results from ``results/*_bitstrings.csv`` and judging challenges.

Each CSV row is one *submission* of a peak-bitstring guess for a challenge. A single
challenge can have several rows: when a guess is rejected it is recorded with status
``failed`` and another bitstring is tried, so a challenge may accumulate multiple failed
bitstrings over time. The helpers here aggregate **every** row for a challenge so all of
those failures are considered when judging whether the challenge has worked so far.

CSV columns: ``challenge``, ``qubits``, ``method``, ``bitstring``, ``probability``,
``status`` (one of ``success`` / ``failed`` / ``pending``).
"""

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RESULTS_DIR = Path("results")
STATUSES = ("success", "failed", "pending")


@dataclass(frozen=True)
class SubmissionRecord:
    """A single recorded submission, parsed from one CSV row.

    Attributes:
        challenge (str): Challenge name, e.g. ``"challenge-16_12"``.
        qubits (int): Number of qubits in the circuit.
        method (str): Simulation method that produced the bitstring (e.g. ``"MPS"``).
        bitstring (str): The submitted peak-bitstring guess.
        probability (float | None): Estimated probability in ``[0, 1]``, or ``None`` if blank.
        status (str): One of ``"success"``, ``"failed"``, ``"pending"``.
        difficulty (str): Difficulty tier, derived from the CSV filename (e.g. ``"easy"``).
    """

    challenge: str
    qubits: int
    method: str
    bitstring: str
    probability: float | None
    status: str
    difficulty: str


@dataclass(frozen=True)
class ChallengeStatus:
    """Aggregated outcome for one challenge across all of its submissions.

    Attributes:
        challenge (str): The (normalized) challenge name that was looked up.
        records (tuple[SubmissionRecord, ...]): Every submission recorded for this challenge,
            in file order.
        worked (bool | None): ``True`` if at least one submission succeeded; ``False`` if
            submissions exist but none succeeded; ``None`` if nothing conclusive is recorded
            (no matching rows, or only ``pending`` ones).
    """

    challenge: str
    records: tuple[SubmissionRecord, ...]
    worked: bool | None

    @property
    def successes(self) -> list[SubmissionRecord]:
        """Return the accepted submissions.

        Returns:
            list[SubmissionRecord]: Records whose status is ``"success"``.
        """
        return [r for r in self.records if r.status == "success"]

    @property
    def failures(self) -> list[SubmissionRecord]:
        """Return the rejected submissions.

        Returns:
            list[SubmissionRecord]: Records whose status is ``"failed"``.
        """
        return [r for r in self.records if r.status == "failed"]

    @property
    def pending(self) -> list[SubmissionRecord]:
        """Return the submissions with an unconfirmed outcome.

        Returns:
            list[SubmissionRecord]: Records whose status is ``"pending"``.
        """
        return [r for r in self.records if r.status == "pending"]

    @property
    def failed_bitstrings(self) -> list[str]:
        """Return every bitstring already known to fail, so they can be avoided.

        Returns:
            list[str]: Bitstrings from all ``failed`` submissions (may contain duplicates).
        """
        return [r.bitstring for r in self.failures]

    @property
    def summary(self) -> str:
        """Return a one-line, human-readable suggestion about this challenge.

        Returns:
            str: A recommendation reflecting whether the challenge worked so far, naming the
            accepted bitstring when solved and listing every failed bitstring to avoid.
        """
        if not self.records:
            return f"{self.challenge}: no submissions recorded yet - untried."

        fail_note = ""
        if self.failures:
            joined = ", ".join(self.failed_bitstrings)
            fail_note = f" {len(self.failures)} failed bitstring(s) to avoid: {joined}."

        if self.worked:
            best = max(self.successes, key=lambda r: r.probability or 0.0)
            prob = "n/a" if best.probability is None else f"{best.probability:.1%}"
            return (
                f"{self.challenge}: solved - accepted bitstring {best.bitstring} "
                f"({best.method}, {prob}).{fail_note}"
            )
        if self.worked is False:
            return (
                f"{self.challenge}: not solved — all {len(self.failures)} submission(s) "
                f"failed.{fail_note} Try a different bitstring."
            )
        return (
            f"{self.challenge}: {len(self.pending)} candidate(s) pending, "
            f"none confirmed yet.{fail_note}"
        )

    def __str__(self) -> str:
        """Return the human-readable suggestion.

        Returns:
            str: The value of :attr:`summary`.
        """
        return self.summary


def _record_from_row(row: dict[str, str], difficulty: str) -> SubmissionRecord:
    """Build a :class:`SubmissionRecord` from a parsed CSV row.

    Args:
        row (dict[str, str]): One row from :class:`csv.DictReader`.
        difficulty (str): Difficulty tier derived from the source filename.

    Returns:
        SubmissionRecord: The parsed record. A blank ``probability`` becomes ``None`` and a
        blank ``status`` defaults to ``"pending"``.
    """
    prob = (row.get("probability") or "").strip()
    status = (row.get("status") or "").strip().lower() or "pending"
    return SubmissionRecord(
        challenge=(row.get("challenge") or "").strip(),
        qubits=int(row["qubits"]),
        method=(row.get("method") or "").strip(),
        bitstring=(row.get("bitstring") or "").strip(),
        probability=float(prob) if prob else None,
        status=status,
        difficulty=difficulty,
    )


def load_results(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> list[SubmissionRecord]:
    """Load every submission from the ``*_bitstrings.csv`` files under ``results_dir``.

    Args:
        results_dir (Path | str): Directory holding the per-difficulty result CSVs.

    Returns:
        list[SubmissionRecord]: One record per CSV row across all difficulty files, ordered
        by filename then by row order within each file.
    """
    results_dir = Path(results_dir)
    records: list[SubmissionRecord] = []
    for csv_path in sorted(results_dir.glob("*_bitstrings.csv")):
        difficulty = csv_path.name.removesuffix("_bitstrings.csv")
        with csv_path.open(newline="", encoding="utf-8") as fh:
            records.extend(_record_from_row(row, difficulty) for row in csv.DictReader(fh))
    return records


def _normalize_challenge(challenge: str) -> str:
    """Reduce a challenge reference to its bare name.

    Strips any directory prefix (``"easy/challenge-16_12"``) and a trailing ``.qasm``
    extension so paths, ``iter_challenges`` names, and bare names all match.

    Args:
        challenge (str): A challenge name, ``"<difficulty>/<name>"``, or path/filename.

    Returns:
        str: The bare challenge name, e.g. ``"challenge-16_12"``.
    """
    name = challenge.replace("\\", "/").split("/")[-1].strip()
    return name.removesuffix(".qasm")


def challenge_status(
    challenge: str,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    records: Iterable[SubmissionRecord] | None = None,
) -> ChallengeStatus:
    """Judge whether a challenge has worked so far, considering all of its submissions.

    Gathers every recorded submission for ``challenge`` and reports an aggregate verdict:
    solved if any submission succeeded, otherwise failed (with **all** failed bitstrings
    collected so none are retried), otherwise pending/untried. Accepts a bare name
    (``"challenge-16_12"``), an ``iter_challenges`` name (``"easy/challenge-16_12"``), or a
    path/filename.

    Args:
        challenge (str): Challenge to look up (name, ``difficulty/name``, or path).
        results_dir (Path | str): Directory of result CSVs; used only when ``records`` is
            ``None``.
        records (Iterable[SubmissionRecord] | None): Pre-loaded records to query instead of
            reading from disk; handy for batching or testing.

    Returns:
        ChallengeStatus: The aggregated outcome, including the matching records, the
        ``worked`` verdict, and a human-readable :attr:`~ChallengeStatus.summary`.
    """
    name = _normalize_challenge(challenge)
    pool = list(records) if records is not None else load_results(results_dir)
    matched = tuple(r for r in pool if r.challenge == name)

    if any(r.status == "success" for r in matched):
        worked: bool | None = True
    elif any(r.status == "failed" for r in matched):
        worked = False
    else:
        worked = None
    return ChallengeStatus(challenge=name, records=matched, worked=worked)


def is_known_failure(
    challenge: str,
    bitstring: str,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    records: Iterable[SubmissionRecord] | None = None,
) -> bool:
    """Report whether ``bitstring`` was already submitted for ``challenge`` and rejected.

    Args:
        challenge (str): Challenge to look up (name, ``difficulty/name``, or path).
        bitstring (str): The candidate bitstring to check against known failures.
        results_dir (Path | str): Directory of result CSVs; used only when ``records`` is
            ``None``.
        records (Iterable[SubmissionRecord] | None): Pre-loaded records to query instead of
            reading from disk.

    Returns:
        bool: ``True`` if ``bitstring`` appears among this challenge's failed submissions.
    """
    status = challenge_status(challenge, results_dir=results_dir, records=records)
    return bitstring in status.failed_bitstrings
