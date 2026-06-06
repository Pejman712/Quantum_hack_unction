"""Batch-solve every challenge circuit, with checkpoint/resume and SLURM sharding.

Each circuit is solved by :func:`~quantum_hack.peak.solve.solve_peak` and its full
:class:`~quantum_hack.peak.result.PeakResult` (bitstring, probability, confidence, local-max
evidence, and the explanation notes for the presentation) is cached as one JSON file under
``results/_peak_cache/<difficulty>/<challenge>.json``. Per-circuit JSON files are safe to write
from many SLURM array tasks at once and make restarts idempotent: a finished circuit is skipped.

:func:`write_results_csvs` then appends *pending* candidate rows to the per-difficulty
``results/<difficulty>_bitstrings.csv`` files in the 6-column schema that
:mod:`quantum_hack.results` reads, without touching rows a human has already marked
``success``/``failed``. The rich evidence stays in the JSON cache and the report, keeping the
CSVs aligned with what the portal needs (a bitstring).
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty

from qiskit import QuantumCircuit

from quantum_hack.challenges import DEFAULT_DATA_DIR, DIFFICULTIES, load_circuit
from quantum_hack.optimize import default_pipeline, optimize_circuit
from quantum_hack.peak.result import PeakResult
from quantum_hack.peak.solve import solve_peak
from quantum_hack.results import (
    DEFAULT_RESULTS_DIR,
    ChallengeStatus,
    SubmissionRecord,
    challenge_status,
    load_results,
)

CACHE_DIRNAME = "_peak_cache"
CSV_COLUMNS = ("challenge", "qubits", "method", "bitstring", "probability", "status")
# Default per-circuit wall-clock budget (seconds) for the CLI; a circuit exceeding it is killed
# and recorded as a timeout rather than stalling the whole run. Overridable with --time-budget.
DEFAULT_TIME_BUDGET_S = 300.0

# Verdicts from comparing a cached solver answer to the recorded CSV submissions.
CACHE_MATCH = "match"  # cache bitstring equals an accepted (success) bitstring
CACHE_MISMATCH = "mismatch"  # an accepted bitstring exists and the cache disagrees with it
CACHE_KNOWN_FAILURE = "known_failure"  # cache bitstring was already submitted and rejected
CACHE_UNCONFIRMED = "unconfirmed"  # no accepted bitstring on record to compare against
# A cached answer is "failed" if it provably disagrees with the known-correct answer.
CACHE_FAILED_VERDICTS = frozenset({CACHE_MISMATCH, CACHE_KNOWN_FAILURE})

# A solver maps a circuit (plus keyword context) to a PeakResult.
Solver = Callable[..., PeakResult]


class CircuitJob:
    """One challenge circuit to solve.

    Attributes:
        name (str): Bare challenge name, e.g. ``"challenge-8_1"``.
        difficulty (str): Difficulty tier (parent folder name).
        path (Path): Path to the ``.qasm`` file.
        num_qubits (int): Qubit count parsed from the filename (``challenge-<n>_<id>``).
    """

    __slots__ = ("name", "difficulty", "path", "num_qubits")

    def __init__(self, name: str, difficulty: str, path: Path, num_qubits: int) -> None:
        """Initialise the job.

        Args:
            name (str): Bare challenge name.
            difficulty (str): Difficulty tier.
            path (Path): Path to the QASM file.
            num_qubits (int): Qubit count.
        """
        self.name = name
        self.difficulty = difficulty
        self.path = path
        self.num_qubits = num_qubits


def _parse_qubits(stem: str) -> int:
    """Parse the qubit count from a challenge filename stem.

    Args:
        stem (str): Filename stem, e.g. ``"challenge-24_3"``.

    Returns:
        int: The qubit count (``24`` here), or ``0`` if it cannot be parsed.
    """
    try:
        return int(stem.split("-", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError):
        return 0


def list_jobs(
    data_dir: Path | str = DEFAULT_DATA_DIR, difficulty: str | None = None
) -> list[CircuitJob]:
    """List challenge circuits, smallest first, so quick wins finish before the big ones.

    Args:
        data_dir (Path | str): Root directory of the challenge QASM files.
        difficulty (str | None): Restrict to one tier, or ``None`` for all.

    Returns:
        list[CircuitJob]: Jobs sorted by ``(num_qubits, name)``.
    """
    data_dir = Path(data_dir)
    search_root = data_dir if difficulty is None else data_dir / difficulty
    jobs = [
        CircuitJob(
            name=path.stem,
            difficulty=path.parent.name,
            path=path,
            num_qubits=_parse_qubits(path.stem),
        )
        for path in search_root.rglob("*.qasm")
    ]
    return sorted(jobs, key=lambda j: (j.num_qubits, j.name))


def _cache_path(results_dir: Path, job: CircuitJob) -> Path:
    """Return the JSON checkpoint path for a job.

    Args:
        results_dir (Path): Root results directory.
        job (CircuitJob): The job.

    Returns:
        Path: ``<results_dir>/_peak_cache/<difficulty>/<name>.json``.
    """
    return results_dir / CACHE_DIRNAME / job.difficulty / f"{job.name}.json"


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write ``data`` as JSON to ``path`` atomically (temp file then replace).

    Args:
        path (Path): Destination path.
        data (dict): JSON-serialisable mapping.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _optimize_for_solve(qc: QuantumCircuit) -> tuple[QuantumCircuit, str]:
    """Reduce ``qc`` with the peak-preserving optimize pipeline before solving.

    Args:
        qc (QuantumCircuit): Circuit loaded from disk.

    Returns:
        tuple[QuantumCircuit, str]: The reduced circuit and a note naming the steps that were
        accepted and the resulting outcome guarantee, e.g.
        ``"optimize[steps=strip+snap, guarantee=peak_verified]"``.
    """
    result = optimize_circuit(qc)
    steps = "+".join(result.steps) if result.steps else "none"
    return result.optimized, f"optimize[steps={steps}, guarantee={result.equivalence}]"


def _solve_circuit(
    qc: QuantumCircuit,
    solver: Solver,
    challenge: str,
    records: list[SubmissionRecord] | None,
    solve_kwargs: dict,
    optimize: bool,
) -> PeakResult:
    """Optionally optimize ``qc``, solve it, and fold any optimize note into the result.

    Args:
        qc (QuantumCircuit): Loaded circuit to solve.
        solver (Solver): Solver callable (e.g. ``solve_peak``).
        challenge (str): Bare challenge name.
        records (list[SubmissionRecord] | None): Existing submissions for known-failure avoidance.
        solve_kwargs (dict): Extra keyword arguments forwarded to ``solver``.
        optimize (bool): If ``True``, reduce ``qc`` with :func:`_optimize_for_solve` before solving.

    Returns:
        PeakResult: The solved result, with the optimize note prepended to ``notes`` when
        ``optimize`` is set.
    """
    note = ""
    if optimize:
        qc, note = _optimize_for_solve(qc)
    result = solver(qc, challenge=challenge, records=records, **solve_kwargs)
    if note:
        result = replace(result, notes=f"{note}; {result.notes}" if result.notes else note)
    return result


def _solve_worker(
    queue: "mp.Queue",
    solver: Solver,
    path: str,
    challenge: str,
    records: list[SubmissionRecord] | None,
    solve_kwargs: dict,
    optimize: bool,
) -> None:
    """Subprocess entry point: load a circuit, solve it, and post the result to ``queue``.

    Runs in a separate process so the parent can hard-kill it on a time budget (Aer's C++ MPS
    sampling does not respond to thread/signal interrupts). The result is sent as a plain dict.

    Args:
        queue (mp.Queue): Queue to post ``("ok", result_dict)`` or ``("err", repr)`` to.
        solver (Solver): Picklable solver callable (e.g. ``solve_peak``).
        path (str): Path to the QASM file.
        challenge (str): Bare challenge name.
        records (list[SubmissionRecord] | None): Existing submissions for known-failure avoidance.
        solve_kwargs (dict): Extra keyword arguments forwarded to ``solver``.
        optimize (bool): If ``True``, reduce the circuit with the optimize pipeline before solving.

    Returns:
        None
    """
    try:
        qc = load_circuit(path)
        result = _solve_circuit(qc, solver, challenge, records, solve_kwargs, optimize)
        queue.put(("ok", result.to_dict()))
    except Exception as exc:  # report any solver failure back to the parent process
        queue.put(("err", repr(exc)))


def _solve_with_budget(
    solver: Solver,
    path: str,
    challenge: str,
    records: list[SubmissionRecord] | None,
    solve_kwargs: dict,
    time_budget_s: float,
    optimize: bool,
) -> PeakResult:
    """Run ``solver`` on a circuit in a subprocess, killing it if it exceeds ``time_budget_s``.

    Args:
        solver (Solver): Picklable solver callable (e.g. ``solve_peak``).
        path (str): Path to the QASM file.
        challenge (str): Bare challenge name.
        records (list[SubmissionRecord] | None): Existing submissions for known-failure avoidance.
        solve_kwargs (dict): Extra keyword arguments forwarded to ``solver``.
        time_budget_s (float): Wall-clock seconds allowed before the subprocess is terminated.
        optimize (bool): If ``True``, reduce the circuit with the optimize pipeline before solving.

    Returns:
        PeakResult: The solved result.

    Raises:
        TimeoutError: If the solve does not finish within ``time_budget_s``.
        RuntimeError: If the subprocess reports an error or produces no result.
    """
    ctx = mp.get_context("spawn")  # cross-platform; avoids forking a process holding Aer/threads
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_solve_worker,
        args=(queue, solver, path, challenge, records, solve_kwargs, optimize),
        daemon=True,
    )
    proc.start()
    try:
        status, payload = queue.get(timeout=time_budget_s)
    except Empty:
        proc.terminate()
        proc.join()
        raise TimeoutError(f"solve exceeded {time_budget_s:.0f}s") from None
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join()
    if status == "err":
        raise RuntimeError(f"solve worker failed: {payload}")
    return PeakResult.from_dict(payload)


def solve_job(
    job: CircuitJob,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    resume: bool = True,
    solver: Solver = solve_peak,
    records: list[SubmissionRecord] | None = None,
    time_budget_s: float | None = None,
    optimize: bool = False,
    **solve_kwargs: object,
) -> PeakResult:
    """Solve one circuit, loading from / writing to its JSON checkpoint.

    Args:
        job (CircuitJob): The circuit to solve.
        results_dir (Path | str): Root results directory holding the cache.
        resume (bool): If ``True`` and a checkpoint exists, return it without recomputing.
        solver (Solver): Callable ``(qc, **kwargs) -> PeakResult``; defaults to
            :func:`~quantum_hack.peak.solve.solve_peak`. Must be picklable when ``time_budget_s``
            is set (it runs in a subprocess); the default is.
        records (list[SubmissionRecord] | None): Existing submissions, forwarded to the solver so
            it avoids already-rejected bitstrings.
        time_budget_s (float | None): Per-circuit wall-clock budget in seconds. ``None`` (default)
            runs the solver inline with no limit; a number runs it in a subprocess and terminates
            it (raising :class:`TimeoutError`) if it overruns.
        optimize (bool): If ``True``, reduce the circuit with the peak-preserving optimize pipeline
            (:func:`~quantum_hack.optimize.optimize_circuit`) before solving and record the applied
            steps in the result's notes. ``False`` (default) solves the circuit as loaded.
        **solve_kwargs (object): Extra keyword arguments forwarded to ``solver``.

    Returns:
        PeakResult: The solved (or cached) result.

    Raises:
        TimeoutError: If ``time_budget_s`` is set and the solve overruns it.
    """
    results_dir = Path(results_dir)
    cache = _cache_path(results_dir, job)
    if resume and cache.is_file():
        return PeakResult.from_dict(json.loads(cache.read_text(encoding="utf-8")))

    if time_budget_s is None:
        qc = load_circuit(job.path)
        result = _solve_circuit(qc, solver, job.name, records, solve_kwargs, optimize)
    else:
        result = _solve_with_budget(
            solver, str(job.path), job.name, records, solve_kwargs, time_budget_s, optimize
        )
    _write_json_atomic(cache, result.to_dict())
    return result


def run_batch(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    difficulty: str | None = None,
    challenge: str | None = None,
    shard: tuple[int, int] | None = None,
    resume: bool = True,
    solver: Solver = solve_peak,
    time_budget_s: float | None = None,
    optimize: bool = False,
    on_solved: Callable[["CircuitJob", PeakResult, bool], None] | None = None,
    **solve_kwargs: object,
) -> list[PeakResult]:
    """Solve a (optionally sharded) set of challenge circuits, caching each result.

    Existing submissions are loaded once and passed to the solver so it never re-proposes a
    rejected bitstring. A circuit that exceeds ``time_budget_s`` is killed and recorded as a
    ``"timeout"`` result (empty bitstring, not cached) so the run continues; other per-circuit
    exceptions are not caught.

    Args:
        data_dir (Path | str): Root directory of challenge QASM files.
        results_dir (Path | str): Root results directory for the cache and CSVs.
        difficulty (str | None): Restrict to one tier, or ``None`` for all.
        challenge (str | None): Solve only this one challenge by name (e.g. ``"challenge-64_25"``,
            with or without a tier prefix or ``.qasm`` suffix), searching all tiers and ignoring
            ``difficulty``; ``None`` solves the whole (filtered) set.
        shard (tuple[int, int] | None): ``(index, total)`` to process only every ``total``-th job
            starting at ``index`` (maps to a SLURM ``--array`` task); ``None`` processes all.
        resume (bool): Skip circuits with an existing checkpoint.
        solver (Solver): Callable ``(qc, **kwargs) -> PeakResult``.
        time_budget_s (float | None): Per-circuit wall-clock budget in seconds; ``None`` (default)
            runs inline with no limit. A number runs each solve in a subprocess and terminates it
            on overrun, recording a ``"timeout"`` result instead.
        optimize (bool): If ``True``, reduce each circuit with the peak-preserving optimize pipeline
            (strip -> snap -> transpile) before solving and record the applied steps in each
            result's notes. ``False`` (default) solves circuits as loaded. Note that resume reuses a
            cached result regardless of this flag, so re-run with ``resume=False`` to apply it to
            already-cached circuits.
        on_solved (Callable[[CircuitJob, PeakResult, bool], None] | None): Optional callback
            invoked after each circuit with ``(job, result, from_cache)`` so callers can stream
            progress; ``None`` disables it.
        **solve_kwargs (object): Extra keyword arguments forwarded to ``solver`` (e.g.
            ``gpu_available``, ``device``, ``shots``, ``bond_dim``).

    Returns:
        list[PeakResult]: Results for the circuits this call processed, in job order. A timed-out
        circuit yields a ``PeakResult`` with ``method == "timeout"`` and an empty bitstring.

    Raises:
        ValueError: If ``shard`` is given with a non-positive total or an out-of-range index.
        FileNotFoundError: If ``challenge`` is given but matches no circuit.
    """
    results_dir = Path(results_dir)
    # A single named challenge ignores the tier filter (search everywhere).
    jobs = list_jobs(data_dir, None if challenge is not None else difficulty)
    if challenge is not None:
        target = challenge.replace("\\", "/").split("/")[-1].removesuffix(".qasm")
        jobs = [job for job in jobs if job.name == target]
        if not jobs:
            raise FileNotFoundError(f"no challenge named {challenge!r} under {data_dir}")
    if shard is not None:
        index, total = shard
        if total <= 0 or not (0 <= index < total):
            raise ValueError(f"invalid shard {shard!r}; need 0 <= index < total and total > 0")
        jobs = jobs[index::total]

    records = load_results(results_dir) if Path(results_dir).exists() else []
    results: list[PeakResult] = []
    for job in jobs:
        from_cache = resume and _cache_path(results_dir, job).is_file()
        try:
            result = solve_job(
                job,
                results_dir=results_dir,
                resume=resume,
                solver=solver,
                records=records,
                time_budget_s=time_budget_s,
                optimize=optimize,
                **solve_kwargs,
            )
        except TimeoutError as exc:
            # Killed for overrunning the budget: record a sentinel (not cached, so it is retried
            # next run, e.g. on a GPU) and keep going.
            from_cache = False
            result = PeakResult(
                bitstring="",
                probability=0.0,
                method="timeout",
                num_qubits=job.num_qubits,
                confidence=0.0,
                notes=str(exc),
            )
        if on_solved is not None:
            on_solved(job, result, from_cache)
        results.append(result)
    return results


def _load_cached_results(results_dir: Path, difficulty: str) -> list[tuple[str, PeakResult]]:
    """Load every cached ``PeakResult`` for a difficulty tier.

    Args:
        results_dir (Path): Root results directory.
        difficulty (str): Difficulty tier.

    Returns:
        list[tuple[str, PeakResult]]: ``(challenge_name, result)`` pairs sorted by name.
    """
    cache_dir = results_dir / CACHE_DIRNAME / difficulty
    if not cache_dir.is_dir():
        return []
    pairs = [
        (path.stem, PeakResult.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        for path in sorted(cache_dir.glob("*.json"))
    ]
    return pairs


def write_results_csvs(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> list[Path]:
    """Append pending candidate rows to the per-difficulty result CSVs from the JSON cache.

    For each cached result: skip the challenge if it is already solved (a ``success`` row exists)
    or if that exact bitstring was already recorded; otherwise append a ``pending`` row. Existing
    rows (including human ``success``/``failed`` marks) are never modified. New files get a header.

    Args:
        results_dir (Path | str): Root results directory holding the cache and CSVs.

    Returns:
        list[Path]: The CSV paths that were created or appended to.
    """
    results_dir = Path(results_dir)
    existing = load_results(results_dir)
    touched: list[Path] = []
    for difficulty in DIFFICULTIES:
        cached = _load_cached_results(results_dir, difficulty)
        if not cached:
            continue
        csv_path = results_dir / f"{difficulty}_bitstrings.csv"
        new_rows: list[tuple[str, int, str, str, float]] = []
        for name, result in cached:
            status = challenge_status(name, records=existing)
            if status.worked:
                continue  # already solved — leave it alone
            if result.bitstring in {rec.bitstring for rec in status.records}:
                continue  # this candidate is already recorded
            new_rows.append(
                (name, result.num_qubits, result.method, result.bitstring, result.probability)
            )
        if not new_rows:
            continue
        write_header = not csv_path.exists()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(CSV_COLUMNS)
            for name, qubits, method, bitstring, probability in new_rows:
                writer.writerow([name, qubits, method, bitstring, f"{probability:.4g}", "pending"])
        touched.append(csv_path)
    return touched


@dataclass(frozen=True)
class CacheComparison:
    """How a cached solver answer compares to the recorded CSV submissions for a challenge.

    Attributes:
        challenge (str): Bare challenge name.
        difficulty (str): Difficulty tier.
        cached_bitstring (str): The bitstring the solver cached for this challenge.
        cached_method (str): The method that produced the cached bitstring.
        cached_confidence (float): The solver's confidence in the cached bitstring.
        verdict (str): One of :data:`CACHE_MATCH`, :data:`CACHE_MISMATCH`,
            :data:`CACHE_KNOWN_FAILURE`, :data:`CACHE_UNCONFIRMED`.
        accepted_bitstring (str | None): The portal-accepted (``success``) bitstring on record,
            or ``None`` if the challenge has no accepted answer yet.
        detail (str): A human-readable explanation of the verdict.
    """

    challenge: str
    difficulty: str
    cached_bitstring: str
    cached_method: str
    cached_confidence: float
    verdict: str
    accepted_bitstring: str | None
    detail: str

    @property
    def failed(self) -> bool:
        """Whether the cached answer provably disagrees with the known-correct answer.

        Returns:
            bool: ``True`` if the verdict is a mismatch or a known failure.
        """
        return self.verdict in CACHE_FAILED_VERDICTS


def _accepted_bitstring(status: ChallengeStatus) -> str | None:
    """Return the highest-probability accepted bitstring for a challenge, if any.

    Args:
        status (ChallengeStatus): Aggregated submissions for the challenge.

    Returns:
        str | None: The accepted (``success``) bitstring, or ``None`` if none is recorded.
    """
    if not status.successes:
        return None
    return max(status.successes, key=lambda r: r.probability or 0.0).bitstring


def compare_cache_to_results(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> list[CacheComparison]:
    """Compare every cached solver answer to the recorded CSV submissions.

    For each ``results/_peak_cache/<difficulty>/<challenge>.json`` the cached bitstring is
    checked against the challenge's CSV rows: it *matches* the accepted (``success``) bitstring,
    *mismatches* it, equals a *known-failed* bitstring, or is *unconfirmed* (no accepted answer on
    record to compare against).

    Args:
        results_dir (Path | str): Root results directory holding the cache and CSVs.

    Returns:
        list[CacheComparison]: One entry per cached challenge, ordered by difficulty then name.
    """
    results_dir = Path(results_dir)
    records = load_results(results_dir)
    comparisons: list[CacheComparison] = []
    for difficulty in DIFFICULTIES:
        for name, result in _load_cached_results(results_dir, difficulty):
            status = challenge_status(name, records=records)
            accepted = _accepted_bitstring(status)
            if accepted is not None:
                if result.bitstring == accepted:
                    verdict, detail = CACHE_MATCH, f"matches accepted {accepted}"
                else:
                    verdict = CACHE_MISMATCH
                    detail = f"cache {result.bitstring} != accepted {accepted}"
            elif result.bitstring and result.bitstring in status.failed_bitstrings:
                verdict = CACHE_KNOWN_FAILURE
                detail = f"{result.bitstring} was already submitted and rejected"
            else:
                verdict, detail = CACHE_UNCONFIRMED, "no accepted bitstring on record to compare"
            comparisons.append(
                CacheComparison(
                    challenge=name,
                    difficulty=difficulty,
                    cached_bitstring=result.bitstring,
                    cached_method=result.method,
                    cached_confidence=result.confidence,
                    verdict=verdict,
                    accepted_bitstring=accepted,
                    detail=detail,
                )
            )
    return comparisons


def failed_challenges(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> list[CacheComparison]:
    """Return the cached challenges whose answer provably failed (mismatch or known failure).

    Args:
        results_dir (Path | str): Root results directory holding the cache and CSVs.

    Returns:
        list[CacheComparison]: Comparisons whose verdict is a mismatch or a known failure.
    """
    return [c for c in compare_cache_to_results(results_dir) if c.failed]


def _parse_shard(text: str) -> tuple[int, int]:
    """Parse a ``"i/N"`` shard argument.

    Args:
        text (str): Shard spec like ``"0/4"``.

    Returns:
        tuple[int, int]: ``(index, total)``.

    Raises:
        argparse.ArgumentTypeError: If ``text`` is not ``int/int``.
    """
    try:
        index, total = (int(part) for part in text.split("/", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"shard must be 'i/N', got {text!r}") from exc
    return index, total


def build_parser() -> argparse.ArgumentParser:
    """Build the ``quantum-hack-batch`` argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Batch-solve peaked-circuit challenges.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default=None)
    parser.add_argument(
        "--challenge",
        default=None,
        help="Solve one challenge by name (e.g. challenge-64_25); ignores --difficulty/--shard.",
    )
    parser.add_argument("--shard", type=_parse_shard, default=None, help="SLURM array shard 'i/N'.")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--gpu", action="store_true", help="Use GPU Aer (LUMI standard-g).")
    parser.add_argument("--device", default="CPU", help="Aer device when --gpu is set.")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--bond-dim", type=int, default=128)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=DEFAULT_TIME_BUDGET_S,
        help=(
            f"Per-circuit wall-clock budget in seconds (default {DEFAULT_TIME_BUDGET_S:.0f}); a "
            "circuit exceeding it is killed and recorded as a timeout. Use 0 for no limit."
        ),
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help=(
            "Before solving, reduce each circuit with the peak-preserving optimize pipeline "
            "(strip -> snap -> transpile). Off by default; the lossy 'snap' step is kept only when "
            "the exact peak is verified unchanged. The applied steps are reported per circuit."
        ),
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="After solving, append pending rows to the per-difficulty CSVs from the cache.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip solving; only append pending rows to the CSVs from the existing JSON cache.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Skip solving; compare cached answers to the recorded CSVs and list which failed.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the batch solver from the command line.

    Args:
        argv (list[str] | None): Argument vector; ``None`` uses ``sys.argv``.

    Returns:
        None
    """
    args = build_parser().parse_args(argv)

    if args.aggregate_only:
        for path in write_results_csvs(args.results_dir):
            print(f"Updated {path}")
        return

    if args.compare:
        comparisons = compare_cache_to_results(args.results_dir)
        for c in comparisons:
            print(
                f"[{c.verdict}] {c.difficulty}/{c.challenge}: "
                f"cache={c.cached_bitstring or '-'} ({c.cached_method}, "
                f"conf={c.cached_confidence:.2f}) -- {c.detail}",
                flush=True,
            )
        failed = [c for c in comparisons if c.failed]
        print(f"\n{len(failed)} failed / {len(comparisons)} cached challenge(s).")
        for c in failed:
            print(f"  FAILED {c.difficulty}/{c.challenge}: {c.detail}")
        return

    if args.optimize:
        pipeline_steps = " -> ".join(name for name, _, _ in default_pipeline())
        print(
            f"Optimization enabled; reducing each circuit before solving with: {pipeline_steps} "
            "(peak-preserving -- the lossy 'snap' step is kept only when the exact peak is "
            "verified unchanged, and is skipped on circuits too wide to verify).",
            flush=True,
        )

    def _report(job: CircuitJob, result: PeakResult, from_cache: bool) -> None:
        if result.method == "timeout":
            print(f"[timeout] {job.difficulty}/{job.name} (n={job.num_qubits}): {result.notes}",
                  flush=True)
            return
        tag = "cached" if from_cache else "solved"
        flag = "" if result.is_local_max in (True, None) else " [NOT local max]"
        opt = result.notes.split("; ", 1)[0]
        opt_tag = f"  {opt}" if opt.startswith("optimize[") else ""
        print(
            f"[{tag}] {job.difficulty}/{job.name} (n={result.num_qubits}): "
            f"{result.bitstring}  p={result.probability:.3f}  conf={result.confidence:.2f}  "
            f"{result.method}{flag}{opt_tag}",
            flush=True,
        )

    run_batch(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        difficulty=args.difficulty,
        challenge=args.challenge,
        shard=args.shard,
        resume=args.resume,
        time_budget_s=args.time_budget or None,  # --time-budget 0 means no limit
        optimize=args.optimize,
        on_solved=_report,
        gpu_available=args.gpu,
        device=args.device,
        shots=args.shots,
        bond_dim=args.bond_dim,
    )
    if args.write_csv:
        for path in write_results_csvs(args.results_dir):
            print(f"Updated {path}")


if __name__ == "__main__":
    main()
