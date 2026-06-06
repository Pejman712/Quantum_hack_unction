import csv

import pytest
from qiskit import QuantumCircuit, qasm2

from quantum_hack.batch import (
    compare_cache_to_results,
    failed_challenges,
    list_jobs,
    run_batch,
    solve_job,
    write_results_csvs,
)
from quantum_hack.peak.result import PeakResult


def _write_qasm(path, num_qubits):
    """Write a tiny deterministic QASM circuit to ``path``.

    Args:
        path (Path): Destination ``.qasm`` path.
        num_qubits (int): Number of qubits; qubit 0 is flipped to |1>.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    path.write_text(qasm2.dumps(qc), encoding="utf-8")


def _make_dataset(tmp_path):
    """Create a small qasm_data tree with three circuits across two tiers.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        Path: The data directory containing the circuits.
    """
    data = tmp_path / "qasm"
    _write_qasm(data / "very_easy" / "challenge-2_1.qasm", 2)
    _write_qasm(data / "very_easy" / "challenge-4_2.qasm", 4)
    _write_qasm(data / "easy" / "challenge-3_3.qasm", 3)
    return data


class _CountingSolver:
    """A stub solver that records how many circuits it was asked to solve."""

    def __init__(self):
        self.calls = 0

    def __call__(self, qc, *, challenge=None, records=None, **kwargs):
        self.calls += 1
        bits = "0" * qc.num_qubits
        return PeakResult(
            bitstring=bits,
            probability=0.5,
            method="stub",
            num_qubits=qc.num_qubits,
            confidence=0.5,
            is_local_max=True,
        )


def test_list_jobs_sorted_by_qubits(tmp_path):
    data = _make_dataset(tmp_path)
    jobs = list_jobs(data)
    assert [j.num_qubits for j in jobs] == [2, 3, 4]
    assert jobs[0].name == "challenge-2_1"
    assert jobs[0].difficulty == "very_easy"


def test_shards_partition_jobs(tmp_path):
    data = _make_dataset(tmp_path)
    jobs = list_jobs(data)
    s0, s1 = jobs[0::2], jobs[1::2]
    # Shards cover everything and are disjoint.
    assert {j.name for j in s0} | {j.name for j in s1} == {j.name for j in jobs}
    assert not ({j.name for j in s0} & {j.name for j in s1})
    r0 = run_batch(
        data_dir=data, results_dir=tmp_path / "r", shard=(0, 2), solver=_CountingSolver()
    )
    r1 = run_batch(
        data_dir=data, results_dir=tmp_path / "r", shard=(1, 2), solver=_CountingSolver()
    )
    assert len(r0) == len(s0)
    assert len(r1) == len(s1)


def test_run_batch_single_challenge(tmp_path):
    data = _make_dataset(tmp_path)
    results = run_batch(
        data_dir=data,
        results_dir=tmp_path / "r",
        challenge="challenge-4_2",
        solver=_CountingSolver(),
    )
    assert len(results) == 1
    assert results[0].num_qubits == 4


def test_run_batch_single_challenge_accepts_tier_prefix(tmp_path):
    data = _make_dataset(tmp_path)
    results = run_batch(
        data_dir=data,
        results_dir=tmp_path / "r",
        challenge="easy/challenge-3_3.qasm",
        solver=_CountingSolver(),
    )
    assert len(results) == 1
    assert results[0].num_qubits == 3


def test_run_batch_unknown_challenge_raises(tmp_path):
    data = _make_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_batch(
            data_dir=data,
            results_dir=tmp_path / "r",
            challenge="challenge-99_9",
            solver=_CountingSolver(),
        )


def test_invalid_shard_raises(tmp_path):
    data = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match="invalid shard"):
        run_batch(data_dir=data, results_dir=tmp_path / "r", shard=(5, 2))


def test_resume_skips_cached_circuits(tmp_path):
    data = _make_dataset(tmp_path)
    results = tmp_path / "r"
    solver = _CountingSolver()
    run_batch(data_dir=data, results_dir=results, solver=solver)
    assert solver.calls == 3
    # Second run with resume should hit the cache and not call the solver again.
    run_batch(data_dir=data, results_dir=results, solver=solver)
    assert solver.calls == 3


def test_no_resume_recomputes(tmp_path):
    data = _make_dataset(tmp_path)
    results = tmp_path / "r"
    solver = _CountingSolver()
    run_batch(data_dir=data, results_dir=results, solver=solver)
    run_batch(data_dir=data, results_dir=results, solver=solver, resume=False)
    assert solver.calls == 6


def test_write_csv_appends_pending_rows(tmp_path):
    data = _make_dataset(tmp_path)
    results = tmp_path / "r"
    run_batch(data_dir=data, results_dir=results, solver=_CountingSolver())
    touched = write_results_csvs(results)

    very_easy = results / "very_easy_bitstrings.csv"
    assert very_easy in touched
    rows = list(csv.DictReader(very_easy.open(encoding="utf-8")))
    assert {r["challenge"] for r in rows} == {"challenge-2_1", "challenge-4_2"}
    assert all(r["status"] == "pending" for r in rows)
    assert rows[0]["bitstring"] == "00"  # 2-qubit stub


def test_write_csv_skips_already_solved_challenge(tmp_path):
    data = _make_dataset(tmp_path)
    results = tmp_path / "r"
    results.mkdir()
    # Pre-mark challenge-2_1 as solved with a different bitstring.
    solved = results / "very_easy_bitstrings.csv"
    solved.write_text(
        "challenge,qubits,method,bitstring,probability,status\n"
        "challenge-2_1,2,statevector,11,0.9,success\n",
        encoding="utf-8",
    )
    run_batch(data_dir=data, results_dir=results, solver=_CountingSolver())
    write_results_csvs(results)

    rows = list(csv.DictReader(solved.open(encoding="utf-8")))
    challenges = [r["challenge"] for r in rows]
    # The solved challenge keeps only its success row; the unsolved one gets a pending row.
    assert challenges.count("challenge-2_1") == 1
    assert "challenge-4_2" in challenges


def test_run_batch_records_timeout_sentinel(monkeypatch, tmp_path):
    # Deterministically exercise the timeout handling without spawning real subprocesses.
    import quantum_hack.batch as batch_mod

    data = _make_dataset(tmp_path)

    def _always_timeout(job, **kwargs):
        raise TimeoutError("solve exceeded 5s")

    monkeypatch.setattr(batch_mod, "solve_job", _always_timeout)
    results = batch_mod.run_batch(data_dir=data, results_dir=tmp_path / "r", time_budget_s=5)
    assert results
    assert all(r.method == "timeout" and r.bitstring == "" for r in results)
    # Timed-out circuits are not cached, so they will be retried next run.
    assert not list((tmp_path / "r").rglob("*.json"))


def test_solve_job_times_out_with_tiny_budget(tmp_path):
    # A microscopic budget kills the subprocess before it can finish (real spawn).
    data = _make_dataset(tmp_path)
    job = list_jobs(data)[0]
    with pytest.raises(TimeoutError):
        solve_job(job, results_dir=tmp_path / "r", time_budget_s=0.001)
    assert not list((tmp_path / "r").rglob("*.json"))


def test_solve_job_succeeds_within_budget(tmp_path):
    # Generous budget: the real subprocess solves the 2-qubit circuit and returns a PeakResult.
    data = _make_dataset(tmp_path)
    job = list_jobs(data)[0]  # 2 qubits
    result = solve_job(job, results_dir=tmp_path / "r", time_budget_s=120)
    assert result.method != "timeout"
    assert len(result.bitstring) == 2


def test_main_aggregate_only_writes_csv_without_solving(monkeypatch, tmp_path):
    import quantum_hack.batch as batch_mod

    data = _make_dataset(tmp_path)
    results = tmp_path / "r"
    # Seed the cache with a real run, then aggregate-only must not call run_batch again.
    run_batch(data_dir=data, results_dir=results, solver=_CountingSolver())
    called = {"run": False}

    def _flag_run(*args, **kwargs):
        called["run"] = True
        return []

    monkeypatch.setattr(batch_mod, "run_batch", _flag_run)
    batch_mod.main(["--results-dir", str(results), "--aggregate-only"])
    assert called["run"] is False
    assert (results / "very_easy_bitstrings.csv").exists()


def test_compare_cache_to_results_classifies_each_case(tmp_path):
    import json as _json

    results = tmp_path / "r"
    cache_dir = results / "_peak_cache" / "easy"
    cache_dir.mkdir(parents=True)

    def _cache(name, bitstring):
        result = PeakResult(
            bitstring=bitstring, probability=0.5, method="MPS", num_qubits=len(bitstring)
        )
        (cache_dir / f"{name}.json").write_text(_json.dumps(result.to_dict()), encoding="utf-8")

    _cache("challenge-2_1", "01")  # equals the accepted bitstring -> match
    _cache("challenge-2_2", "11")  # accepted is "00" -> mismatch
    _cache("challenge-2_3", "10")  # equals a known-failed bitstring -> known_failure
    _cache("challenge-2_4", "01")  # no CSV row at all -> unconfirmed

    (results / "easy_bitstrings.csv").write_text(
        "challenge,qubits,method,bitstring,probability,status\n"
        "challenge-2_1,2,MPS,01,0.9,success\n"
        "challenge-2_2,2,MPS,00,0.9,success\n"
        "challenge-2_3,2,MPS,10,0.1,failed\n",
        encoding="utf-8",
    )

    by_name = {c.challenge: c for c in compare_cache_to_results(results)}
    assert by_name["challenge-2_1"].verdict == "match"
    assert by_name["challenge-2_2"].verdict == "mismatch"
    assert by_name["challenge-2_2"].accepted_bitstring == "00"
    assert by_name["challenge-2_3"].verdict == "known_failure"
    assert by_name["challenge-2_4"].verdict == "unconfirmed"

    failed = {c.challenge for c in failed_challenges(results)}
    assert failed == {"challenge-2_2", "challenge-2_3"}


def test_solve_job_uses_cache_on_resume(tmp_path):
    data = _make_dataset(tmp_path)
    results = tmp_path / "r"
    job = list_jobs(data)[0]
    solver = _CountingSolver()
    first = solve_job(job, results_dir=results, solver=solver)
    second = solve_job(job, results_dir=results, solver=solver)
    assert solver.calls == 1
    assert first == second
