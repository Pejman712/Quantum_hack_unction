from quantum_hack.results import (
    ChallengeStatus,
    SubmissionRecord,
    challenge_status,
    is_known_failure,
    load_results,
)

HEADER = "challenge,qubits,method,bitstring,probability,status\n"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "".join(rows))


def _results_dir(tmp_path):
    _write_csv(
        tmp_path / "very_easy_bitstrings.csv",
        ["challenge-8_1,8,statevector,10101101,0.913,success\n"],
    )
    _write_csv(
        tmp_path / "easy_bitstrings.csv",
        [
            "challenge-36_15,36,MPS,1101,0.001,failed\n",
            "challenge-36_15,36,MPS,1010,0.002,failed\n",
            "challenge-36_15,36,MPS,1111,0.250,success\n",
            "challenge-40_16,40,MPS,0000,0.300,pending\n",
            "challenge-64_26,64,MPS,0110,0.071,failed\n",
            "challenge-64_26,64,MPS,1001,0.060,failed\n",
        ],
    )
    return tmp_path


def test_load_results_parses_rows_and_derives_difficulty(tmp_path):
    data_dir = _results_dir(tmp_path)
    records = load_results(data_dir)
    assert all(isinstance(r, SubmissionRecord) for r in records)
    # 1 very_easy + 6 easy rows.
    assert len(records) == 7
    very_easy = next(r for r in records if r.challenge == "challenge-8_1")
    assert very_easy.difficulty == "very_easy"
    assert very_easy.qubits == 8
    assert very_easy.probability == 0.913
    assert very_easy.status == "success"


def test_load_results_sorted_by_filename(tmp_path):
    data_dir = _results_dir(tmp_path)
    difficulties = [r.difficulty for r in load_results(data_dir)]
    # "easy" sorts before "very_easy".
    assert difficulties == ["easy"] * 6 + ["very_easy"]


def test_blank_probability_and_status_default(tmp_path):
    _write_csv(tmp_path / "moderate_bitstrings.csv", ["challenge-1_1,4,MPS,0000,,\n"])
    (record,) = load_results(tmp_path)
    assert record.probability is None
    assert record.status == "pending"


def test_challenge_status_success(tmp_path):
    data_dir = _results_dir(tmp_path)
    status = challenge_status("challenge-8_1", results_dir=data_dir)
    assert status.worked is True
    assert status.failed_bitstrings == []
    assert "solved" in status.summary
    assert "10101101" in status.summary


def test_challenge_status_collects_all_failed_bitstrings(tmp_path):
    data_dir = _results_dir(tmp_path)
    status = challenge_status("challenge-64_26", results_dir=data_dir)
    assert status.worked is False
    # Both failed bitstrings must be considered, not just the first.
    assert status.failed_bitstrings == ["0110", "1001"]
    assert "0110" in status.summary and "1001" in status.summary
    assert "different bitstring" in status.summary


def test_challenge_status_success_despite_earlier_failures(tmp_path):
    data_dir = _results_dir(tmp_path)
    status = challenge_status("challenge-36_15", results_dir=data_dir)
    # Two failures then a success -> solved, but failures are still surfaced to avoid.
    assert status.worked is True
    assert status.failed_bitstrings == ["1101", "1010"]
    assert "1111" in status.summary
    assert "to avoid" in status.summary


def test_challenge_status_pending_only(tmp_path):
    data_dir = _results_dir(tmp_path)
    status = challenge_status("challenge-40_16", results_dir=data_dir)
    assert status.worked is None
    assert status.failed_bitstrings == []
    assert "pending" in status.summary


def test_challenge_status_unknown_challenge(tmp_path):
    data_dir = _results_dir(tmp_path)
    status = challenge_status("challenge-99_99", results_dir=data_dir)
    assert isinstance(status, ChallengeStatus)
    assert status.records == ()
    assert status.worked is None
    assert "untried" in status.summary


def test_challenge_status_normalizes_name(tmp_path):
    data_dir = _results_dir(tmp_path)
    for ref in ("challenge-8_1", "very_easy/challenge-8_1", "very_easy/challenge-8_1.qasm"):
        assert challenge_status(ref, results_dir=data_dir).worked is True


def test_challenge_status_accepts_preloaded_records():
    records = [
        SubmissionRecord("c-1", 4, "MPS", "0000", 0.1, "failed", "easy"),
        SubmissionRecord("c-1", 4, "MPS", "1111", 0.2, "failed", "easy"),
    ]
    status = challenge_status("c-1", records=records)
    assert status.worked is False
    assert status.failed_bitstrings == ["0000", "1111"]


def test_is_known_failure(tmp_path):
    data_dir = _results_dir(tmp_path)
    assert is_known_failure("challenge-64_26", "0110", results_dir=data_dir) is True
    assert is_known_failure("challenge-64_26", "1001", results_dir=data_dir) is True
    assert is_known_failure("challenge-64_26", "1111", results_dir=data_dir) is False
    # A successful bitstring is not a known failure.
    assert is_known_failure("challenge-8_1", "10101101", results_dir=data_dir) is False


def test_challenge_status_str_matches_summary(tmp_path):
    data_dir = _results_dir(tmp_path)
    status = challenge_status("challenge-8_1", results_dir=data_dir)
    assert str(status) == status.summary
