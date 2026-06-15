"""
Sprint 0 — data pipeline tests.

Run from repo root:
    pytest exp/transformer-nqs/tests/test_data.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# make the transformer-nqs package importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.tokenizer import bitstring_to_tokens, tokens_to_bitstring
from data.circuit_encoder import (
    encode_circuit,
    parse_qasm,
    build_interaction_matrix,
    GATE_VOCAB,
    GATE_VOCAB_SIZE,
)
from data.dataset import load_dataset, CircuitSample, DIFFICULTIES

REPO_ROOT = Path(__file__).resolve().parents[3]
QASM_8Q   = REPO_ROOT / "qasm_data" / "very_easy" / "challenge-8_1.qasm"
QASM_48Q  = REPO_ROOT / "qasm_data" / "very_hard" / "challenge-48_42.qasm"


# ── tokenizer ────────────────────────────────────────────────────────────────

def test_bitstring_to_tokens_basic():
    assert bitstring_to_tokens("01101") == [0, 1, 1, 0, 1]

def test_bitstring_to_tokens_all_zeros():
    assert bitstring_to_tokens("0000") == [0, 0, 0, 0]

def test_bitstring_to_tokens_all_ones():
    assert bitstring_to_tokens("1111") == [1, 1, 1, 1]

def test_tokens_to_bitstring_basic():
    assert tokens_to_bitstring([0, 1, 1, 0, 1]) == "01101"

def test_roundtrip_short():
    bs = "10110010"
    assert tokens_to_bitstring(bitstring_to_tokens(bs)) == bs

def test_roundtrip_long():
    bs = "0011010010110001110010111001100100101100010111110110010100101011"
    assert tokens_to_bitstring(bitstring_to_tokens(bs)) == bs

def test_token_values_are_binary():
    tokens = bitstring_to_tokens("10101010")
    assert all(t in (0, 1) for t in tokens)


# ── circuit_encoder: parse_qasm ───────────────────────────────────────────────

def test_parse_qasm_n_qubits_8():
    n, _ = parse_qasm(QASM_8Q)
    assert n == 8

def test_parse_qasm_n_qubits_48():
    n, _ = parse_qasm(QASM_48Q)
    assert n == 48

def test_parse_qasm_returns_gates():
    _, gates = parse_qasm(QASM_8Q)
    assert len(gates) > 0

def test_parse_qasm_all_gate_types_known():
    _, gates = parse_qasm(QASM_48Q)
    for gate in gates:
        assert gate["type"] in GATE_VOCAB, f"unknown gate type: {gate['type']}"

def test_parse_qasm_qubit_indices_in_range():
    n, gates = parse_qasm(QASM_48Q)
    for gate in gates:
        for q in gate["qubits"]:
            assert 0 <= q < n, f"qubit {q} out of range for {n}-qubit circuit"

def test_parse_qasm_angles_are_finite():
    _, gates = parse_qasm(QASM_8Q)
    for gate in gates:
        assert math.isfinite(gate["angle"]), f"non-finite angle: {gate['angle']}"

def test_parse_qasm_1q_gates_have_one_qubit():
    _, gates = parse_qasm(QASM_8Q)
    for gate in gates:
        if gate["type"] in ("rx", "rz"):
            assert len(gate["qubits"]) == 1

def test_parse_qasm_2q_gates_have_two_qubits():
    _, gates = parse_qasm(QASM_8Q)
    for gate in gates:
        if gate["type"] in ("cx", "swap"):
            assert len(gate["qubits"]) == 2

def test_parse_qasm_pi_angle_parsed_correctly():
    _, gates = parse_qasm(QASM_8Q)
    pi_gates = [g for g in gates if abs(abs(g["angle"]) - math.pi) < 1e-4]
    assert len(pi_gates) > 0, "expected at least one rx(pi) gate in challenge-8_1"


# ── circuit_encoder: interaction matrix ──────────────────────────────────────

def test_interaction_matrix_shape_8q():
    n, gates = parse_qasm(QASM_8Q)
    m = build_interaction_matrix(n, gates)
    assert m.shape == (8, 8)

def test_interaction_matrix_shape_48q():
    n, gates = parse_qasm(QASM_48Q)
    m = build_interaction_matrix(n, gates)
    assert m.shape == (48, 48)

def test_interaction_matrix_is_symmetric():
    n, gates = parse_qasm(QASM_48Q)
    m = build_interaction_matrix(n, gates)
    np.testing.assert_array_equal(m, m.T)

def test_interaction_matrix_diagonal_is_zero():
    n, gates = parse_qasm(QASM_8Q)
    m = build_interaction_matrix(n, gates)
    np.testing.assert_array_equal(np.diag(m), np.zeros(n))

def test_interaction_matrix_nonnegative():
    n, gates = parse_qasm(QASM_48Q)
    m = build_interaction_matrix(n, gates)
    assert (m >= 0).all()

def test_interaction_matrix_counts_2q_gates():
    n, gates = parse_qasm(QASM_8Q)
    m = build_interaction_matrix(n, gates)
    two_q_total = sum(1 for g in gates if len(g["qubits"]) == 2)
    # each 2q gate contributes 2 entries (i,j) and (j,i), so sum/2 == count
    assert int(m.sum()) // 2 == two_q_total


# ── circuit_encoder: encode_circuit ──────────────────────────────────────────

def test_encode_circuit_returns_three_values():
    result = encode_circuit(QASM_8Q)
    assert len(result) == 3

def test_encode_circuit_n_qubits():
    n, _, _ = encode_circuit(QASM_8Q)
    assert n == 8

def test_encode_circuit_gate_sequence_nonempty():
    _, gate_seq, _ = encode_circuit(QASM_8Q)
    assert len(gate_seq) > 0

def test_encode_circuit_gate_sequence_columns():
    _, gate_seq, _ = encode_circuit(QASM_8Q)
    # each row: [type_id, cos(angle), sin(angle), q0, q1]
    for row in gate_seq:
        assert len(row) == 5

def test_encode_circuit_type_ids_in_vocab():
    _, gate_seq, _ = encode_circuit(QASM_48Q)
    valid_ids = set(range(GATE_VOCAB_SIZE))
    for row in gate_seq:
        assert row[0] in valid_ids

def test_encode_circuit_cos_sin_unit():
    _, gate_seq, _ = encode_circuit(QASM_8Q)
    for row in gate_seq:
        cos_val, sin_val = row[1], row[2]
        norm = cos_val ** 2 + sin_val ** 2
        assert abs(norm - 1.0) < 1e-5, f"cos²+sin²={norm} not 1"

def test_encode_circuit_padding_index_for_1q_gates():
    n, gate_seq, _ = encode_circuit(QASM_8Q)
    for row in gate_seq:
        type_id, _, _, q0, q1 = row
        gate_type = {v: k for k, v in GATE_VOCAB.items()}[type_id]
        if gate_type in ("rx", "rz"):
            assert q1 == n, f"expected padding index {n} for 1q gate, got {q1}"

def test_encode_circuit_interaction_matrix_type():
    _, _, m = encode_circuit(QASM_8Q)
    assert isinstance(m, np.ndarray)
    assert m.dtype == np.float32


# ── dataset ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    return load_dataset(difficulties=("very_easy", "easy"))

def test_load_dataset_nonempty(dataset):
    assert len(dataset) > 0

def test_load_dataset_returns_circuit_samples(dataset):
    for s in dataset:
        assert isinstance(s, CircuitSample)

def test_all_samples_have_qasm_path(dataset):
    for s in dataset:
        assert s.qasm_path.exists(), f"missing QASM: {s.qasm_path}"

def test_bitstring_length_matches_n_qubits(dataset):
    for s in dataset:
        assert len(s.bitstring) == s.n_qubits, (
            f"{s.challenge}: bitstring len {len(s.bitstring)} != n_qubits {s.n_qubits}"
        )

def test_token_length_matches_n_qubits(dataset):
    for s in dataset:
        assert len(s.tokens) == s.n_qubits

def test_tokens_are_binary(dataset):
    for s in dataset:
        assert all(t in (0, 1) for t in s.tokens)

def test_interaction_matrix_shape_matches_n_qubits(dataset):
    for s in dataset:
        n = s.n_qubits
        assert s.interaction_matrix.shape == (n, n), (
            f"{s.challenge}: matrix shape {s.interaction_matrix.shape} != ({n},{n})"
        )

def test_gate_sequence_nonempty_for_all_samples(dataset):
    for s in dataset:
        assert len(s.gate_sequence) > 0, f"{s.challenge}: empty gate sequence"

def test_only_success_rows_loaded(dataset):
    # every loaded sample must have a non-empty bitstring (failed rows have empty/wrong ones)
    for s in dataset:
        assert len(s.bitstring) > 0

def test_filter_by_difficulty_very_easy():
    samples = load_dataset(difficulties=("very_easy",))
    assert all(s.difficulty == "very_easy" for s in samples)

def test_filter_by_difficulty_easy():
    samples = load_dataset(difficulties=("easy",))
    assert all(s.difficulty == "easy" for s in samples)

def test_no_duplicate_challenges_per_difficulty(dataset):
    seen = set()
    for s in dataset:
        key = (s.challenge, s.difficulty)
        assert key not in seen, f"duplicate: {key}"
        seen.add(key)

def test_probabilities_are_nonnegative(dataset):
    for s in dataset:
        assert s.probability >= 0, f"{s.challenge}: probability {s.probability} < 0"

def test_very_easy_has_ten_challenges():
    samples = load_dataset(difficulties=("very_easy",))
    assert len(samples) == 10, f"expected 10 very_easy challenges, got {len(samples)}"
