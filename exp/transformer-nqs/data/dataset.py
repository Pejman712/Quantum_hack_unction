"""
Load ground-truth (challenge, bitstring) pairs from CSV files and pair each
with its encoded circuit representation.

Usage
-----
from data.dataset import load_dataset

samples = load_dataset(difficulties=("very_easy", "easy", "moderate"))
for s in samples:
    print(s.challenge, s.n_qubits, s.bitstring, s.probability)
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .circuit_encoder import encode_circuit
from .tokenizer import bitstring_to_tokens

DIFFICULTIES = ("very_easy", "easy", "moderate", "hard", "very_hard")

_HERE        = Path(__file__).resolve().parent
_REPO_ROOT   = _HERE.parents[2]          # exp/transformer-nqs/data → repo root
RESULTS_DIR  = _REPO_ROOT / "results"
DATA_DIR     = _REPO_ROOT / "qasm_data"


@dataclass
class CircuitSample:
    challenge:          str
    difficulty:         str
    n_qubits:           int
    bitstring:          str
    tokens:             list
    probability:        float
    gate_sequence:      list
    interaction_matrix: np.ndarray
    qasm_path:          Path


def _find_qasm(challenge, data_dir):
    """Search all difficulty tiers for the QASM file of a challenge."""
    for diff in DIFFICULTIES:
        path = data_dir / diff / f"{challenge}.qasm"
        if path.exists():
            return path, diff
    return None, None


def load_dataset(difficulties=None, data_dir=DATA_DIR, results_dir=RESULTS_DIR):
    """
    Load success rows from result CSVs and return encoded CircuitSample objects.

    Parameters
    ----------
    difficulties : iterable of str, optional
        Which difficulty tiers to include. Defaults to all tiers that have a
        corresponding *_bitstrings.csv file.
    data_dir : Path
        Root of the qasm_data/ directory tree.
    results_dir : Path
        Directory containing the *_bitstrings.csv files.

    Returns
    -------
    list[CircuitSample]  — one entry per (challenge, bitstring) success row,
                           deduplicated so each challenge appears at most once
                           (the last success row wins, matching the existing
                           load_results convention in the repo).
    """
    if difficulties is None:
        difficulties = DIFFICULTIES

    latest = {}  # challenge → row dict, keeps last success per challenge

    for diff in difficulties:
        csv_path = results_dir / f"{diff}_bitstrings.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row.get("status") != "success":
                    continue
                latest[row["challenge"]] = row

    samples = []
    for challenge, row in sorted(latest.items()):
        qasm_path, found_diff = _find_qasm(challenge, data_dir)
        if qasm_path is None:
            continue

        n_qubits, gate_seq, interaction = encode_circuit(qasm_path)

        samples.append(CircuitSample(
            challenge=challenge,
            difficulty=found_diff,
            n_qubits=n_qubits,
            bitstring=row["bitstring"],
            tokens=bitstring_to_tokens(row["bitstring"]),
            probability=float(row["probability"]),
            gate_sequence=gate_seq,
            interaction_matrix=interaction,
            qasm_path=qasm_path,
        ))

    return samples
