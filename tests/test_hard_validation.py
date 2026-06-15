"""
Validate the MPS peak-finding pipeline against known-correct hard bitstrings.

Reads the success rows from results/hard_bitstrings.csv as ground truth,
runs matrix_product_operators on each challenge QASM, and asserts the
predicted bitstring matches.

Hard circuits are 40-64 qubits — bond_dim needs to be high enough to
represent the peaked state accurately.  Pass --bond-dim on the command
line (defined in conftest.py) to control truncation; 256+ is recommended.

Run with:
    uv run pytest tests/test_hard_validation.py -v --bond-dim 256
    uv run pytest tests/test_hard_validation.py -v -k "challenge_48"   # one circuit

Note: results/hard_bitstrings.csv is populated by Puhti runs.  Until at
least one success row exists the test collects 0 cases.
"""

from pathlib import Path
from collections import defaultdict

import pytest

from quantum_hack.challenges import find_challenge, load_circuit
from quantum_hack.results import load_results
from quantum_hack.simulation import matrix_product_operators

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR    = Path(__file__).resolve().parents[1] / "qasm_data"

# ── build ground-truth table: challenge → latest success bitstring ──────────

def _ground_truth() -> dict[str, str]:
    """Return {challenge_name: bitstring} for the last success row per challenge."""
    records = load_results(RESULTS_DIR)
    latest: dict[str, str] = {}
    for rec in records:
        if rec.difficulty == "hard" and rec.status == "success":
            latest[rec.challenge] = rec.bitstring   # last success wins
    return latest


_GT = _ground_truth()

# ── parametrize one test case per hard challenge ──────────────────────────────

@pytest.mark.parametrize(
    "challenge,expected",
    sorted(_GT.items()),
    ids=sorted(_GT.keys()),
)
def test_mps_matches_csv_bitstring(challenge: str, expected: str, bond_dim: int, device: str) -> None:
    """MPS prediction for each hard challenge must match the CSV success entry."""
    qasm_path = find_challenge(challenge, data_dir=DATA_DIR / "hard")
    qc = load_circuit(qasm_path)

    predicted, prob = matrix_product_operators(qc, shots=4096, bond_dim=bond_dim, device=device)

    assert predicted == expected, (
        f"{challenge}: MPS predicted {predicted!r} (prob={prob:.3f}) "
        f"but CSV says {expected!r}"
    )
