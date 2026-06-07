"""
Validate the MPS peak-finding pipeline against known-correct moderate bitstrings.

Reads the success rows from results/moderate_bitstrings.csv as ground truth,
runs matrix_product_operators on each challenge QASM, and asserts the
predicted bitstring matches.

Run with:
    uv run pytest tests/test_moderate_validation.py -v
    uv run pytest tests/test_moderate_validation.py -v -k "challenge_16"   # one circuit
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
        if rec.difficulty == "moderate" and rec.status == "success":
            latest[rec.challenge] = rec.bitstring   # last success wins
    return latest


_GT = _ground_truth()

# ── parametrize one test case per moderate challenge ─────────────────────────

@pytest.mark.parametrize(
    "challenge,expected",
    sorted(_GT.items()),
    ids=sorted(_GT.keys()),
)
def test_mps_matches_csv_bitstring(challenge: str, expected: str, bond_dim: int) -> None:
    """MPS prediction for each moderate challenge must match the CSV success entry."""
    qasm_path = find_challenge(challenge, data_dir=DATA_DIR / "moderate")
    qc = load_circuit(qasm_path)

    predicted, prob = matrix_product_operators(qc, shots=4096, bond_dim=bond_dim)

    assert predicted == expected, (
        f"{challenge}: MPS predicted {predicted!r} (prob={prob:.3f}) "
        f"but CSV says {expected!r}"
    )
