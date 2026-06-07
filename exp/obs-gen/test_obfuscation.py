"""
Tests that make_peaked obfuscation preserves the planted secret bitstring.

Strategy: generate obfuscated circuits for small n (≤12) where exact statevector
simulation is cheap, then verify the peak bitstring matches the planted secret.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from qiskit.quantum_info import Statevector

# ---------------------------------------------------------------------------
# Import make_peaked from exp/obs-gen/obs_gen.py (not a package)
# ---------------------------------------------------------------------------
_OBS_GEN = Path(__file__).resolve().parents[1] / "exp" / "obs-gen" / "obs_gen.py"
_spec = importlib.util.spec_from_file_location("obs_gen", _OBS_GEN)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["obs_gen"] = _mod
_spec.loader.exec_module(_mod)
make_peaked = _mod.make_peaked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _statevector_peak(qc) -> tuple[str, float]:
    probs = Statevector(qc).probabilities_dict()
    peak = max(probs, key=probs.get)
    return peak, probs[peak]


# ---------------------------------------------------------------------------
# Basic correctness: peak bitstring == planted secret
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_peak_matches_secret_n8(seed):
    qc, secret = make_peaked(n=8, depth=3, seed=seed)
    peak, _ = _statevector_peak(qc)
    assert peak == secret, f"seed={seed}: peak={peak!r} != secret={secret!r}"


@pytest.mark.parametrize("seed", [0, 5])
def test_peak_matches_secret_n4(seed):
    qc, secret = make_peaked(n=4, depth=2, seed=seed)
    peak, _ = _statevector_peak(qc)
    assert peak == secret


@pytest.mark.parametrize("seed", [0, 3])
def test_peak_matches_secret_n12(seed):
    qc, secret = make_peaked(n=12, depth=3, seed=seed)
    peak, _ = _statevector_peak(qc)
    assert peak == secret


# ---------------------------------------------------------------------------
# Peak probability is non-trivially above uniform baseline (1/2^n)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6, 8])
def test_peak_probability_above_uniform(n):
    qc, _ = make_peaked(n=n, depth=3, seed=0)
    _, prob = _statevector_peak(qc)
    uniform = 1.0 / (2**n)
    assert prob > 5 * uniform, f"n={n}: peak prob {prob:.4f} barely above uniform {uniform:.4f}"


# ---------------------------------------------------------------------------
# Different seeds plant different secrets
# ---------------------------------------------------------------------------

def test_different_seeds_plant_different_secrets():
    secrets = {make_peaked(n=8, depth=3, seed=s)[1] for s in range(10)}
    assert len(secrets) > 1, "all seeds produced the same secret — RNG not seeded properly"


# ---------------------------------------------------------------------------
# Explicit secret is respected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("secret,expected", [
    ([1, 0, 1, 0], "0101"),   # q0 rightmost → reversed
    ([0, 0, 0, 0], "0000"),
    ([1, 1, 1, 1], "1111"),
])
def test_explicit_secret_planted_correctly(secret, expected):
    qc, secret_str = make_peaked(n=len(secret), depth=2, secret=secret, seed=0)
    assert secret_str == expected
    peak, _ = _statevector_peak(qc)
    assert peak == secret_str


# ---------------------------------------------------------------------------
# Obfuscation knobs do not break the planted answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("swap_k", [1, 2, 4])
def test_swap_trick_still_peaks_at_secret(swap_k):
    qc, secret = make_peaked(n=8, depth=3, swap_k=swap_k, seed=0)
    peak, _ = _statevector_peak(qc)
    assert peak == secret, f"swap_k={swap_k}: peak={peak!r} != secret={secret!r}"


@pytest.mark.parametrize("sweep_std", [0.0, 0.005, 0.01])
def test_angle_sweep_still_peaks_at_secret(sweep_std):
    qc, secret = make_peaked(n=8, depth=3, sweep_std=sweep_std, seed=0)
    peak, _ = _statevector_peak(qc)
    assert peak == secret, f"sweep_std={sweep_std}: peak={peak!r} != secret={secret!r}"


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_deeper_blocks_still_peak_at_secret(depth):
    qc, secret = make_peaked(n=8, depth=depth, seed=0)
    peak, _ = _statevector_peak(qc)
    assert peak == secret, f"depth={depth}: peak={peak!r} != secret={secret!r}"


# ---------------------------------------------------------------------------
# Output format: secret_str is valid binary of correct length
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_secret_str_is_binary_and_correct_length(n):
    _, secret_str = make_peaked(n=n, depth=2, seed=0)
    assert len(secret_str) == n
    assert set(secret_str) <= {"0", "1"}
