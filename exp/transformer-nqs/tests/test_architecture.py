"""
Sprint 1 — core transformer architecture tests.

Run from repo root (using simulation venv which has torch):
    .venv path: exp/peak-kremer/peaked-circuit-simulation/.venv
    pytest exp/transformer-nqs/tests/test_architecture.py -v
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.transformer import AutoregressiveTransformer
from model.sampler import sample

# ── fixtures ─────────────────────────────────────────────────────────────────

N_SMALL = 8   # very_easy circuit size
N_MED   = 16
BATCH   = 4

@pytest.fixture
def small_model():
    return AutoregressiveTransformer(
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_qubits=128, dropout=0.0,
    )

@pytest.fixture
def interaction_8():
    """Fake interaction matrix: qubit i talks to qubit i+1."""
    m = torch.zeros(N_SMALL, N_SMALL)
    for i in range(N_SMALL - 1):
        m[i, i + 1] = 3.0
        m[i + 1, i] = 3.0
    return m

@pytest.fixture
def tokens_8():
    return torch.randint(0, 2, (BATCH, N_SMALL))


# ── output shape and range ────────────────────────────────────────────────────

def test_forward_output_shape(small_model, tokens_8, interaction_8):
    probs = small_model(tokens_8, interaction_8)
    assert probs.shape == (BATCH, N_SMALL), f"expected ({BATCH},{N_SMALL}), got {probs.shape}"

def test_output_probabilities_in_0_1(small_model, tokens_8, interaction_8):
    probs = small_model(tokens_8, interaction_8)
    assert (probs >= 0).all() and (probs <= 1).all(), "probabilities must be in [0,1]"

def test_output_no_nan_or_inf(small_model, tokens_8, interaction_8):
    probs = small_model(tokens_8, interaction_8)
    assert torch.isfinite(probs).all(), "output contains NaN or Inf"

def test_different_inputs_different_outputs(small_model, interaction_8):
    a = torch.zeros(1, N_SMALL, dtype=torch.long)
    b = torch.ones(1, N_SMALL, dtype=torch.long)
    pa = small_model(a, interaction_8)
    pb = small_model(b, interaction_8)
    assert not torch.allclose(pa, pb), "model should respond to different inputs"


# ── causal masking ────────────────────────────────────────────────────────────

def test_causal_mask_blocks_future_tokens(small_model, interaction_8):
    """
    Changing token at position k must NOT affect output probabilities at
    positions 0..k.  (Those positions predict bits 0..k using only the
    BOS token and bits 0..k-1 as inputs — they cannot see bit k.)
    """
    small_model.eval()
    with torch.no_grad():
        base = torch.zeros(1, N_SMALL, dtype=torch.long)

        for k in range(1, N_SMALL):
            modified = base.clone()
            modified[0, k] = 1          # change only position k

            p_base = small_model(base, interaction_8)
            p_mod  = small_model(modified, interaction_8)

            # positions 0..k must be unaffected
            torch.testing.assert_close(
                p_base[0, :k + 1], p_mod[0, :k + 1],
                atol=1e-5, rtol=0,
                msg=f"causal mask violated: changing token {k} affected output at 0..{k}",
            )

def test_first_output_ignores_all_tokens(small_model, interaction_8):
    """p(x₀) depends only on BOS — should be the same for any token sequence."""
    small_model.eval()
    with torch.no_grad():
        a = torch.zeros(1, N_SMALL, dtype=torch.long)
        b = torch.ones(1, N_SMALL, dtype=torch.long)
        pa = small_model(a, interaction_8)
        pb = small_model(b, interaction_8)
        torch.testing.assert_close(pa[0, 0], pb[0, 0], atol=1e-5, rtol=0)


# ── autoregressive probability distribution ───────────────────────────────────

def test_autoregressive_factorization_sums_to_1(small_model):
    """
    For a small n, Σ_{x ∈ {0,1}^n} p(x) must equal 1.0.
    Uses n=4 to keep the loop over 2^4=16 bitstrings fast.
    """
    small_model.eval()
    n = 4
    interaction = torch.zeros(n, n)

    total = 0.0
    with torch.no_grad():
        for bits in itertools.product([0, 1], repeat=n):
            tokens = torch.tensor([list(bits)])           # (1, n)
            probs  = small_model(tokens, interaction)     # (1, n)
            p = 1.0
            for i, bit in enumerate(bits):
                p_i = probs[0, i].item()
                p  *= p_i if bit == 1 else (1.0 - p_i)
            total += p

    assert abs(total - 1.0) < 1e-4, f"probabilities sum to {total}, expected 1.0"


# ── interaction bias ──────────────────────────────────────────────────────────

def test_no_interaction_bias_still_works(small_model, tokens_8):
    """Model must work with a zero interaction matrix (no circuit structure)."""
    zero_bias = torch.zeros(N_SMALL, N_SMALL)
    probs = small_model(tokens_8, zero_bias)
    assert probs.shape == (BATCH, N_SMALL)
    assert torch.isfinite(probs).all()

def test_interaction_bias_changes_output(small_model, tokens_8):
    """Non-zero interaction matrix must change the output (given trained non-zero scale)."""
    zero_bias   = torch.zeros(N_SMALL, N_SMALL)
    strong_bias = torch.ones(N_SMALL, N_SMALL) * 10.0
    strong_bias.fill_diagonal_(0.0)

    # force the bias scale to be non-zero so we can detect the change
    with torch.no_grad():
        small_model.conditioner.layer_scales.fill_(1.0)

    p_zero   = small_model(tokens_8, zero_bias)
    p_strong = small_model(tokens_8, strong_bias)
    assert not torch.allclose(p_zero, p_strong, atol=1e-4), \
        "interaction bias had no effect even with scale=1"


# ── variable circuit sizes ────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [8, 16, 32, 64, 104])
def test_handles_variable_qubit_counts(small_model, n):
    tokens      = torch.randint(0, 2, (2, n))
    interaction = torch.zeros(n, n)
    probs = small_model(tokens, interaction)
    assert probs.shape == (2, n)
    assert torch.isfinite(probs).all()


# ── sampler ───────────────────────────────────────────────────────────────────

def test_sample_shape(small_model, interaction_8):
    bitstrings = sample(small_model, interaction_8, n_qubits=N_SMALL, n_samples=16)
    assert bitstrings.shape == (16, N_SMALL)

def test_sample_values_are_binary(small_model, interaction_8):
    bitstrings = sample(small_model, interaction_8, n_qubits=N_SMALL, n_samples=32)
    assert set(bitstrings.unique().tolist()).issubset({0, 1})

def test_sample_reproducible_with_seed(small_model, interaction_8):
    torch.manual_seed(42)
    s1 = sample(small_model, interaction_8, n_qubits=N_SMALL, n_samples=8)
    torch.manual_seed(42)
    s2 = sample(small_model, interaction_8, n_qubits=N_SMALL, n_samples=8)
    torch.testing.assert_close(s1.float(), s2.float())

def test_sample_produces_variety(small_model, interaction_8):
    """With 64 samples from an 8-qubit model, we should see more than one unique bitstring."""
    bitstrings = sample(small_model, interaction_8, n_qubits=N_SMALL, n_samples=64)
    n_unique = len(set(tuple(b.tolist()) for b in bitstrings))
    assert n_unique > 1, "sampler always returns the same bitstring — check randomness"

def test_sample_log_probs_shape(small_model, interaction_8):
    _, log_probs = sample(
        small_model, interaction_8, n_qubits=N_SMALL, n_samples=8, return_log_probs=True
    )
    assert log_probs.shape == (8,)
    assert (log_probs <= 0).all(), "log probabilities must be <= 0"
