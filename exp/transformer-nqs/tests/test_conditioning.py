"""
Sprint 3 — circuit conditioning tests.

Run with the simulation venv:
    exp/peak-kremer/peaked-circuit-simulation/.venv/bin/python \
        -m pytest exp/transformer-nqs/tests/test_conditioning.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.conditioning import CircuitConditioner, normalize_interaction
from model.transformer import AutoregressiveTransformer
from model.sampler import sample
from data.circuit_encoder import encode_circuit_typed, GATE_VOCAB
from data.dataset import load_dataset
from training.supervised import compute_loss, train_epoch

DEVICE = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[3]
QASM_8Q  = REPO_ROOT / "qasm_data" / "very_easy" / "challenge-8_1.qasm"
QASM_48Q = REPO_ROOT / "qasm_data" / "very_hard"  / "challenge-48_42.qasm"


# ── normalize_interaction ─────────────────────────────────────────────────────

def test_normalize_zero_matrix_gives_zero():
    m = torch.zeros(8, 8)
    out = normalize_interaction(m)
    assert (out == 0).all()

def test_normalize_output_in_unit_range():
    m = torch.randint(0, 20, (8, 8)).float()
    m = (m + m.T) / 2  # symmetric
    out = normalize_interaction(m)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6

def test_normalize_preserves_symmetry():
    m = torch.randint(0, 10, (8, 8)).float()
    m = m + m.T
    out = normalize_interaction(m)
    torch.testing.assert_close(out, out.T, atol=1e-6, rtol=0)

def test_normalize_is_monotone():
    """Higher raw count → higher normalized value."""
    m = torch.zeros(4, 4)
    m[0, 1] = m[1, 0] = 5.0
    m[2, 3] = m[3, 2] = 10.0
    out = normalize_interaction(m)
    assert out[2, 3] > out[0, 1]

def test_normalize_log1p_compresses_large_values():
    """log1p(100) / log1p(100) = 1.0, log1p(1) / log1p(100) << 1."""
    import math
    m = torch.zeros(4, 4)
    m[0, 1] = m[1, 0] = 1.0
    m[2, 3] = m[3, 2] = 100.0
    out = normalize_interaction(m)
    # Without log compression, ratio would be 1/100 = 0.01
    # With log1p: log(2)/log(101) ≈ 0.15 — much more balanced
    ratio = out[0, 1].item() / out[2, 3].item()
    assert ratio > 0.1, f"log compression not applied: ratio={ratio:.3f}"


# ── CircuitConditioner ─────────────────────────────────────────────────────────

@pytest.fixture
def conditioner():
    return CircuitConditioner(n_layers=4)

def test_conditioner_layer_scales_start_at_zero(conditioner):
    assert (conditioner.layer_scales == 0).all()

def test_conditioner_cx_swap_weights_are_positive(conditioner):
    """Gate-type weights should initialise to positive values."""
    assert conditioner.cx_weight.item() > 0
    assert conditioner.swap_weight.item() > 0

def test_conditioner_combine_gives_tensor(conditioner):
    cx   = torch.zeros(8, 8)
    swap = torch.zeros(8, 8)
    cx[0,1] = cx[1,0] = 2.0
    swap[2,3] = swap[3,2] = 1.0
    combined = conditioner.combine(cx, swap)
    assert combined.shape == (8, 8)
    assert combined[0, 1] > 0
    assert combined[2, 3] > 0

def test_conditioner_combine_symmetric(conditioner):
    cx   = torch.zeros(8, 8); cx[1,3] = cx[3,1] = 3.0
    swap = torch.zeros(8, 8); swap[2,4] = swap[4,2] = 1.0
    combined = conditioner.combine(cx, swap)
    torch.testing.assert_close(combined, combined.T, atol=1e-6, rtol=0)

def test_conditioner_get_layer_bias_shape(conditioner):
    bias = torch.rand(8, 8)
    for i in range(4):
        out = conditioner.get_layer_bias(bias, i)
        assert out.shape == (8, 8)

def test_conditioner_per_layer_scales_independent(conditioner):
    """Setting one layer's scale must not affect another."""
    with torch.no_grad():
        conditioner.layer_scales[0] = 1.0
        conditioner.layer_scales[1] = 0.5
    bias = torch.ones(4, 4)
    b0 = conditioner.get_layer_bias(bias, 0)
    b1 = conditioner.get_layer_bias(bias, 1)
    assert not torch.allclose(b0, b1)

def test_conditioner_gradient_flows_through_layer_scales():
    cond = CircuitConditioner(n_layers=2)
    bias = normalize_interaction(torch.rand(4, 4))
    with torch.no_grad():
        cond.layer_scales.fill_(1.0)
    out = cond.get_layer_bias(bias, 0)
    loss = out.sum()
    loss.backward()
    assert cond.layer_scales.grad is not None
    assert cond.layer_scales.grad[0].abs() > 0

def test_conditioner_gradient_flows_through_cx_weight():
    cond   = CircuitConditioner(n_layers=2)
    cx     = torch.rand(4, 4)
    swap   = torch.zeros(4, 4)
    combined = cond.combine(cx, swap)
    combined.sum().backward()
    assert cond.cx_weight.grad is not None


# ── encode_circuit_typed ──────────────────────────────────────────────────────

def test_encode_circuit_typed_returns_five_values():
    result = encode_circuit_typed(QASM_8Q)
    assert len(result) == 5

def test_encode_circuit_typed_n_qubits_8():
    n, _, cx, swap, _ = encode_circuit_typed(QASM_8Q)
    assert n == 8

def test_cx_matrix_shape():
    n, _, cx, swap, _ = encode_circuit_typed(QASM_8Q)
    assert cx.shape == (n, n)

def test_swap_matrix_shape():
    n, _, cx, swap, _ = encode_circuit_typed(QASM_8Q)
    assert swap.shape == (n, n)

def test_cx_and_swap_matrices_are_symmetric():
    n, _, cx, swap, _ = encode_circuit_typed(QASM_8Q)
    np.testing.assert_array_equal(cx, cx.T)
    np.testing.assert_array_equal(swap, swap.T)

def test_cx_and_swap_sum_equals_combined():
    """cx_matrix + swap_matrix must equal the original interaction_matrix."""
    from data.circuit_encoder import encode_circuit
    n_orig, _, combined = encode_circuit(QASM_8Q)
    n, _, cx, swap, _   = encode_circuit_typed(QASM_8Q)
    np.testing.assert_array_almost_equal(cx + swap, combined)

def test_typed_matrices_nonnegative():
    _, _, cx, swap, _ = encode_circuit_typed(QASM_48Q)
    assert (cx >= 0).all()
    assert (swap >= 0).all()


# ── transformer uses conditioner ──────────────────────────────────────────────

def _tiny_model():
    return AutoregressiveTransformer(
        d_model=32, n_heads=2, n_layers=2, d_ff=64, max_qubits=128, dropout=0.0
    )

def test_transformer_has_conditioner():
    model = _tiny_model()
    assert hasattr(model, "conditioner")
    assert isinstance(model.conditioner, CircuitConditioner)

def test_transformer_forward_with_typed_matrices():
    model = _tiny_model()
    n, _, cx, swap, _ = encode_circuit_typed(QASM_8Q)
    tokens = torch.randint(0, 2, (2, n))
    cx_t   = torch.tensor(cx,   dtype=torch.float32)
    swap_t = torch.tensor(swap, dtype=torch.float32)
    probs  = model(tokens, cx_matrix=cx_t, swap_matrix=swap_t)
    assert probs.shape == (2, n)
    assert torch.isfinite(probs).all()

def test_transformer_forward_interaction_matrix_still_works():
    """Old API must remain backward compatible."""
    model = _tiny_model()
    tokens      = torch.randint(0, 2, (1, 8))
    interaction = torch.zeros(8, 8)
    probs = model(tokens, interaction)
    assert probs.shape == (1, 8)

def test_layer_scales_become_nonzero_after_training():
    """
    After training on a circuit, the conditioner layer scales should
    move away from zero — the model learns to use circuit structure.
    """
    torch.manual_seed(7)
    model  = _tiny_model()
    sample_list = load_dataset(difficulties=("very_easy",))
    target = next(s for s in sample_list if s.challenge == "challenge-8_1")

    _, _, cx, swap, _ = encode_circuit_typed(target.qasm_path)
    cx_t   = torch.tensor(cx,   dtype=torch.float32)
    swap_t = torch.tensor(swap, dtype=torch.float32)
    tokens = torch.tensor(target.tokens, dtype=torch.long).unsqueeze(0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        probs = model(tokens, cx_matrix=cx_t, swap_matrix=swap_t)
        loss  = torch.nn.functional.binary_cross_entropy(probs, tokens.float())
        loss.backward()
        opt.step()

    scales = model.conditioner.layer_scales.abs()
    assert scales.max().item() > 1e-3, (
        f"all layer scales near zero after training: {scales.tolist()}"
    )


# ── conditioned vs unconditioned ─────────────────────────────────────────────

def test_conditioned_model_lower_loss_than_unconditioned():
    """
    Train two identical models (same random seed) for 30 epochs.  One sees
    the real circuit interaction matrix; the other sees zeros.  The conditioned
    model should achieve lower loss because:
      - layer_scales get non-zero gradients and develop a structural prior
      - the interaction matrix encodes genuine qubit correlations
    Both models are seeded identically so the comparison is fair.
    """
    samples = load_dataset(difficulties=("very_easy",))

    def _run(use_interaction, seed=42):
        torch.manual_seed(seed)
        model = _tiny_model()
        opt   = torch.optim.Adam(model.parameters(), lr=5e-3)
        for _ in range(30):
            for s in samples:
                tokens = torch.tensor(s.tokens, dtype=torch.long).unsqueeze(0)
                if use_interaction:
                    inter = torch.tensor(s.interaction_matrix, dtype=torch.float32)
                else:
                    inter = torch.zeros(s.n_qubits, s.n_qubits)
                loss = compute_loss(model, tokens, inter)
                opt.zero_grad()
                loss.backward()
                opt.step()

        total = 0.0
        for s in samples:
            tokens = torch.tensor(s.tokens, dtype=torch.long).unsqueeze(0)
            inter  = (torch.tensor(s.interaction_matrix, dtype=torch.float32)
                      if use_interaction else torch.zeros(s.n_qubits, s.n_qubits))
            total += compute_loss(model, tokens, inter).item()
        return total / len(samples)

    loss_conditioned   = _run(use_interaction=True,  seed=42)
    loss_unconditioned = _run(use_interaction=False, seed=42)

    assert loss_conditioned < loss_unconditioned, (
        f"conditioned loss {loss_conditioned:.4f} >= unconditioned {loss_unconditioned:.4f}"
    )
