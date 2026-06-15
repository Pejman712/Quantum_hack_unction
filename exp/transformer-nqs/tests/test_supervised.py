"""
Sprint 2 — supervised training tests.

Run with the simulation venv (has torch):
    exp/peak-kremer/peaked-circuit-simulation/.venv/bin/python \
        -m pytest exp/transformer-nqs/tests/test_supervised.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import load_dataset
from data.tokenizer import bitstring_to_tokens
from model.transformer import AutoregressiveTransformer
from model.sampler import sample
from training.supervised import train_epoch, compute_loss, TrainingHistory
from training.curriculum import CurriculumScheduler

DEVICE = "cpu"

# ── helpers ───────────────────────────────────────────────────────────────────

def _tiny_model():
    return AutoregressiveTransformer(
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_qubits=128, dropout=0.0,
    )

def _make_tensor(tokens, interaction, device=DEVICE):
    t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    m = torch.tensor(interaction, dtype=torch.float32, device=device)
    return t, m


# ── compute_loss ──────────────────────────────────────────────────────────────

def test_loss_is_positive():
    model = _tiny_model()
    tokens = [1, 0, 1, 0, 1, 1, 0, 1]
    t, m = _make_tensor(tokens, np.zeros((8, 8)))
    loss = compute_loss(model, t, m)
    assert loss.item() > 0

def test_loss_is_finite():
    model = _tiny_model()
    tokens = [0, 1, 1, 0, 1, 0, 0, 1]
    t, m = _make_tensor(tokens, np.zeros((8, 8)))
    loss = compute_loss(model, t, m)
    assert torch.isfinite(loss)

def test_perfect_prediction_gives_near_zero_loss():
    """
    If we force model output to match target exactly, BCE loss should be ~0.
    We test this by checking that loss is lower when probs match tokens.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    tokens = [1, 0, 1, 0, 1, 1, 0, 1]
    t, m = _make_tensor(tokens, np.zeros((8, 8)))
    loss_before = compute_loss(model, t, m).item()

    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(500):
        opt.zero_grad()
        compute_loss(model, t, m).backward()
        opt.step()

    loss_after = compute_loss(model, t, m).item()
    assert loss_after < loss_before, "loss did not decrease after 500 steps"
    assert loss_after < 0.1, f"loss {loss_after:.4f} did not converge near 0"

def test_loss_upper_bound():
    """Random model BCE loss should be ≤ log(2) ≈ 0.693 (uniform distribution)."""
    import math
    model = _tiny_model()
    tokens = [1, 0, 1, 0, 1, 1, 0, 1]
    t, m = _make_tensor(tokens, np.zeros((8, 8)))
    loss = compute_loss(model, t, m).item()
    assert loss <= math.log(2) + 0.05, f"loss {loss:.4f} exceeds log(2) bound"


# ── train_epoch ───────────────────────────────────────────────────────────────

def test_train_epoch_returns_float():
    model = _tiny_model()
    samples = load_dataset(difficulties=("very_easy",))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_epoch(model, opt, samples, device=DEVICE)
    assert isinstance(loss, float)
    assert loss > 0

def test_train_epoch_updates_parameters():
    model = _tiny_model()
    params_before = [p.clone() for p in model.parameters()]
    samples = load_dataset(difficulties=("very_easy",))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_epoch(model, opt, samples, device=DEVICE)
    for before, after in zip(params_before, model.parameters()):
        if before.requires_grad:
            assert not torch.equal(before, after), "parameters did not change"
            return  # one changed param is enough

def test_loss_decreases_over_multiple_epochs():
    """
    After 50 epochs on very_easy data, mean loss should be lower than at epoch 0.
    Uses a small model and high lr to converge fast.
    """
    model = _tiny_model()
    samples = load_dataset(difficulties=("very_easy",))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    loss_start = train_epoch(model, opt, samples, device=DEVICE)
    for _ in range(49):
        loss_end = train_epoch(model, opt, samples, device=DEVICE)

    assert loss_end < loss_start, (
        f"loss did not decrease: {loss_start:.4f} → {loss_end:.4f}"
    )

def test_training_history_records_loss():
    model = _tiny_model()
    samples = load_dataset(difficulties=("very_easy",))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = TrainingHistory()

    for epoch in range(5):
        loss = train_epoch(model, opt, samples, device=DEVICE)
        history.record(epoch, loss)

    assert len(history.losses) == 5
    assert all(isinstance(v, float) for v in history.losses)


# ── memorization: core Sprint 2 milestone ────────────────────────────────────

def test_model_memorizes_single_very_easy_circuit():
    """
    Train on challenge-8_1 (8 qubits, ground truth "10101101", p=0.913).
    After training, the model's most-sampled bitstring must match ground truth.
    """
    samples = load_dataset(difficulties=("very_easy",))
    target = next(s for s in samples if s.challenge == "challenge-8_1")

    model = _tiny_model()
    t, m = _make_tensor(target.tokens, target.interaction_matrix)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(300):
        opt.zero_grad()
        compute_loss(model, t, m).backward()
        opt.step()

    # Sample and check top-1
    bitstrings = sample(model, m, n_qubits=target.n_qubits, n_samples=200)
    counts = {}
    for bs in bitstrings:
        key = tuple(bs.tolist())
        counts[key] = counts.get(key, 0) + 1

    top1 = max(counts, key=counts.__getitem__)
    top1_str = "".join(str(b) for b in top1)
    assert top1_str == target.bitstring, (
        f"top-1 sample '{top1_str}' != ground truth '{target.bitstring}'"
    )

def test_ground_truth_probability_increases_after_training():
    """p(x*) should be substantially higher after training than before."""
    samples = load_dataset(difficulties=("very_easy",))
    target = next(s for s in samples if s.challenge == "challenge-8_1")

    model = _tiny_model()
    t, m = _make_tensor(target.tokens, target.interaction_matrix)

    # p(x*) before training — random model
    with torch.no_grad():
        probs_before = model(t, m)
        log_p_before = sum(
            (torch.log(p) if b == 1 else torch.log(1 - p)).item()
            for b, p in zip(target.tokens, probs_before[0])
        )

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        compute_loss(model, t, m).backward()
        opt.step()

    with torch.no_grad():
        probs_after = model(t, m)
        log_p_after = sum(
            (torch.log(p) if b == 1 else torch.log(1 - p)).item()
            for b, p in zip(target.tokens, probs_after[0])
        )

    assert log_p_after > log_p_before, (
        f"log p(x*) did not increase: {log_p_before:.3f} → {log_p_after:.3f}"
    )


# ── curriculum ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def curriculum():
    ve = load_dataset(difficulties=("very_easy",))
    ea = load_dataset(difficulties=("easy",))
    mo = load_dataset(difficulties=("moderate",))
    return CurriculumScheduler([
        ("very_easy", ve, 20),
        ("easy",      ea, 30),
        ("moderate",  mo, 50),
    ])

def test_curriculum_starts_at_very_easy(curriculum):
    assert curriculum.get_stage(0) == "very_easy"

def test_curriculum_advances_to_easy(curriculum):
    assert curriculum.get_stage(20) == "easy"

def test_curriculum_advances_to_moderate(curriculum):
    assert curriculum.get_stage(50) == "moderate"

def test_curriculum_total_epochs(curriculum):
    assert curriculum.total_epochs == 100

def test_curriculum_very_easy_samples_only_in_stage_0(curriculum):
    samples = curriculum.get_samples(0)
    assert all(s.difficulty == "very_easy" for s in samples)

def test_curriculum_easy_stage_is_cumulative(curriculum):
    samples = curriculum.get_samples(20)
    difficulties = {s.difficulty for s in samples}
    assert "very_easy" in difficulties
    assert "easy" in difficulties

def test_curriculum_moderate_stage_is_cumulative(curriculum):
    samples = curriculum.get_samples(50)
    difficulties = {s.difficulty for s in samples}
    assert "very_easy" in difficulties
    assert "easy" in difficulties
    assert "moderate" in difficulties

def test_curriculum_sample_count_grows(curriculum):
    n_ve = len(curriculum.get_samples(0))
    n_ea = len(curriculum.get_samples(20))
    n_mo = len(curriculum.get_samples(50))
    assert n_ve < n_ea < n_mo

def test_curriculum_clamps_final_stage_beyond_total(curriculum):
    """Any epoch past the end should stay in the last stage."""
    assert curriculum.get_stage(9999) == "moderate"
    samples = curriculum.get_samples(9999)
    difficulties = {s.difficulty for s in samples}
    assert "moderate" in difficulties
