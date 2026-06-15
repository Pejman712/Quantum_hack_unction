"""
Sprint 5 — VMC fine-tuning tests.

Run with the simulation venv:
    exp/peak-kremer/peaked-circuit-simulation/.venv/bin/python \
        -m pytest exp/transformer-nqs/tests/test_vmc.py -v
"""

import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.transformer import AutoregressiveTransformer
from model.sampler import sample
from data.dataset import load_dataset
from training.vmc import compute_log_prob, reinforce_loss, vmc_step, vmc_train
from training.amplitude import quimb_amplitude, make_peaked_mock


REPO_ROOT = Path(__file__).resolve().parents[3]
QASM_8Q   = REPO_ROOT / "qasm_data" / "very_easy" / "challenge-8_1.qasm"


def _tiny_model():
    return AutoregressiveTransformer(
        d_model=32, n_heads=2, n_layers=2, d_ff=64, max_qubits=128, dropout=0.0
    )


# ── compute_log_prob ──────────────────────────────────────────────────────────

def test_compute_log_prob_shape():
    model  = _tiny_model()
    tokens = torch.randint(0, 2, (4, 8))
    inter  = torch.zeros(8, 8)
    log_p  = compute_log_prob(model, tokens, inter)
    assert log_p.shape == (4,)

def test_compute_log_prob_is_nonpositive():
    model  = _tiny_model()
    tokens = torch.randint(0, 2, (4, 8))
    inter  = torch.zeros(8, 8)
    log_p  = compute_log_prob(model, tokens, inter)
    assert (log_p <= 0 + 1e-6).all(), "log probabilities must be ≤ 0"

def test_compute_log_prob_is_finite():
    model  = _tiny_model()
    tokens = torch.randint(0, 2, (4, 8))
    inter  = torch.zeros(8, 8)
    log_p  = compute_log_prob(model, tokens, inter)
    assert torch.isfinite(log_p).all()

def test_compute_log_prob_gradient_flows():
    model  = _tiny_model()
    tokens = torch.randint(0, 2, (2, 8))
    inter  = torch.zeros(8, 8)
    log_p  = compute_log_prob(model, tokens, inter)
    log_p.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_grad = sum(g.abs().sum().item() for g in grads)
    assert total_grad > 0, "no gradient flowed back through log_prob"

def test_compute_log_prob_different_sequences_give_different_values():
    model  = _tiny_model()
    inter  = torch.zeros(8, 8)
    t1     = torch.zeros(1, 8, dtype=torch.long)
    t2     = torch.ones(1, 8, dtype=torch.long)
    lp1    = compute_log_prob(model, t1, inter)
    lp2    = compute_log_prob(model, t2, inter)
    assert not torch.allclose(lp1, lp2), "different bitstrings must give different log probs"

def test_compute_log_prob_sums_to_approx_zero_for_n1():
    """For n=1, sum of log p(0) and log p(1) should equal log 1 = 0."""
    model  = _tiny_model()
    model.eval()
    inter  = torch.zeros(1, 1)
    t0     = torch.zeros(1, 1, dtype=torch.long)
    t1     = torch.ones(1, 1, dtype=torch.long)
    with torch.no_grad():
        lp0 = compute_log_prob(model, t0, inter)
        lp1 = compute_log_prob(model, t1, inter)
    total = (lp0.exp() + lp1.exp()).item()
    assert abs(total - 1.0) < 1e-4, f"p(0) + p(1) = {total:.6f}, expected 1.0"


# ── reinforce_loss ────────────────────────────────────────────────────────────

def test_reinforce_loss_is_scalar():
    log_probs = torch.tensor([-1.0, -2.0, -3.0])
    rewards   = torch.tensor([1.0, 0.5, -1.0])
    loss = reinforce_loss(log_probs, rewards)
    assert loss.shape == ()

def test_reinforce_loss_gradient_exists():
    log_probs = torch.tensor([-1.0, -2.0], requires_grad=True)
    rewards   = torch.tensor([1.0, 0.5])
    loss = reinforce_loss(log_probs, rewards)
    loss.backward()
    assert log_probs.grad is not None
    assert log_probs.grad.abs().sum() > 0

def test_reinforce_baseline_reduces_magnitude():
    """
    For rewards with high spread, the baseline (mean subtraction) should
    reduce the magnitude of the loss compared to no baseline.
    """
    log_probs = torch.tensor([-1.0, -2.0, -3.0])
    rewards   = torch.tensor([100.0, 1.0, 0.01])
    with_bl    = reinforce_loss(log_probs, rewards, use_baseline=True)
    without_bl = reinforce_loss(log_probs, rewards, use_baseline=False)
    assert abs(with_bl.item()) < abs(without_bl.item())

def test_reinforce_high_reward_sample_gets_higher_prob_gradient():
    """
    The gradient of reinforce_loss w.r.t. log_probs should be larger in magnitude
    for the sample with the highest advantage (reward - baseline).
    """
    log_probs = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    rewards   = torch.tensor([10.0, 1.0, 0.0])
    loss = reinforce_loss(log_probs, rewards, use_baseline=True)
    loss.backward()
    # Sample 0 has highest advantage (10 - mean) — should get the largest |grad|
    assert log_probs.grad[0].abs() > log_probs.grad[2].abs()

def test_reinforce_constant_reward_gives_zero_gradient():
    """When all rewards are equal, the baseline removes them and loss is 0."""
    log_probs = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    rewards   = torch.tensor([5.0, 5.0, 5.0])
    loss = reinforce_loss(log_probs, rewards, use_baseline=True)
    assert abs(loss.item()) < 1e-6


# ── quimb_amplitude ───────────────────────────────────────────────────────────

def test_quimb_amplitude_ground_truth():
    """The 8-qubit circuit's known peaked bitstring must have amplitude ≈ 0.913."""
    log_amp_sq = quimb_amplitude(QASM_8Q, "10101101")
    assert math.isfinite(log_amp_sq)
    amp_sq = math.exp(log_amp_sq)
    assert abs(amp_sq - 0.913) < 0.01, f"|A|^2 = {amp_sq:.4f}, expected ≈ 0.913"

def test_quimb_amplitude_returns_float():
    log_amp_sq = quimb_amplitude(QASM_8Q, "10101101")
    assert isinstance(log_amp_sq, float)

def test_quimb_amplitude_nonpeak_is_small():
    """A random wrong bitstring should have much lower amplitude than the peak."""
    log_peak   = quimb_amplitude(QASM_8Q, "10101101")
    log_random = quimb_amplitude(QASM_8Q, "00000000")
    assert log_peak > log_random

def test_quimb_amplitude_log_scale():
    """Return value is log|A|² (negative for probabilities < 1)."""
    log_amp_sq = quimb_amplitude(QASM_8Q, "10101101")
    assert log_amp_sq < 0  # log(0.913) < 0


# ── make_peaked_mock ──────────────────────────────────────────────────────────

def test_make_peaked_mock_peaks_at_target():
    target = "11001100"
    fn = make_peaked_mock(target, peak_prob=0.9)
    log_peak   = fn(None, target)
    log_other  = fn(None, "00000000")
    assert log_peak > log_other

def test_make_peaked_mock_peak_amplitude_is_correct():
    target = "11001100"
    fn = make_peaked_mock(target, peak_prob=0.9)
    amp_sq = math.exp(fn(None, target))
    assert abs(amp_sq - 0.9) < 0.01

def test_make_peaked_mock_is_finite():
    fn = make_peaked_mock("10101010", peak_prob=0.8)
    log_a = fn(None, "10101010")
    assert math.isfinite(log_a)


# ── vmc_step ──────────────────────────────────────────────────────────────────

def test_vmc_step_returns_float_pair():
    model = _tiny_model()
    inter = torch.zeros(8, 8)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    mock  = make_peaked_mock("11111111")
    mean_r, loss_val = vmc_step(model, inter, None, mock, n_qubits=8, n_samples=8, optimizer=opt)
    assert isinstance(mean_r, float)
    assert isinstance(loss_val, float)

def test_vmc_step_loss_is_finite():
    model = _tiny_model()
    inter = torch.zeros(8, 8)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    mock  = make_peaked_mock("10101010")
    _, loss_val = vmc_step(model, inter, None, mock, n_qubits=8, n_samples=16, optimizer=opt)
    assert math.isfinite(loss_val)

def test_vmc_step_updates_parameters():
    model = _tiny_model()
    params_before = [p.clone().detach() for p in model.parameters()]
    inter = torch.zeros(8, 8)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
    mock  = make_peaked_mock("11111111")
    vmc_step(model, inter, None, mock, n_qubits=8, n_samples=32, optimizer=opt)
    for before, after in zip(params_before, model.parameters()):
        if before.requires_grad:
            assert not torch.equal(before, after)
            return

def test_vmc_step_with_real_circuit():
    """VMC step on the real 8-qubit circuit should not crash."""
    samples = load_dataset(difficulties=("very_easy",))
    target  = next(s for s in samples if s.challenge == "challenge-8_1")
    model   = _tiny_model()
    inter   = torch.tensor(target.interaction_matrix, dtype=torch.float32)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    mean_r, loss = vmc_step(
        model, inter, target.qasm_path, quimb_amplitude,
        n_qubits=target.n_qubits, n_samples=8, optimizer=opt,
    )
    assert math.isfinite(loss)


# ── vmc_train ─────────────────────────────────────────────────────────────────

def test_vmc_train_history_keys():
    model = _tiny_model()
    mock  = make_peaked_mock("10101010")
    samples = load_dataset(difficulties=("very_easy",))
    history = vmc_train(model, samples[:2], mock, n_epochs=3, n_samples=8, log_every=0)
    assert set(history.keys()) >= {"epochs", "mean_rewards", "losses"}

def test_vmc_train_history_length():
    model = _tiny_model()
    mock  = make_peaked_mock("10101010")
    samples = load_dataset(difficulties=("very_easy",))
    history = vmc_train(model, samples[:2], mock, n_epochs=4, n_samples=8, log_every=0)
    assert len(history["epochs"]) == 4
    assert len(history["losses"]) == 4
    assert len(history["mean_rewards"]) == 4

def test_vmc_train_losses_are_finite():
    model = _tiny_model()
    mock  = make_peaked_mock("11110000")
    samples = load_dataset(difficulties=("very_easy",))
    history = vmc_train(model, samples[:2], mock, n_epochs=3, n_samples=8, log_every=0)
    assert all(math.isfinite(l) for l in history["losses"])

def test_vmc_reward_improves_on_peaked_mock():
    """
    After enough VMC steps with a sharply peaked mock amplitude, the model
    should assign increasing probability to the target bitstring.
    """
    torch.manual_seed(5)
    TARGET = "11111111"
    mock   = make_peaked_mock(TARGET, peak_prob=0.95, other_prob=1e-6)
    model  = _tiny_model()
    inter  = torch.zeros(8, 8)

    history = vmc_train(model, [], mock, n_epochs=0)  # init history object
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    rewards_early = []
    for _ in range(10):
        r, _ = vmc_step(model, inter, None, mock, 8, 64, opt)
        rewards_early.append(r)

    rewards_late = []
    for _ in range(60):
        r, _ = vmc_step(model, inter, None, mock, 8, 64, opt)
        rewards_late.append(r)

    early_mean = sum(rewards_early) / len(rewards_early)
    late_mean  = sum(rewards_late)  / len(rewards_late)
    assert late_mean > early_mean, (
        f"VMC reward did not improve: {early_mean:.3f} → {late_mean:.3f}"
    )
