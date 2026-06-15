"""
Sprint 4 — checkpoint I/O and curriculum-scale training tests.

Run with the simulation venv:
    exp/peak-kremer/peaked-circuit-simulation/.venv/bin/python \
        -m pytest exp/transformer-nqs/tests/test_sprint4.py -v
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.transformer import AutoregressiveTransformer
from data.dataset import load_dataset
from training.checkpointing import save_checkpoint, load_checkpoint
from training.supervised import train, train_epoch, TrainingHistory
from training.curriculum import CurriculumScheduler


def _tiny_model():
    return AutoregressiveTransformer(
        d_model=32, n_heads=2, n_layers=2, d_ff=64, max_qubits=128, dropout=0.0
    )


# ── checkpoint save ────────────────────────────────────────────────────────────

def test_save_checkpoint_creates_file(tmp_path):
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters())
    hist  = TrainingHistory(); hist.record(0, 0.5)
    path  = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=0, history=hist, config={"d_model": 32})
    assert path.exists()


def test_checkpoint_contains_expected_keys(tmp_path):
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters())
    hist  = TrainingHistory()
    path  = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=3, history=hist, config={})
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    assert {"epoch", "model_state", "optimizer_state", "history", "config"} <= set(ckpt.keys())


def test_checkpoint_epoch_stored_correctly(tmp_path):
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters())
    path  = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=42, history=TrainingHistory(), config={})
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 42


def test_checkpoint_history_stored(tmp_path):
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters())
    hist  = TrainingHistory()
    hist.record(0, 0.7); hist.record(1, 0.6); hist.record(2, 0.5)
    path  = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=2, history=hist, config={})
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["history"]["losses"] == pytest.approx([0.7, 0.6, 0.5])


# ── checkpoint load ────────────────────────────────────────────────────────────

def test_load_checkpoint_returns_epoch(tmp_path):
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters())
    hist  = TrainingHistory(); hist.record(7, 0.3)
    path  = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=7, history=hist, config={})

    model2 = _tiny_model()
    epoch, _, _ = load_checkpoint(path, model2)
    assert epoch == 7


def test_load_checkpoint_returns_history(tmp_path):
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters())
    hist  = TrainingHistory()
    hist.record(0, 0.8); hist.record(1, 0.6)
    path  = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=1, history=hist, config={})

    model2 = _tiny_model()
    _, loaded_hist, _ = load_checkpoint(path, model2)
    assert loaded_hist.losses == pytest.approx([0.8, 0.6])
    assert loaded_hist.epochs == [0, 1]


def test_load_checkpoint_restores_model_predictions(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters())
    # Train a little so weights differ from init
    samples = load_dataset(difficulties=("very_easy",))
    for s in samples[:3]:
        tokens = torch.tensor(s.tokens, dtype=torch.long).unsqueeze(0)
        inter  = torch.tensor(s.interaction_matrix, dtype=torch.float32)
        loss   = torch.nn.functional.binary_cross_entropy(
            model(tokens, inter), tokens.float()
        )
        opt.zero_grad(); loss.backward(); opt.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=0, history=TrainingHistory(), config={})

    model2 = _tiny_model()
    load_checkpoint(path, model2)

    tokens = torch.randint(0, 2, (1, 8))
    inter  = torch.zeros(8, 8)
    model.eval(); model2.eval()
    with torch.no_grad():
        p1 = model(tokens, inter)
        p2 = model2(tokens, inter)
    torch.testing.assert_close(p1, p2, atol=1e-6, rtol=0)


def test_load_checkpoint_restores_optimizer_state(tmp_path):
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
    # Do gradient steps so optimizer has non-trivial momentum buffers
    samples = load_dataset(difficulties=("very_easy",))
    for s in samples:
        tokens = torch.tensor(s.tokens, dtype=torch.long).unsqueeze(0)
        inter  = torch.tensor(s.interaction_matrix, dtype=torch.float32)
        loss   = torch.nn.functional.binary_cross_entropy(
            model(tokens, inter), tokens.float()
        )
        opt.zero_grad(); loss.backward(); opt.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=0, history=TrainingHistory(), config={})

    model2 = _tiny_model()
    opt2   = torch.optim.Adam(model2.parameters(), lr=1e-2)
    load_checkpoint(path, model2, opt2)

    # First moment buffers should match after loading
    for s1, s2 in zip(opt.state.values(), opt2.state.values()):
        if "exp_avg" in s1:
            torch.testing.assert_close(s1["exp_avg"], s2["exp_avg"], atol=1e-6, rtol=0)
            break


def test_load_checkpoint_returns_config(tmp_path):
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters())
    cfg   = {"d_model": 32, "n_layers": 2}
    path  = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=0, history=TrainingHistory(), config=cfg)
    _, _, loaded_cfg = load_checkpoint(path, model)
    assert loaded_cfg["d_model"] == 32
    assert loaded_cfg["n_layers"] == 2


# ── train() checkpoint integration ────────────────────────────────────────────

def test_train_saves_checkpoint_at_interval(tmp_path):
    model   = _tiny_model()
    samples = load_dataset(difficulties=("very_easy",))
    train(model, samples, n_epochs=10,
          checkpoint_dir=tmp_path, checkpoint_every=5, config={"d_model": 32})
    # (epoch+1) % 5 == 0  → epoch 4 and epoch 9
    assert (tmp_path / "epoch_0004.pt").exists()
    assert (tmp_path / "epoch_0009.pt").exists()


def test_train_does_not_save_without_checkpoint_dir():
    model   = _tiny_model()
    samples = load_dataset(difficulties=("very_easy",))
    # Should not raise and should not create any files
    train(model, samples, n_epochs=3, checkpoint_dir=None)


def test_checkpoint_file_contains_correct_epoch(tmp_path):
    model   = _tiny_model()
    samples = load_dataset(difficulties=("very_easy",))
    train(model, samples, n_epochs=5,
          checkpoint_dir=tmp_path, checkpoint_every=5, config={})
    ckpt = torch.load(tmp_path / "epoch_0004.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 4


# ── resume ────────────────────────────────────────────────────────────────────

def test_resume_continues_from_correct_epoch(tmp_path):
    """
    Train for 5 epochs (0–4), save checkpoint, then resume for 5 more (5–9).
    The combined history must span epochs 0–9.
    """
    torch.manual_seed(1)
    model   = _tiny_model()
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    samples = load_dataset(difficulties=("very_easy",))

    # Phase 1: train epochs 0–4
    history = train(model, samples, n_epochs=5, optimizer=opt,
                    checkpoint_dir=tmp_path, checkpoint_every=5, config={})
    assert (tmp_path / "epoch_0004.pt").exists()

    # Phase 2: resume from epoch 4 → train epochs 5–9
    model2 = _tiny_model()
    opt2   = torch.optim.Adam(model2.parameters(), lr=1e-3)
    start_epoch, existing_hist, _ = load_checkpoint(
        tmp_path / "epoch_0004.pt", model2, opt2
    )
    assert start_epoch == 4

    history2 = train(model2, samples, n_epochs=5, optimizer=opt2,
                     start_epoch=start_epoch + 1,
                     existing_history=existing_hist, config={})

    assert history2.epochs == list(range(10))
    assert len(history2.losses) == 10


def test_resumed_loss_is_finite(tmp_path):
    torch.manual_seed(2)
    model   = _tiny_model()
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    samples = load_dataset(difficulties=("very_easy",))

    train(model, samples, n_epochs=5, optimizer=opt,
          checkpoint_dir=tmp_path, checkpoint_every=5, config={})

    model2 = _tiny_model()
    opt2   = torch.optim.Adam(model2.parameters(), lr=1e-3)
    start, hist, _ = load_checkpoint(tmp_path / "epoch_0004.pt", model2, opt2)

    history = train(model2, samples, n_epochs=3, optimizer=opt2,
                    start_epoch=start + 1, existing_history=hist, config={})
    assert all(torch.isfinite(torch.tensor(l)) for l in history.losses)


# ── curriculum scale ──────────────────────────────────────────────────────────

def test_curriculum_train_touches_very_easy_and_easy():
    ve = load_dataset(difficulties=("very_easy",))
    ea = load_dataset(difficulties=("easy",))
    scheduler = CurriculumScheduler([
        ("very_easy", ve, 5),
        ("easy",      ea, 5),
    ])

    difficulties_seen = set()
    model = _tiny_model()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        epoch_samples = scheduler.get_samples(epoch)
        for s in epoch_samples:
            difficulties_seen.add(s.difficulty)
        train_epoch(model, opt, epoch_samples)

    assert "very_easy" in difficulties_seen
    assert "easy" in difficulties_seen


def test_curriculum_moderate_stage_includes_all_prior():
    ve = load_dataset(difficulties=("very_easy",))
    ea = load_dataset(difficulties=("easy",))
    mo = load_dataset(difficulties=("moderate",))
    scheduler = CurriculumScheduler([
        ("very_easy", ve, 10),
        ("easy",      ea, 20),
        ("moderate",  mo, 30),
    ])
    # Epoch 30 is the first moderate epoch
    samples = scheduler.get_samples(30)
    diffs   = {s.difficulty for s in samples}
    assert diffs == {"very_easy", "easy", "moderate"}


def test_curriculum_loss_decreases_over_stages():
    """
    Train across a two-stage curriculum for 30 epochs.
    Final loss should be lower than initial loss.
    """
    torch.manual_seed(99)
    ve = load_dataset(difficulties=("very_easy",))
    ea = load_dataset(difficulties=("easy",))
    scheduler = CurriculumScheduler([
        ("very_easy", ve, 15),
        ("easy",      ea, 15),
    ])

    model = _tiny_model()
    history = train(model, samples=None, n_epochs=30,
                    lr=5e-3, scheduler=scheduler)

    assert history.last_loss() < history.losses[0], (
        f"loss did not decrease: {history.losses[0]:.4f} → {history.last_loss():.4f}"
    )
