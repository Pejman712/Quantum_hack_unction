"""
Supervised training for AutoregressiveTransformer.

Objective: maximise log p(x* | circuit) for every ground-truth (circuit, x*)
pair in the dataset.  This is binary cross-entropy between the model's per-qubit
probabilities and the ground-truth bit values.

    loss = -Σᵢ [ bᵢ·log p(xᵢ=1) + (1-bᵢ)·log p(xᵢ=0) ]

Each CircuitSample is processed individually (batch size 1) because circuits
have different qubit counts and therefore different interaction matrix shapes.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F


def compute_loss(model, tokens, interaction_matrix):
    """
    Binary cross-entropy loss for one circuit sample.

    Parameters
    ----------
    model              : AutoregressiveTransformer
    tokens             : LongTensor (1, n_qubits)  — ground-truth bits {0,1}
    interaction_matrix : FloatTensor (n_qubits, n_qubits)

    Returns
    -------
    loss : scalar Tensor
    """
    probs = model(tokens, interaction_matrix)          # (1, n_qubits)
    target = tokens.float()                            # (1, n_qubits) in {0.0, 1.0}
    return F.binary_cross_entropy(probs, target)


def train_epoch(model, optimizer, samples, device="cpu"):
    """
    One full pass over all samples.

    Parameters
    ----------
    model     : AutoregressiveTransformer
    optimizer : torch.optim.Optimizer
    samples   : list[CircuitSample]
    device    : str

    Returns
    -------
    mean_loss : float
    """
    model.train()
    order = list(range(len(samples)))
    random.shuffle(order)

    total_loss = 0.0
    for idx in order:
        s = samples[idx]
        tokens      = torch.tensor(s.tokens, dtype=torch.long,  device=device).unsqueeze(0)
        interaction = torch.tensor(s.interaction_matrix, dtype=torch.float32, device=device)

        loss = compute_loss(model, tokens, interaction)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(samples)


@dataclass
class TrainingHistory:
    """Lightweight training log."""
    epochs: list = field(default_factory=list)
    losses: list = field(default_factory=list)

    def record(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(float(loss))

    def best_loss(self):
        return min(self.losses) if self.losses else float("inf")

    def last_loss(self):
        return self.losses[-1] if self.losses else float("inf")


def train(
    model,
    samples,
    n_epochs,
    lr=1e-3,
    device="cpu",
    log_every=10,
    scheduler=None,
    start_epoch=0,
    optimizer=None,
    checkpoint_dir=None,
    checkpoint_every=None,
    config=None,
    existing_history=None,
):
    """
    Full training loop with optional curriculum scheduler and checkpointing.

    Parameters
    ----------
    model            : AutoregressiveTransformer
    samples          : list[CircuitSample] — used when scheduler is None
    n_epochs         : int — number of epochs to run from start_epoch
    lr               : float — only used when optimizer is None
    device           : str
    log_every        : int — print every N epochs (0 = silent)
    scheduler        : CurriculumScheduler or None
    start_epoch      : int — absolute epoch index to start from (for resume)
    optimizer        : torch.optim.Optimizer or None
                       Pass a pre-loaded optimizer when resuming so momentum
                       buffers are preserved.
    checkpoint_dir   : str or Path or None
    checkpoint_every : int or None — save a checkpoint every N epochs
    config           : dict or None — model constructor kwargs stored in the file
    existing_history : TrainingHistory or None — loss log to resume from

    Returns
    -------
    history : TrainingHistory
    """
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = existing_history if existing_history is not None else TrainingHistory()

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    end_epoch = start_epoch + n_epochs
    for epoch in range(start_epoch, end_epoch):
        epoch_samples = scheduler.get_samples(epoch) if scheduler is not None else samples
        loss = train_epoch(model, optimizer, epoch_samples, device=device)
        history.record(epoch, loss)

        if log_every and epoch % log_every == 0:
            stage = scheduler.get_stage(epoch) if scheduler is not None else "—"
            print(f"epoch {epoch:4d} | stage {stage:10s} | loss {loss:.4f}")

        if checkpoint_dir and checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            from .checkpointing import save_checkpoint  # lazy: avoids circular import
            ckpt_path = Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, history, config or {})

    return history
