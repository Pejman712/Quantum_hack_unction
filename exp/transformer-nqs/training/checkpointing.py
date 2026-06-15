"""
Checkpoint save/load for AutoregressiveTransformer training.

Checkpoint format (single .pt file):
    epoch          : int   — last completed epoch (0-indexed)
    model_state    : dict  — model.state_dict()
    optimizer_state: dict  — optimizer.state_dict()
    history        : dict  — {"epochs": [...], "losses": [...]}
    config         : dict  — model hyperparameters for reconstruction

Usage
-----
    from training.checkpointing import save_checkpoint, load_checkpoint

    # Save after epoch N
    save_checkpoint(path, model, optimizer, epoch=N, history=history, config=cfg)

    # Resume
    model     = AutoregressiveTransformer(**cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch, history, cfg = load_checkpoint(path, model, optimizer)
    train(model, ..., start_epoch=start_epoch + 1, existing_history=history)
"""

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch, history, config):
    """
    Persist full training state to a .pt file.

    Parameters
    ----------
    path      : str or Path
    model     : AutoregressiveTransformer
    optimizer : torch.optim.Optimizer
    epoch     : int — the epoch that just completed (0-indexed)
    history   : TrainingHistory
    config    : dict — model constructor kwargs, stored for later reconstruction
    """
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": {
                "epochs": list(history.epochs),
                "losses": list(history.losses),
            },
            "config": dict(config),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    """
    Restore training state from a checkpoint file.

    Loads model weights in-place.  If optimizer is provided, also restores
    its state (momentum buffers, step counts, etc.) so training resumes
    identically to where it stopped.

    Parameters
    ----------
    path      : str or Path
    model     : AutoregressiveTransformer — weights restored in-place
    optimizer : torch.optim.Optimizer or None

    Returns
    -------
    epoch   : int              — last completed epoch
    history : TrainingHistory  — full loss log up to this checkpoint
    config  : dict             — model hyperparameters stored at save time
    """
    from .supervised import TrainingHistory  # lazy import: avoids circular dependency

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    history        = TrainingHistory()
    history.epochs = list(ckpt["history"]["epochs"])
    history.losses = list(ckpt["history"]["losses"])

    return ckpt["epoch"], history, ckpt.get("config", {})
