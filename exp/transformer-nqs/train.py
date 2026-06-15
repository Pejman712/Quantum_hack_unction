#!/usr/bin/env python3
"""
Standalone training script for AutoregressiveTransformer NQS.

Train from scratch:
    python exp/transformer-nqs/train.py \
        --difficulties very_easy easy moderate \
        --n-epochs 500 \
        --lr 1e-3 \
        --d-model 256 --n-heads 8 --n-layers 6 --d-ff 512 \
        --device cuda \
        --checkpoint-dir checkpoints/nqs \
        --checkpoint-every 50

Resume after a Puhti timeout:
    python exp/transformer-nqs/train.py \
        --resume-from checkpoints/nqs/epoch_0099.pt \
        --n-epochs 500 \
        --checkpoint-dir checkpoints/nqs
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import load_dataset
from model.transformer import AutoregressiveTransformer
from training.supervised import train
from training.curriculum import CurriculumScheduler
from training.checkpointing import save_checkpoint, load_checkpoint


def _parse_args():
    p = argparse.ArgumentParser(description="Train NQS Transformer")

    # Data
    p.add_argument(
        "--difficulties", nargs="+",
        default=["very_easy", "easy", "moderate"],
        choices=["very_easy", "easy", "moderate", "hard", "very_hard"],
        help="Difficulty tiers to include (cumulative curriculum in this order)",
    )

    # Training
    p.add_argument("--n-epochs",  type=int,   default=500)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int,   default=10)

    # Model architecture
    p.add_argument("--d-model",    type=int,   default=256)
    p.add_argument("--n-heads",    type=int,   default=8)
    p.add_argument("--n-layers",   type=int,   default=6)
    p.add_argument("--d-ff",       type=int,   default=512)
    p.add_argument("--max-qubits", type=int,   default=128)
    p.add_argument("--dropout",    type=float, default=0.1)

    # Checkpointing
    p.add_argument("--checkpoint-dir",   type=Path, default=Path("checkpoints/nqs"))
    p.add_argument("--checkpoint-every", type=int,  default=50)
    p.add_argument("--resume-from",      type=Path, default=None,
                   help="Path to a .pt checkpoint file to resume from")

    return p.parse_args()


def _build_curriculum(difficulties, n_epochs):
    """
    Load each difficulty tier and distribute epochs evenly (cumulative stages).

    If a difficulty has no data (e.g. hard circuits haven't been solved yet),
    it is skipped with a warning rather than crashing.
    """
    loaded = {}
    for d in difficulties:
        s = load_dataset(difficulties=(d,))
        if s:
            loaded[d] = s
            print(f"  loaded {d}: {len(s)} circuits")
        else:
            print(f"  WARNING: no samples for '{d}' — skipping")

    if not loaded:
        raise RuntimeError("No data loaded. Check that CSV files exist and have success rows.")

    n = len(loaded)
    n_per = n_epochs // n
    stages = []
    for i, (d, s) in enumerate(loaded.items()):
        epochs_this_stage = n_per if i < n - 1 else n_epochs - i * n_per
        stages.append((d, s, epochs_this_stage))
        print(f"  stage {i}: {d} — {epochs_this_stage} epochs")

    return CurriculumScheduler(stages)


def main():
    args = _parse_args()
    device = args.device
    print(f"Device: {device}")

    config = {
        "d_model":    args.d_model,
        "n_heads":    args.n_heads,
        "n_layers":   args.n_layers,
        "d_ff":       args.d_ff,
        "max_qubits": args.max_qubits,
        "dropout":    args.dropout,
    }

    model     = AutoregressiveTransformer(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch      = 0
    existing_history = None

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
        start_epoch, existing_history, ckpt_config = load_checkpoint(
            args.resume_from, model, optimizer
        )
        start_epoch += 1  # resume from the next epoch
        # Merge stored config so architecture is consistent
        for k in ("d_model", "n_heads", "n_layers", "d_ff", "max_qubits"):
            if k in ckpt_config:
                config[k] = ckpt_config[k]
        print(f"Resumed from {args.resume_from} — continuing from epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    print("Building curriculum:")
    scheduler = _build_curriculum(args.difficulties, args.n_epochs)

    n_remaining = args.n_epochs - start_epoch
    if n_remaining <= 0:
        print(f"Already completed {args.n_epochs} epochs. Nothing to do.")
        return

    print(f"Training epochs {start_epoch}–{args.n_epochs - 1} on {device}")
    history = train(
        model,
        samples=None,
        n_epochs=n_remaining,
        lr=args.lr,
        device=device,
        log_every=args.log_every,
        scheduler=scheduler,
        start_epoch=start_epoch,
        optimizer=optimizer,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        config=config,
        existing_history=existing_history,
    )

    # Always save a final checkpoint
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.checkpoint_dir / "final.pt"
    save_checkpoint(final_path, model, optimizer, args.n_epochs - 1, history, config)
    print(f"Saved: {final_path}")
    print(f"Best loss: {history.best_loss():.4f}   Last loss: {history.last_loss():.4f}")


if __name__ == "__main__":
    main()
