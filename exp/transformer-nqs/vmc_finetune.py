#!/usr/bin/env python3
"""
VMC fine-tuning script for hard circuits.

Loads a pre-trained checkpoint (from train.py) and continues training
using REINFORCE with quimb circuit amplitudes as reward signal.

Usage:
    python exp/transformer-nqs/vmc_finetune.py \
        --checkpoint checkpoints/nqs/final.pt \
        --difficulties hard \
        --n-epochs 200 \
        --n-samples 64 \
        --lr 1e-4 \
        --device cuda \
        --out-dir checkpoints/nqs_vmc

Resume after a timeout:
    python exp/transformer-nqs/vmc_finetune.py \
        --checkpoint checkpoints/nqs_vmc/vmc_epoch_0099.pt \
        --difficulties hard \
        --n-epochs 200 \
        --out-dir checkpoints/nqs_vmc
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import load_dataset
from model.transformer import AutoregressiveTransformer
from training.checkpointing import load_checkpoint, save_checkpoint
from training.supervised import TrainingHistory
from training.amplitude import quimb_amplitude
from training.vmc import vmc_step


def _parse_args():
    p = argparse.ArgumentParser(description="VMC fine-tuning for hard circuits")

    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Pre-trained or previously VMC-checkpointed .pt file")
    p.add_argument("--difficulties", nargs="+", default=["hard"],
                   choices=["easy", "moderate", "hard", "very_hard"],
                   help="Which circuit tiers to fine-tune on")
    p.add_argument("--n-epochs",  type=int,   default=200)
    p.add_argument("--n-samples", type=int,   default=64,
                   help="Monte Carlo sample budget per circuit per step")
    p.add_argument("--lr",        type=float, default=1e-4,
                   help="Learning rate (lower than supervised to avoid forgetting)")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",   type=Path,  default=Path("checkpoints/nqs_vmc"))
    p.add_argument("--save-every",type=int,   default=50,
                   help="Save VMC checkpoint every N epochs")
    p.add_argument("--log-every", type=int,   default=10)

    return p.parse_args()


def main():
    args = _parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    # We need model config to reconstruct it — pull from checkpoint
    ckpt_raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config   = ckpt_raw.get("config", {})

    model = AutoregressiveTransformer(
        d_model    = config.get("d_model",    256),
        n_heads    = config.get("n_heads",    8),
        n_layers   = config.get("n_layers",   6),
        d_ff       = config.get("d_ff",       512),
        max_qubits = config.get("max_qubits", 128),
        dropout    = config.get("dropout",    0.0),   # no dropout during VMC
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    history     = {"epochs": [], "mean_rewards": [], "losses": []}

    # Check whether this is a supervised checkpoint or a VMC one
    if "vmc_history" in ckpt_raw:
        # Resuming from a previous VMC checkpoint
        model.load_state_dict(ckpt_raw["model_state"])
        optimizer.load_state_dict(ckpt_raw["optimizer_state"])
        start_epoch = ckpt_raw["epoch"] + 1
        history     = ckpt_raw["vmc_history"]
        print(f"Resumed VMC from {args.checkpoint} (epoch {start_epoch - 1})")
    else:
        # Supervised checkpoint — just load weights, fresh optimizer
        _, _, _ = load_checkpoint(args.checkpoint, model)
        print(f"Loaded supervised checkpoint: {args.checkpoint}")

    model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}   device: {args.device}")

    # ── load data ─────────────────────────────────────────────────────────────
    all_samples = []
    for d in args.difficulties:
        s = load_dataset(difficulties=(d,))
        print(f"  {d}: {len(s)} circuits")
        all_samples.extend(s)

    if not all_samples:
        raise RuntimeError(
            "No samples found. Hard/very_hard circuits require ground truth CSVs "
            "or can be run without them — but the QASM files must exist."
        )

    print(f"Total circuits: {len(all_samples)}   n_samples/step: {args.n_samples}")
    print(f"VMC epochs {start_epoch}–{start_epoch + args.n_epochs - 1}")

    # ── VMC loop ──────────────────────────────────────────────────────────────
    import random
    for epoch in range(start_epoch, start_epoch + args.n_epochs):
        epoch_rewards = []
        epoch_losses  = []

        random.shuffle(all_samples)
        for s in all_samples:
            inter = torch.tensor(s.interaction_matrix, dtype=torch.float32, device=args.device)
            mean_r, loss_val = vmc_step(
                model, inter, s.qasm_path, quimb_amplitude,
                s.n_qubits, args.n_samples, optimizer, device=args.device,
            )
            epoch_rewards.append(mean_r)
            epoch_losses.append(loss_val)

        finite  = [r for r in epoch_rewards if math.isfinite(r)]
        epoch_r = sum(finite) / len(finite) if finite else float("-inf")
        epoch_l = sum(epoch_losses) / len(epoch_losses)

        history["epochs"].append(epoch)
        history["mean_rewards"].append(epoch_r)
        history["losses"].append(epoch_l)

        if args.log_every and epoch % args.log_every == 0:
            print(f"VMC epoch {epoch:4d} | reward {epoch_r:+.4f} | loss {epoch_l:.4f}")

        # Periodic VMC checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = args.out_dir / f"vmc_epoch_{epoch:04d}.pt"
            _save_vmc_checkpoint(ckpt_path, model, optimizer, epoch, history, config)
            print(f"  → saved {ckpt_path}")

    # ── final checkpoint ──────────────────────────────────────────────────────
    final_path = args.out_dir / "vmc_final.pt"
    _save_vmc_checkpoint(final_path, model, optimizer,
                         start_epoch + args.n_epochs - 1, history, config)
    print(f"Saved: {final_path}")

    if history["mean_rewards"]:
        finite = [r for r in history["mean_rewards"] if math.isfinite(r)]
        if finite:
            print(f"Best reward: {max(finite):+.4f}   Last: {history['mean_rewards'][-1]:+.4f}")

    # Save human-readable reward log
    log_path = args.out_dir / "vmc_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History: {log_path}")


def _save_vmc_checkpoint(path, model, optimizer, epoch, history, config):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vmc_history":     history,
        "config":          config,
    }, path)


if __name__ == "__main__":
    main()
