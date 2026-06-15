"""
VMC (Variational Monte Carlo) fine-tuning for hard circuits.

For circuits with no ground-truth bitstring we cannot use supervised BCE.
Instead we treat the circuit amplitude |⟨x|ψ⟩|² as the reward signal and
use the REINFORCE policy-gradient estimator:

    ∇E_{x~p_θ}[log|A(x)|²] = E_x[(log|A(x)|² − b) · ∇log p_θ(x)]

where b = mean(log|A|²) over the batch is a variance-reducing baseline.

Typical workflow
----------------
1. Pre-train on very_easy / easy / moderate with supervised BCE (Sprints 2-4)
2. Load the pre-trained checkpoint
3. Fine-tune on hard circuits with `vmc_train()` using a lower lr (≈1e-4)
"""

import math
import random

import torch
import torch.nn.functional as F

from model.sampler import sample as autoregressive_sample


def compute_log_prob(model, tokens, interaction_matrix):
    """
    Compute log p_θ(x) for a batch of bitstrings, WITH gradient.

    This is separate from the sampler (which runs under no_grad) so that
    REINFORCE can differentiate through the log probability.

    Parameters
    ----------
    model              : AutoregressiveTransformer
    tokens             : LongTensor (B, n_qubits)  — sampled bitstrings {0,1}
    interaction_matrix : FloatTensor (n_qubits, n_qubits)

    Returns
    -------
    log_p : FloatTensor (B,)  — log p_θ(x) ≤ 0 for each sample
    """
    probs   = model(tokens, interaction_matrix)   # (B, n)
    targets = tokens.float()
    log_p_per_bit = (
        targets       * torch.log(probs.clamp(min=1e-10)) +
        (1 - targets) * torch.log((1 - probs).clamp(min=1e-10))
    )
    return log_p_per_bit.sum(dim=-1)             # (B,)


def reinforce_loss(log_probs, rewards, use_baseline=True):
    """
    REINFORCE loss (negated expected reward) with optional variance-reducing
    baseline.

    Parameters
    ----------
    log_probs    : FloatTensor (n_samples,) — log p_θ(x) for each sample
                   Must have requires_grad=True so backward() works.
    rewards      : FloatTensor (n_samples,) — log|A(x)|² for each sample
    use_baseline : bool — subtract mean reward to reduce gradient variance

    Returns
    -------
    loss : scalar Tensor  (minimize this)
    """
    if use_baseline:
        baseline   = rewards.mean().detach()
        advantages = rewards - baseline
    else:
        advantages = rewards
    return -(advantages * log_probs).mean()


def vmc_step(model, interaction_matrix, qasm_path, amplitude_fn,
             n_qubits, n_samples, optimizer, device="cpu"):
    """
    One VMC update: sample → evaluate amplitudes → REINFORCE gradient step.

    Parameters
    ----------
    model              : AutoregressiveTransformer
    interaction_matrix : FloatTensor (n_qubits, n_qubits)
    qasm_path          : str or Path (passed to amplitude_fn unchanged)
    amplitude_fn       : callable(qasm_path, bitstring_str) -> float
                         Returns log|A(x)|². Use training.amplitude.quimb_amplitude
                         for real circuits, or make_peaked_mock() for tests.
    n_qubits           : int
    n_samples          : int — Monte Carlo sample budget
    optimizer          : torch.optim.Optimizer
    device             : str

    Returns
    -------
    mean_reward : float — mean log|A|² over the samples (excluding -inf)
    loss_value  : float — REINFORCE loss
    """
    inter = interaction_matrix.to(device)

    # 1. Sample bitstrings without grad
    model.eval()
    with torch.no_grad():
        bitstrings = autoregressive_sample(model, inter, n_qubits, n_samples)
        # bitstrings: LongTensor (n_samples, n_qubits)

    # 2. Evaluate amplitudes (no grad needed)
    rewards_list = []
    for bs in bitstrings:
        bs_str = "".join(str(b.item()) for b in bs)
        log_a  = amplitude_fn(qasm_path, bs_str)
        rewards_list.append(log_a)
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

    # 3. Re-compute log p_θ(x) WITH gradient for the sampled tokens
    model.train()
    log_probs = compute_log_prob(model, bitstrings.to(device), inter)

    # 4. Mask out -inf rewards (zero-amplitude bitstrings):
    #    replace them with the minimum finite reward so they get zero advantage
    finite_mask = rewards.isfinite()
    if finite_mask.any():
        min_finite = rewards[finite_mask].min().detach()
        rewards    = torch.where(finite_mask, rewards, min_finite)

    loss = reinforce_loss(log_probs, rewards.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Compute mean reward over finite samples only
    finite_rewards = [r for r in rewards_list if math.isfinite(r)]
    mean_reward = sum(finite_rewards) / len(finite_rewards) if finite_rewards else float("-inf")

    return mean_reward, loss.item()


def vmc_train(model, hard_samples, amplitude_fn, n_epochs, n_samples=64,
              lr=1e-4, device="cpu", log_every=10):
    """
    VMC fine-tuning loop.

    Typically called after `training.supervised.train()` to continue training
    on hard circuits where no ground truth exists.  Uses a lower learning rate
    (default 1e-4 vs 1e-3 for supervised) to avoid catastrophic forgetting of
    the patterns learned on easy/moderate circuits.

    Parameters
    ----------
    model        : AutoregressiveTransformer — pre-trained
    hard_samples : list[CircuitSample] — each sample needs .qasm_path,
                   .interaction_matrix, .n_qubits.  .bitstring is ignored.
    amplitude_fn : callable(qasm_path, bitstring_str) -> float
    n_epochs     : int
    n_samples    : int — MC samples per circuit per step
    lr           : float
    device       : str
    log_every    : int — 0 = silent

    Returns
    -------
    history : dict with keys "epochs", "mean_rewards", "losses"
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history   = {"epochs": [], "mean_rewards": [], "losses": []}

    for epoch in range(n_epochs):
        epoch_rewards = []
        epoch_losses  = []

        order = list(range(len(hard_samples)))
        random.shuffle(order)

        for idx in order:
            s     = hard_samples[idx]
            inter = torch.tensor(s.interaction_matrix, dtype=torch.float32, device=device)
            mean_r, loss_val = vmc_step(
                model, inter, s.qasm_path, amplitude_fn,
                s.n_qubits, n_samples, optimizer, device=device,
            )
            epoch_rewards.append(mean_r)
            epoch_losses.append(loss_val)

        finite = [r for r in epoch_rewards if math.isfinite(r)]
        epoch_r = sum(finite) / len(finite) if finite else float("-inf")
        epoch_l = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")

        history["epochs"].append(epoch)
        history["mean_rewards"].append(epoch_r)
        history["losses"].append(epoch_l)

        if log_every and epoch % log_every == 0:
            print(f"VMC epoch {epoch:4d} | reward {epoch_r:+.4f} | loss {epoch_l:.4f}")

    return history
