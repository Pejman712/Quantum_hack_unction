"""
Autoregressive sampler for AutoregressiveTransformer.

Generates bitstrings one qubit at a time:
    x₀ ~ p(x₀ | circuit)
    x₁ ~ p(x₁ | x₀, circuit)
    ...

Each step is a single forward pass; positions already sampled are passed as
context and the next position's probability is read from the output.
"""

import torch


@torch.no_grad()
def sample(model, interaction_matrix, n_qubits, n_samples, return_log_probs=False):
    """
    Draw bitstrings autoregressively from the model.

    Parameters
    ----------
    model              : AutoregressiveTransformer
    interaction_matrix : FloatTensor (n_qubits, n_qubits)
    n_qubits           : int
    n_samples          : int
    return_log_probs   : bool
        If True, also return log p(x) for each sampled bitstring.

    Returns
    -------
    bitstrings : LongTensor (n_samples, n_qubits)  — values in {0, 1}
    log_probs  : FloatTensor (n_samples,)           — only if return_log_probs=True
    """
    device    = next(model.parameters()).device
    was_train = model.training
    model.eval()

    tokens   = torch.zeros(n_samples, n_qubits, dtype=torch.long, device=device)
    log_prob = torch.zeros(n_samples, device=device)

    for i in range(n_qubits):
        # Forward pass with tokens sampled so far (future positions are 0, but
        # the causal mask prevents the model from attending to them anyway).
        probs = model(tokens, interaction_matrix)   # (n_samples, n_qubits)
        p_i   = probs[:, i]                         # p(xᵢ = 1 | x₀...xᵢ₋₁)

        sampled = torch.bernoulli(p_i).long()
        tokens[:, i] = sampled

        if return_log_probs:
            log_p_i   = torch.log(p_i.clamp(min=1e-10))
            log_1mp_i = torch.log((1.0 - p_i).clamp(min=1e-10))
            log_prob += torch.where(sampled == 1, log_p_i, log_1mp_i)

    if was_train:
        model.train()

    if return_log_probs:
        return tokens, log_prob
    return tokens
