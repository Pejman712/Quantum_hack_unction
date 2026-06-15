"""
Decoder-only autoregressive transformer for peaked quantum circuit states.

The model represents:
    p(x₁, x₂, ..., xₙ) = ∏ᵢ p(xᵢ | x₀...xᵢ₋₁, circuit)

Circuit conditioning is handled by CircuitConditioner, which provides
per-layer learnable scales and gate-type decomposition (cx vs swap).
Each transformer block receives its own interaction bias derived from the
circuit's qubit-coupling graph.

Token vocabulary:
    0 → bit 0
    1 → bit 1
    2 → BOS  (beginning-of-sequence, prepended at position 0)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import CircuitConditioner, normalize_interaction

BOS_TOKEN = 2
VOCAB_SIZE = 3  # {0, 1, BOS}


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask, attn_bias=None):
        """
        x          : (B, T, d_model)
        causal_mask: (T, T) bool — True where attention is forbidden
        attn_bias  : (T, T) float — additive bias (circuit interaction prior)
        """
        B, T, _ = x.shape
        H, d_k  = self.n_heads, self.d_k

        Q = self.Wq(x).view(B, T, H, d_k).transpose(1, 2)  # (B,H,T,d_k)
        K = self.Wk(x).view(B, T, H, d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, H, d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B,H,T,T)

        if attn_bias is not None:
            # (T,T) → (1,1,T,T) broadcasts over batch and heads
            scores = scores + attn_bias.unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.drop(attn)

        out = torch.matmul(attn, V)                          # (B,H,T,d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.Wo(out)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn  = _MultiHeadAttention(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, causal_mask, attn_bias=None):
        x = x + self.drop(self.attn(self.norm1(x), causal_mask, attn_bias))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class AutoregressiveTransformer(nn.Module):
    """
    Parameters
    ----------
    d_model    : int   — embedding / hidden dimension
    n_heads    : int   — number of attention heads
    n_layers   : int   — number of transformer blocks
    d_ff       : int   — feed-forward inner dimension
    max_qubits : int   — maximum circuit size (for position embeddings)
    dropout    : float — dropout probability (set 0.0 for inference)
    """

    def __init__(
        self,
        d_model:    int = 128,
        n_heads:    int = 4,
        n_layers:   int = 4,
        d_ff:       int = 256,
        max_qubits: int = 128,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.max_qubits = max_qubits
        self.n_layers   = n_layers

        self.token_embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embed   = nn.Embedding(max_qubits, d_model)

        self.layers = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self.conditioner = CircuitConditioner(n_layers=n_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, n, device):
        """Upper-triangular boolean mask: True = forbidden (future token)."""
        return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    def _resolve_interaction(self, interaction_matrix, cx_matrix, swap_matrix, device, n):
        """
        Resolve whichever interaction API the caller used into a single
        normalised (n, n) bias tensor.

        Priority:
          1. cx_matrix + swap_matrix  → typed API (Sprint 3+)
          2. interaction_matrix        → legacy API (Sprint 1–2)
          3. neither                   → zeros (no circuit conditioning)
        """
        if cx_matrix is not None and swap_matrix is not None:
            combined = self.conditioner.combine(
                cx_matrix.to(device), swap_matrix.to(device)
            )
        elif interaction_matrix is not None:
            combined = interaction_matrix.to(device)
        else:
            combined = torch.zeros(n, n, device=device)

        return normalize_interaction(combined)

    def forward(self, tokens, interaction_matrix=None, *, cx_matrix=None, swap_matrix=None):
        """
        Parameters
        ----------
        tokens             : LongTensor (B, n_qubits)  — target bitstring (values 0/1)
        interaction_matrix : FloatTensor (n_qubits, n_qubits) or None
            Combined circuit coupling counts — legacy API, still supported.
        cx_matrix          : FloatTensor (n_qubits, n_qubits) or None  [keyword-only]
            CNOT gate coupling counts — Sprint 3+ typed API.
        swap_matrix        : FloatTensor (n_qubits, n_qubits) or None  [keyword-only]
            SWAP gate coupling counts — Sprint 3+ typed API.

        Returns
        -------
        probs : FloatTensor (B, n_qubits)
            probs[:, i] = p(xᵢ = 1 | x₀,...,xᵢ₋₁, circuit)
        """
        B, n = tokens.shape
        device = tokens.device

        # Build shifted input: [BOS, x₀, x₁, ..., x_{n-2}]
        bos = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
        inp = torch.cat([bos, tokens[:, :-1]], dim=1)   # (B, n)

        # Embeddings
        positions = torch.arange(n, device=device)
        x = self.token_embed(inp) + self.pos_embed(positions)

        # Causal mask and normalised circuit interaction basis
        mask      = self._causal_mask(n, device)
        norm_bias = self._resolve_interaction(interaction_matrix, cx_matrix, swap_matrix, device, n)

        # Each layer gets its own learned scale of the interaction bias
        for i, layer in enumerate(self.layers):
            attn_bias = self.conditioner.get_layer_bias(norm_bias, i)
            x = layer(x, mask, attn_bias)

        x = self.norm(x)
        logits = self.head(x).squeeze(-1)    # (B, n)
        return torch.sigmoid(logits)
