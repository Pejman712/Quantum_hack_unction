"""
Circuit conditioning for AutoregressiveTransformer.

Instead of a single learnable scalar shared across all layers, CircuitConditioner
provides:
  - Per-layer scales so early vs. late layers can weight the circuit prior differently
  - Separate cx and swap weights because the two gate types encode different physics
    (cx creates entanglement; swap moves entanglement without creating it)
  - log1p normalization so raw gate counts (which range 0–100+) don't dominate
    attention scores regardless of circuit depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_interaction(m):
    """
    Compress an interaction matrix into [0, 1] using log1p scaling.

    log1p is used rather than dividing by the raw max because raw gate
    counts grow with circuit depth and would otherwise push attention
    scores into saturation on deeper circuits.

    Parameters
    ----------
    m : FloatTensor (n, n) — symmetric, non-negative interaction counts

    Returns
    -------
    out : FloatTensor (n, n) — same shape, values in [0, 1]
    """
    out = torch.log1p(m)
    max_val = out.max()
    if max_val > 0:
        out = out / max_val
    return out


class CircuitConditioner(nn.Module):
    """
    Learnable module that converts a circuit's gate-coupling graph into
    per-layer attention biases for the transformer.

    Parameters
    ----------
    n_layers : int — must match the transformer's n_layers
    """

    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers

        # Each transformer layer gets its own scale, initialised to 0 so the
        # model starts as an unconditioned transformer and learns to incorporate
        # circuit structure through gradient descent.
        self.layer_scales = nn.Parameter(torch.zeros(n_layers))

        # cx and swap gates have different physical roles: cx creates
        # entanglement, swap only moves it.  Separate learnable weights let the
        # model discover how much each gate type should bias attention.
        # Initialised to 1.0 (not 0) so gradients flow from the first step.
        self.cx_weight   = nn.Parameter(torch.ones(1))
        self.swap_weight = nn.Parameter(torch.ones(1))

    def combine(self, cx_matrix, swap_matrix):
        """
        Weighted sum of the two gate-type matrices.

        softplus ensures effective weights are always positive — a cx gate
        should never anti-correlate attention, regardless of the optimizer.

        Parameters
        ----------
        cx_matrix   : FloatTensor (n, n)
        swap_matrix : FloatTensor (n, n)

        Returns
        -------
        combined : FloatTensor (n, n)
        """
        cx_w   = F.softplus(self.cx_weight)
        swap_w = F.softplus(self.swap_weight)
        return cx_w * cx_matrix + swap_w * swap_matrix

    def get_layer_bias(self, normalized_bias, layer_idx):
        """
        Scale a normalised interaction matrix for a specific layer.

        Parameters
        ----------
        normalized_bias : FloatTensor (n, n) — output of normalize_interaction
        layer_idx       : int — which transformer layer (0-based)

        Returns
        -------
        bias : FloatTensor (n, n)
        """
        return self.layer_scales[layer_idx] * normalized_bias
