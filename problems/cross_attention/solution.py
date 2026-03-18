"""Cross-Attention

Implement multi-head cross-attention from scratch using only basic torch operations.

Args:
    q: (batch, seq_q, d_model)   — query from decoder
    kv: (batch, seq_kv, d_model) — key/value from encoder
    num_heads: number of attention heads

Returns:
    output: (batch, seq_q, d_model)
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        raise NotImplementedError("Implement this!")

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement this!")
