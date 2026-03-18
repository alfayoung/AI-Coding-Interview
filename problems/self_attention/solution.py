"""Self-Attention

Implement multi-head self-attention from scratch using only basic torch operations.

Args:
    x: (batch, seq_len, d_model)
    num_heads: number of attention heads

Returns:
    output: (batch, seq_len, d_model)
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        raise NotImplementedError("Implement this!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement this!")
