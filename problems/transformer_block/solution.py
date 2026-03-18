"""Transformer Block

Implement a single Transformer encoder block with:
- Multi-head self-attention
- Add & LayerNorm (pre-norm or post-norm, your choice)
- Feed-forward network (2-layer MLP with GELU)
- Add & LayerNorm

Args:
    x: (batch, seq_len, d_model)

Returns:
    output: (batch, seq_len, d_model)
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        raise NotImplementedError("Implement this!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement this!")
