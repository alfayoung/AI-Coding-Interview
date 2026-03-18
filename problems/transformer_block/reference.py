"""Reference: Transformer encoder block (pre-norm) implemented from scratch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlockRef(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # self-attention projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # feed-forward network
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        # layer norms (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(context)

    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(F.gelu(self.ff1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm: norm -> sublayer -> residual add
        x = x + self._self_attention(self.norm1(x))
        x = x + self._feed_forward(self.norm2(x))
        return x
