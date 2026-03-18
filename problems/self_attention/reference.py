"""Reference: multi-head self-attention implemented from scratch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionRef(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        B, S, D = x.shape

        # project to Q, K, V
        q = self.W_q(x)  # (B, S, D)
        k = self.W_k(x)
        v = self.W_v(x)

        # reshape to (B, num_heads, S, d_k)
        q = q.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, S, S)
        attn = F.softmax(scores, dim=-1)  # (B, H, S, S)
        context = torch.matmul(attn, v)  # (B, H, S, d_k)

        # concat heads and project
        context = context.transpose(1, 2).contiguous().view(B, S, D)  # (B, S, D)
        output = self.W_o(context)  # (B, S, D)
        return output
