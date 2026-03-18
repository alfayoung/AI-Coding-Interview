"""Reference: multi-head cross-attention implemented from scratch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionRef(nn.Module):
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

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, S_q, D), kv: (B, S_kv, D)
        B, S_q, D = q.shape
        S_kv = kv.shape[1]

        # project
        q = self.W_q(q)   # (B, S_q, D)
        k = self.W_k(kv)  # (B, S_kv, D)
        v = self.W_v(kv)  # (B, S_kv, D)

        # reshape to (B, num_heads, S, d_k)
        q = q.view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, S_q, S_kv)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # (B, H, S_q, d_k)

        # concat heads and project
        context = context.transpose(1, 2).contiguous().view(B, S_q, D)
        output = self.W_o(context)
        return output
