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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        
        self.q_proj = nn.Linear(d_model, num_heads * d_head)
        self.k_proj = nn.Linear(d_model, num_heads * d_head)
        self.v_proj = nn.Linear(d_model, num_heads * d_head)
        self.o_proj = nn.Linear(num_heads * d_head, d_model)

        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        

    def attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        assert D == self.d_model, "D != d_model!"
        assert mask is None or mask.shape == (B, L, L), "Invalid mask shape"
        
        # calculate Q, K, V
        # [B, L, num_heads, d_head] -> [B, num_heads, L, d_head]
        q = self.q_proj(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        
        # SDPA
        # [B, num_head, L, L]
        score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_head)
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        score = F.softmax(score, dim=-1)
        
        res = torch.matmul(score, v) # [B, num_head, L, d_head]
        # [B, num_head, L, d_head] -> [B, L, num_head, d_head] -> [B, L, d_model]
        res = res.transpose(1, 2).contiguous().view(B, L, D)
        
        # w_o
        res = self.o_proj(res) # [B, L, D]
        
        return res

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention
        hidden_states = self.pre_norm(x)
        x = x + self.attention(hidden_states, mask)
        
        # ffn
        hidden_states = self.post_norm(x)
        x = x + self.ffn(hidden_states)
        
        return x
