"""Flash Attention

Part 1 — Online Softmax
========================
Implement softmax computed in a *single pass* over the last dimension,
using the online (streaming) algorithm:
  - Maintain a running max `m` and a running sum-of-exponentials `d`.
  - For each element x_j:
        m_new = max(m, x_j)
        d     = d * exp(m - m_new) + exp(x_j - m_new)
        m     = m_new
  - Final result: exp(x - m) / d

Args:
    x: (..., N) — logits.
Returns:
    softmax(x), same shape as x.


Part 2 — Flash Attention Forward
==================================
Implement the Flash Attention forward pass (Algorithm 1):
  - Tile Q into blocks of size `block_size_q`.
  - For each Q-block, iterate over K/V blocks of size `block_size_kv`.
  - Use the online softmax trick to accumulate attention output
    without materialising the full N×N score matrix.

Args:
    Q: (batch, seq_len, d_head)
    K: (batch, seq_len, d_head)
    V: (batch, seq_len, d_head)
    block_size_q:  tile size for queries.
    block_size_kv: tile size for keys/values.
Returns:
    O: (batch, seq_len, d_head) — attention output.
"""

import math
import torch


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    # TODO: implement online (single-pass) softmax
    raise NotImplementedError


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size_q: int = 32,
    block_size_kv: int = 32,
) -> torch.Tensor:
    # TODO: implement flash attention forward pass
    raise NotImplementedError
