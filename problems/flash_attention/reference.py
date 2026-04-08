"""Reference: Flash Attention (forward pass) using online softmax, pure PyTorch."""

import math
import torch


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically-stable softmax computed in a single pass (online).

    Args:
        x: (..., N) — logits along the last dimension.

    Returns:
        softmax(x) with the same shape.
    """
    N = x.shape[-1]
    m = torch.full(x.shape[:-1], float("-inf"), device=x.device, dtype=x.dtype)
    d = torch.zeros_like(m)

    for j in range(N):
        xj = x[..., j]
        m_new = torch.maximum(m, xj)
        d = d * torch.exp(m - m_new) + torch.exp(xj - m_new)
        m = m_new

    return torch.exp(x - m.unsqueeze(-1)) / d.unsqueeze(-1)


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size_q: int = 32,
    block_size_kv: int = 32,
) -> torch.Tensor:
    """Flash Attention forward pass (Algorithm 1 from the paper).

    Computes exact scaled dot-product attention without materialising the
    full N×N attention matrix, by tiling Q, K, V into blocks and using
    the online softmax trick to maintain running statistics.

    Args:
        Q: (batch, seq_len, d_head)
        K: (batch, seq_len, d_head)
        V: (batch, seq_len, d_head)
        block_size_q:  tile size along the query dimension.
        block_size_kv: tile size along the key/value dimension.

    Returns:
        O: (batch, seq_len, d_head) — the attention output.
    """
    B, N, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    O = torch.zeros_like(Q)

    # iterate over query blocks
    for i_start in range(0, N, block_size_q):
        i_end = min(i_start + block_size_q, N)
        Qi = Q[:, i_start:i_end, :]  # (B, Br, D)

        # running statistics for this query block
        mi = torch.full(
            (B, i_end - i_start), float("-inf"), device=Q.device, dtype=Q.dtype
        )  # row-wise max
        li = torch.zeros_like(mi)  # row-wise sum of exp
        Oi = torch.zeros(
            B, i_end - i_start, D, device=Q.device, dtype=Q.dtype
        )  # accumulator

        # iterate over key/value blocks
        for j_start in range(0, N, block_size_kv):
            j_end = min(j_start + block_size_kv, N)
            Kj = K[:, j_start:j_end, :]  # (B, Bc, D)
            Vj = V[:, j_start:j_end, :]  # (B, Bc, D)

            # block attention scores
            Sij = (Qi @ Kj.transpose(-1, -2)) * scale  # (B, Br, Bc)

            # online softmax update
            mij = Sij.max(dim=-1).values  # (B, Br)
            mi_new = torch.maximum(mi, mij)

            # correction factor for previously accumulated values
            alpha = torch.exp(mi - mi_new)  # (B, Br)
            # exp of current block scores
            Pij = torch.exp(Sij - mi_new.unsqueeze(-1))  # (B, Br, Bc)

            li = alpha * li + Pij.sum(dim=-1)
            Oi = alpha.unsqueeze(-1) * Oi + Pij @ Vj

            mi = mi_new

        # normalise
        O[:, i_start:i_end, :] = Oi / li.unsqueeze(-1)

    return O
