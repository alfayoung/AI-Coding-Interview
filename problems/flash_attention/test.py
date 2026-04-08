"""Validation script for flash attention.

Compares the solution's outputs against standard PyTorch attention
across multiple random inputs, sequence lengths, and block sizes.
"""

import math
import torch
import torch.nn.functional as F
from solution import online_softmax, flash_attention


def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Vanilla scaled dot-product attention for reference."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-1, -2)) * scale
    attn = F.softmax(scores, dim=-1)
    return attn @ V


def check_close(actual, expected, name, atol=1e-5, rtol=1e-5):
    assert actual.shape == expected.shape, (
        f"{name} shape mismatch: got {actual.shape}, expected {expected.shape}"
    )
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs()
        raise AssertionError(
            f"{name} values mismatch — max diff: {diff.max().item():.6e}, "
            f"mean diff: {diff.mean().item():.6e}"
        )


def test_online_softmax():
    print("Testing online_softmax ...")
    for shape in [(8,), (4, 16), (2, 5, 32), (2, 3, 64)]:
        x = torch.randn(shape)
        expected = F.softmax(x, dim=-1)
        actual = online_softmax(x)
        check_close(actual, expected, f"online_softmax {shape}", atol=1e-5, rtol=1e-5)
    print("  PASSED")


def test_flash_attention():
    print("Testing flash_attention ...")
    configs = [
        # (batch, seq_len, d_head, block_q, block_kv)
        (1, 16, 8, 4, 4),
        (2, 32, 16, 8, 8),
        (2, 64, 32, 16, 32),
        (4, 128, 64, 32, 32),
        # non-divisible seq_len
        (2, 50, 16, 16, 16),
        (1, 37, 8, 8, 8),
    ]
    for B, N, D, bq, bkv in configs:
        Q = torch.randn(B, N, D)
        K = torch.randn(B, N, D)
        V = torch.randn(B, N, D)
        expected = standard_attention(Q, K, V)
        actual = flash_attention(Q, K, V, block_size_q=bq, block_size_kv=bkv)
        check_close(
            actual, expected,
            f"flash_attn B={B} N={N} D={D} bq={bq} bkv={bkv}",
            atol=1e-4, rtol=1e-4,
        )
    print("  PASSED")


if __name__ == "__main__":
    test_online_softmax()
    test_flash_attention()
    print("\nAll tests passed!")
