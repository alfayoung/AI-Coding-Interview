import torch


def check_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str = "output",
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    """Assert two tensors match in shape and values."""
    assert actual.shape == expected.shape, (
        f"{name} shape mismatch: got {actual.shape}, expected {expected.shape}"
    )
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs()
        raise AssertionError(
            f"{name} values mismatch — max diff: {diff.max().item():.6e}, "
            f"mean diff: {diff.mean().item():.6e}"
        )
