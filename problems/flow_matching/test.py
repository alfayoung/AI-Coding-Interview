import torch
import torch.nn as nn
import pytest
from problems.flow_matching.solution import FlowMatching
from problems.flow_matching.reference import FlowMatchingRef
from utils.check import check_close

D = 32


class DummyVelocityModel(nn.Module):
    """Simple MLP velocity predictor for testing."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_feat = t.view(-1, 1) if t.dim() == 1 else t
        return self.net(torch.cat([x, t_feat], dim=-1))


@pytest.fixture
def shared_model():
    torch.manual_seed(42)
    return DummyVelocityModel(D)


@pytest.fixture
def inputs():
    torch.manual_seed(123)
    return torch.randn(4, D)


def test_interpolate(shared_model, inputs):
    ref = FlowMatchingRef(shared_model)
    sol = FlowMatching(shared_model)

    x1 = torch.randn_like(inputs)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])

    ref_out = ref.interpolate(inputs, x1, t)
    sol_out = sol.interpolate(inputs, x1, t)
    check_close(sol_out, ref_out, name="interpolate")


def test_compute_loss(shared_model, inputs):
    sol = FlowMatching(shared_model)
    loss = sol.compute_loss(inputs)
    assert loss.shape == ()
    assert loss.item() > 0
