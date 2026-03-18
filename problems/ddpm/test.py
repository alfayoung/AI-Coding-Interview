import torch
import torch.nn as nn
import pytest
from problems.ddpm.solution import DDPM
from problems.ddpm.reference import DDPMRef
from utils.check import check_close

D = 32
N_STEPS = 100


class DummyEpsModel(nn.Module):
    """Simple MLP noise predictor for testing."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.time_emb = nn.Embedding(N_STEPS, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x + self.time_emb(t))


@pytest.fixture
def shared_eps_model():
    torch.manual_seed(42)
    return DummyEpsModel(D)


@pytest.fixture
def inputs():
    torch.manual_seed(123)
    return torch.randn(4, D)


def test_q_sample(shared_eps_model, inputs):
    ref = DDPMRef(shared_eps_model, n_steps=N_STEPS)
    sol = DDPM(shared_eps_model, n_steps=N_STEPS)

    t = torch.tensor([0, 10, 50, 99])
    noise = torch.randn_like(inputs)

    ref_out = ref.q_sample(inputs, t, noise)
    sol_out = sol.q_sample(inputs, t, noise)
    check_close(sol_out, ref_out, name="q_sample")


def test_compute_loss(shared_eps_model, inputs):
    ref = DDPMRef(shared_eps_model, n_steps=N_STEPS)
    sol = DDPM(shared_eps_model, n_steps=N_STEPS)

    # Both should produce a scalar loss without error
    loss = sol.compute_loss(inputs)
    assert loss.shape == ()
    assert loss.item() > 0
