import torch
import pytest
from problems.self_attention.solution import SelfAttention
from problems.self_attention.reference import SelfAttentionRef
from utils.check import check_close

B, SEQ_LEN, D_MODEL, NUM_HEADS = 2, 8, 64, 8


@pytest.fixture
def inputs():
    torch.manual_seed(42)
    return torch.randn(B, SEQ_LEN, D_MODEL)


def test_output_shape(inputs):
    model = SelfAttention(D_MODEL, NUM_HEADS)
    out = model(inputs)
    assert out.shape == (B, SEQ_LEN, D_MODEL)


def test_matches_reference(inputs):
    torch.manual_seed(0)
    ref = SelfAttentionRef(D_MODEL, NUM_HEADS)
    sol = SelfAttention(D_MODEL, NUM_HEADS)
    sol.load_state_dict(ref.state_dict())

    ref_out = ref(inputs)
    sol_out = sol(inputs)
    check_close(sol_out, ref_out, atol=1e-5, rtol=1e-5)
