import torch
import pytest
from problems.cross_attention.solution import CrossAttention
from problems.cross_attention.reference import CrossAttentionRef
from utils.check import check_close

B, SEQ_Q, SEQ_KV, D_MODEL, NUM_HEADS = 2, 4, 6, 64, 8


@pytest.fixture
def inputs():
    torch.manual_seed(42)
    q = torch.randn(B, SEQ_Q, D_MODEL)
    kv = torch.randn(B, SEQ_KV, D_MODEL)
    return q, kv


def test_output_shape(inputs):
    q, kv = inputs
    model = CrossAttention(D_MODEL, NUM_HEADS)
    out = model(q, kv)
    assert out.shape == (B, SEQ_Q, D_MODEL)


def test_matches_reference(inputs):
    q, kv = inputs
    torch.manual_seed(0)
    ref = CrossAttentionRef(D_MODEL, NUM_HEADS)
    sol = CrossAttention(D_MODEL, NUM_HEADS)
    sol.load_state_dict(ref.state_dict())

    ref_out = ref(q, kv)
    sol_out = sol(q, kv)
    check_close(sol_out, ref_out, atol=1e-5, rtol=1e-5)
