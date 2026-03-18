import torch
import pytest
from problems.transformer_block.solution import TransformerBlock
from problems.transformer_block.reference import TransformerBlockRef
from utils.check import check_close

B, SEQ_LEN, D_MODEL, NUM_HEADS, D_FF = 2, 8, 64, 8, 256


@pytest.fixture
def inputs():
    torch.manual_seed(42)
    return torch.randn(B, SEQ_LEN, D_MODEL)


def test_output_shape(inputs):
    model = TransformerBlock(D_MODEL, NUM_HEADS, D_FF)
    out = model(inputs)
    assert out.shape == (B, SEQ_LEN, D_MODEL)


def test_matches_reference(inputs):
    torch.manual_seed(0)
    ref = TransformerBlockRef(D_MODEL, NUM_HEADS, D_FF)
    sol = TransformerBlock(D_MODEL, NUM_HEADS, D_FF)
    sol.load_state_dict(ref.state_dict())

    ref_out = ref(inputs)
    sol_out = sol(inputs)
    check_close(sol_out, ref_out, atol=1e-5, rtol=1e-5)
