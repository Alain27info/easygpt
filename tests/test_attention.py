"""Tests for docs/part1/attention.md: causal_mask, scaled_dot_product_attention, MultiHeadSelfAttention."""

import pytest
import torch
import torch.nn.functional as F
from reference import mha_reference

from easygpt.attention import MultiHeadSelfAttention, causal_mask, scaled_dot_product_attention


# -- causal mask -------------------------------------------------------------
def test_causal_mask_values():
    mask = causal_mask(4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


# -- scaled dot-product attention --------------------------------------------
def test_sdpa_without_mask_matches_pytorch():
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 7, 8), torch.randn(2, 7, 3)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 5, 3)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v))


def test_sdpa_with_causal_mask_matches_pytorch():
    q, k, v = torch.randn(3, 6, 8), torch.randn(3, 6, 8), torch.randn(3, 6, 8)
    out = scaled_dot_product_attention(q, k, v, causal_mask(6))
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True))


def test_sdpa_4d_with_random_mask_matches_pytorch():
    q, k, v = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 6, 8), torch.randn(2, 4, 6, 8)
    mask = torch.rand(5, 6) > 0.4
    mask[:, 0] = True  # every query attends to at least one key
    out = scaled_dot_product_attention(q, k, v, mask)
    assert out.shape == (2, 4, 5, 8)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask))


def test_sdpa_masked_values_are_ignored():
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    mask = causal_mask(4)
    out = scaled_dot_product_attention(q, k, v, mask)
    v_changed = v.clone()
    v_changed[:, -1] = 1e6  # only the last query may look at the last key
    out_changed = scaled_dot_product_attention(q, k, v_changed, mask)
    torch.testing.assert_close(out[:, :-1], out_changed[:, :-1])


def test_sdpa_uses_scaling_by_sqrt_dk():
    # with identical keys the attention is uniform whatever the scale; use a
    # single query/key pair against a zero key to check the scaling.
    q = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
    k = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    v = torch.tensor([[[1.0], [0.0]]])
    # scores = [4 / sqrt(4), 0] = [2, 0]
    expected = torch.exp(torch.tensor(2.0)) / (torch.exp(torch.tensor(2.0)) + 1)
    out = scaled_dot_product_attention(q, k, v)
    assert out.item() == pytest.approx(expected.item(), rel=1e-5)


# -- multi-head self-attention -----------------------------------------------
def test_mha_parameters():
    mha = MultiHeadSelfAttention(d_model=16, num_heads=4)
    shapes = {name: tuple(p.shape) for name, p in mha.named_parameters()}
    assert shapes == {
        "q_proj.weight": (16, 16),
        "k_proj.weight": (16, 16),
        "v_proj.weight": (16, 16),
        "output_proj.weight": (16, 16),
    }


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_mha_matches_reference(num_heads):
    d_model = 16
    mha = MultiHeadSelfAttention(d_model, num_heads)
    x = torch.randn(3, 7, d_model)
    out = mha(x)
    assert out.shape == (3, 7, d_model)
    expected = mha_reference(
        x, mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight, mha.output_proj.weight, num_heads
    )
    torch.testing.assert_close(out, expected)


def test_mha_is_causal():
    mha = MultiHeadSelfAttention(d_model=16, num_heads=4)
    x = torch.randn(2, 10, 16)
    x_future_changed = x.clone()
    x_future_changed[:, 6:] = torch.randn(2, 4, 16)
    torch.testing.assert_close(mha(x)[:, :6], mha(x_future_changed)[:, :6])


def test_mha_gradients_flow_to_all_projections():
    mha = MultiHeadSelfAttention(d_model=8, num_heads=2)
    mha(torch.randn(2, 5, 8)).sum().backward()
    for name, p in mha.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
