"""Tests for docs/part1/rope.md: RotaryPositionalEmbedding and attention with RoPE."""

import torch
from reference import mha_reference, rope_reference

from easygpt.attention import MultiHeadSelfAttention, RotaryPositionalEmbedding


def test_rope_has_no_trainable_parameter_and_no_state():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=8, max_seq_len=32)
    assert list(rope.parameters()) == []
    assert len(rope.state_dict()) == 0, "cos/sin should be registered with persistent=False"


def test_rope_buffers_shape():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=8, max_seq_len=32)
    assert rope.cos.shape == (32, 4)
    assert rope.sin.shape == (32, 4)


def test_rope_matches_complex_reference():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=16, max_seq_len=64)
    x = torch.randn(2, 12, 16)
    out = rope(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, rope_reference(x, torch.arange(12), 10000.0))


def test_rope_with_heads_and_custom_positions():
    rope = RotaryPositionalEmbedding(theta=100.0, d_k=8, max_seq_len=64)
    x = torch.randn(2, 3, 5, 8)  # (batch, heads, seq, d_k)
    positions = torch.tensor([3, 10, 11, 40, 63])
    torch.testing.assert_close(rope(x, positions), rope_reference(x, positions, 100.0))


def test_rope_position_zero_is_identity():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=8, max_seq_len=4)
    x = torch.randn(1, 1, 8)
    torch.testing.assert_close(rope(x), x)


def test_rope_preserves_norms():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=32, max_seq_len=128)
    x = torch.randn(4, 128, 32)
    torch.testing.assert_close(rope(x).norm(dim=-1), x.norm(dim=-1))


def test_rope_scores_only_depend_on_relative_position():
    d, theta = 16, 10000.0
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d, max_seq_len=64)
    q, k = torch.randn(d), torch.randn(d)

    def score(m, n):
        qm = rope(q.view(1, 1, d), torch.tensor([m]))
        kn = rope(k.view(1, 1, d), torch.tensor([n]))
        return (qm * kn).sum()

    torch.testing.assert_close(score(5, 2), score(40, 37))
    torch.testing.assert_close(score(0, 0), score(63, 63))


def test_mha_with_rope_matches_reference():
    d_model, num_heads = 16, 4
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_model // num_heads, max_seq_len=32)
    mha = MultiHeadSelfAttention(d_model, num_heads, rope=rope)
    x = torch.randn(2, 9, d_model)
    expected = mha_reference(
        x,
        mha.q_proj.weight,
        mha.k_proj.weight,
        mha.v_proj.weight,
        mha.output_proj.weight,
        num_heads,
        rope_theta=10000.0,
    )
    torch.testing.assert_close(mha(x), expected)
