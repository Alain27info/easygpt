"""Causal multi-head self-attention, with rotary position embeddings (RoPE).

Docs: docs/part1/attention.md (causal_mask, scaled_dot_product_attention,
MultiHeadSelfAttention) and docs/part1/rope.md (RotaryPositionalEmbedding).
"""

import math

import torch
from torch import Tensor, nn

from easygpt.functional import softmax
from easygpt.layers import Linear


def causal_mask(seq_len: int, device=None) -> Tensor:
    """Boolean mask of shape ``(seq_len, seq_len)``.

    Entry ``(i, j)`` is ``True`` when query ``i`` may attend to key ``j``,
    that is when ``j <= i``.
    """
    raise NotImplementedError("Exercise: causal_mask (docs/part1/attention.md)")


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
) -> Tensor:
    """``softmax(q k^T / sqrt(d_k)) v`` with an optional boolean mask.

    Args:
        q: queries, shape ``(..., n_queries, d_k)``.
        k: keys, shape ``(..., n_keys, d_k)``.
        v: values, shape ``(..., n_keys, d_v)``.
        mask: optional boolean tensor broadcastable to
            ``(..., n_queries, n_keys)``. ``True`` means "may attend",
            ``False`` means the attention probability must be exactly 0.

    Returns:
        Tensor of shape ``(..., n_queries, d_v)``.
    """
    raise NotImplementedError("Exercise: scaled_dot_product_attention (docs/part1/attention.md)")


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embedding (RoPE).

    The pair of coordinates ``(x[2k], x[2k+1])`` of a vector at position ``i``
    is rotated by the angle ``i * theta^(-2k / d_k)``, for ``k = 0..d_k/2-1``.
    This layer has no learnable parameter: ``cos`` and ``sin`` tables of shape
    ``(max_seq_len, d_k // 2)`` are precomputed and stored as (non-persistent)
    buffers named ``self.cos`` and ``self.sin``.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE needs an even dimension, got d_k={d_k}")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        raise NotImplementedError("Exercise: RotaryPositionalEmbedding.__init__ (docs/part1/rope.md)")

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Rotate ``x`` of shape ``(..., seq_len, d_k)``.

        Args:
            x: input tensor; any number of leading batch dimensions.
            token_positions: integer positions of shape ``(seq_len,)`` (or any
                shape broadcastable to ``x.shape[:-1]``). Defaults to
                ``0, 1, ..., seq_len - 1``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        raise NotImplementedError("Exercise: RotaryPositionalEmbedding.forward (docs/part1/rope.md)")


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Sub-modules (names matter for loading weights), all
    ``Linear(d_model, d_model)``: ``self.q_proj``, ``self.k_proj``,
    ``self.v_proj`` and ``self.output_proj``.

    Each head has dimension ``d_head = d_model // num_heads``. If ``rope`` is
    given, it is applied to queries and keys (not values) of every head.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope
        raise NotImplementedError("Exercise: MultiHeadSelfAttention.__init__ (docs/part1/attention.md)")

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Map ``(batch, seq_len, d_model)`` to ``(batch, seq_len, d_model)``."""
        raise NotImplementedError("Exercise: MultiHeadSelfAttention.forward (docs/part1/attention.md)")
