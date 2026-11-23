"""Stateless building blocks: softmax, cross-entropy and SiLU.

Docs: docs/part1/bigram.md (softmax, cross_entropy) and
docs/part1/transformer.md (silu).
"""

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax along dimension ``dim``.

    Args:
        x: tensor of any shape.
        dim: the dimension along which the output sums to one.

    Returns:
        A tensor with the same shape as ``x``, non-negative, whose slices along
        ``dim`` sum to 1.
    """
    raise NotImplementedError("Exercise: softmax (docs/part1/bigram.md)")


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Average cross-entropy between ``logits`` and integer ``targets``.

    Args:
        logits: tensor of shape ``(..., vocab_size)`` of unnormalized scores.
        targets: integer tensor of shape ``(...)`` with values in
            ``[0, vocab_size)``.

    Returns:
        A scalar tensor: the mean over all leading positions of
        ``-log softmax(logits)[target]``.
    """
    raise NotImplementedError("Exercise: cross_entropy (docs/part1/bigram.md)")


def silu(x: Tensor) -> Tensor:
    """SiLU (a.k.a. Swish) activation ``x * sigmoid(x)``, elementwise."""
    raise NotImplementedError("Exercise: silu (docs/part1/transformer.md)")
