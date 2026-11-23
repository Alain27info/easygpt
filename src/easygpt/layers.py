"""Parametric layers: Linear, Embedding, RMSNorm and the SwiGLU feed-forward.

Docs: docs/part1/bigram.md (Linear, Embedding) and
docs/part1/transformer.md (RMSNorm, SwiGLU).
"""

import math

import torch
from torch import Tensor, nn

from easygpt.functional import silu


class Linear(nn.Module):
    """A linear map ``y = x W^T`` without bias.

    The weight is stored as ``self.weight`` with shape
    ``(out_features, in_features)``, like ``torch.nn.Linear``. It is initialized
    from a normal distribution with mean 0 and variance
    ``2 / (in_features + out_features)``, truncated at three standard
    deviations.
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        raise NotImplementedError("Exercise: Linear.__init__ (docs/part1/bigram.md)")

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(..., in_features)`` to ``(..., out_features)``."""
        raise NotImplementedError("Exercise: Linear.forward (docs/part1/bigram.md)")


class Embedding(nn.Module):
    """A lookup table mapping integer ids to vectors.

    The table is stored as ``self.weight`` with shape
    ``(num_embeddings, embedding_dim)`` and initialized from a standard normal
    truncated to ``[-3, 3]``.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        raise NotImplementedError("Exercise: Embedding.__init__ (docs/part1/bigram.md)")

    def forward(self, token_ids: Tensor) -> Tensor:
        """Map integer ids of shape ``(...)`` to vectors ``(..., embedding_dim)``."""
        raise NotImplementedError("Exercise: Embedding.forward (docs/part1/bigram.md)")


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learnable gain ``self.weight``.

    ``RMSNorm(a)_i = a_i / RMS(a) * g_i`` with
    ``RMS(a) = sqrt(mean(a^2) + eps)``, computed over the last dimension.
    The gain ``g`` has shape ``(d_model,)`` and is initialized to ones.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        raise NotImplementedError("Exercise: RMSNorm.__init__ (docs/part1/transformer.md)")

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` of shape ``(..., d_model)``; same shape and dtype out."""
        in_dtype = x.dtype
        x = x.to(torch.float32)  # squaring in low precision can overflow
        raise NotImplementedError("Exercise: RMSNorm.forward (docs/part1/transformer.md)")
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """Gated feed-forward network ``W2 (SiLU(W1 x) * W3 x)``.

    Sub-modules (names matter for loading weights):

    - ``self.w1``: ``Linear(d_model, d_ff)`` (gate branch)
    - ``self.w2``: ``Linear(d_ff, d_model)`` (output projection)
    - ``self.w3``: ``Linear(d_model, d_ff)`` (value branch)
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Exercise: SwiGLU.__init__ (docs/part1/transformer.md)")

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(..., d_model)`` to ``(..., d_model)``."""
        raise NotImplementedError("Exercise: SwiGLU.forward (docs/part1/transformer.md)")
