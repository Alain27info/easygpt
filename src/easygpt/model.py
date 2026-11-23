"""Language models: a bigram baseline and a pre-norm Transformer LM.

Docs: docs/part1/bigram.md (BigramLM) and docs/part1/transformer.md
(TransformerBlock, TransformerLM).
"""

from torch import Tensor, nn

from easygpt.attention import MultiHeadSelfAttention, RotaryPositionalEmbedding
from easygpt.layers import Embedding, Linear, RMSNorm, SwiGLU


class BigramLM(nn.Module):
    """The simplest language model: the next token only depends on the current one.

    It is a single lookup table ``self.logits_table``, an
    ``Embedding(vocab_size, vocab_size)``: row ``a`` holds the logits of the
    next token after token ``a``.
    """

    def __init__(self, vocab_size: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        raise NotImplementedError("Exercise: BigramLM.__init__ (docs/part1/bigram.md)")

    def forward(self, token_ids: Tensor) -> Tensor:
        """Map ``(batch, seq_len)`` ids to ``(batch, seq_len, vocab_size)`` logits."""
        raise NotImplementedError("Exercise: BigramLM.forward (docs/part1/bigram.md)")


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block.

    ``z = x + attn(ln1(x))`` then ``y = z + ffn(ln2(z))``.

    Sub-modules (names matter for loading weights): ``self.ln1``
    (``RMSNorm``), ``self.attn`` (``MultiHeadSelfAttention``), ``self.ln2``
    (``RMSNorm``) and ``self.ffn`` (``SwiGLU``).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        raise NotImplementedError("Exercise: TransformerBlock.__init__ (docs/part1/transformer.md)")

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Map ``(batch, seq_len, d_model)`` to ``(batch, seq_len, d_model)``."""
        raise NotImplementedError("Exercise: TransformerBlock.forward (docs/part1/transformer.md)")


class TransformerLM(nn.Module):
    """A decoder-only Transformer language model.

    ``token ids -> Embedding -> num_layers x TransformerBlock -> RMSNorm -> Linear``

    Sub-modules (names matter for loading weights):

    - ``self.token_embeddings``: ``Embedding(vocab_size, d_model)``
    - ``self.layers``: ``nn.ModuleList`` of ``num_layers`` ``TransformerBlock``
    - ``self.ln_final``: ``RMSNorm(d_model)``
    - ``self.lm_head``: ``Linear(d_model, vocab_size)``

    A single ``RotaryPositionalEmbedding(rope_theta, d_model // num_heads,
    context_length)`` is shared by all the attention layers. With
    ``rope_theta=None`` no position information is used at all ("NoPE").
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        raise NotImplementedError("Exercise: TransformerLM.__init__ (docs/part1/transformer.md)")

    def forward(self, token_ids: Tensor) -> Tensor:
        """Map ``(batch, seq_len)`` ids to ``(batch, seq_len, vocab_size)`` logits.

        ``seq_len`` must not exceed ``context_length``.
        """
        raise NotImplementedError("Exercise: TransformerLM.forward (docs/part1/transformer.md)")


def count_parameters(model: nn.Module, exclude_embeddings: bool = False) -> int:
    """Number of trainable scalars in ``model`` (given, not an exercise).

    With ``exclude_embeddings=True``, the input embedding table is not counted
    (the usual convention when quoting model sizes).
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if exclude_embeddings and hasattr(model, "token_embeddings"):
        total -= model.token_embeddings.weight.numel()
    return total
