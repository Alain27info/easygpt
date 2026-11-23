"""Decoding: temperature, top-p (nucleus) sampling and autoregressive generation.

Docs: docs/part1/sampling.md
"""

import torch
from torch import Tensor, nn

from easygpt.functional import softmax


def top_p_filter(probs: Tensor, top_p: float) -> Tensor:
    """Keep the smallest set of most likely tokens whose total mass is ``>= top_p``.

    Args:
        probs: probabilities of shape ``(..., vocab_size)`` (rows sum to 1).
        top_p: threshold in ``(0, 1]``.

    Returns:
        A tensor with the same shape as ``probs`` where the probabilities
        outside the "nucleus" are set to 0 and the remaining ones are
        renormalized to sum to 1. The most likely token is always kept.
    """
    raise NotImplementedError("Exercise: top_p_filter (docs/part1/sampling.md)")


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token id per row of ``logits``.

    Args:
        logits: shape ``(batch, vocab_size)``.
        temperature: ``0`` means greedy decoding (argmax); otherwise logits are
            divided by ``temperature`` before the softmax.
        top_p: nucleus threshold; ``1.0`` disables the filter.
        generator: optional ``torch.Generator`` for reproducibility.

    Returns:
        A ``torch.long`` tensor of shape ``(batch,)``.
    """
    raise NotImplementedError("Exercise: sample_next_token (docs/part1/sampling.md)")


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Autoregressively extend ``prompt_ids`` with at most ``max_new_tokens`` tokens.

    At each step the model only sees the last ``context_length`` tokens, the
    next token is drawn with :func:`sample_next_token` from the logits of the
    *last* position and appended to the sequence. Generation stops early if
    ``eos_id`` is produced (it is included in the output).

    Returns:
        The list of *new* token ids (the prompt is not included).
    """
    device = next(model.parameters()).device
    raise NotImplementedError("Exercise: generate (docs/part1/sampling.md)")
