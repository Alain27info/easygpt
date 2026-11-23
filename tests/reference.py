"""Reference implementations used by the tests.

They rely on ``torch.nn.functional`` (which you are *not* allowed to use in
``src/easygpt``) or on deliberately different formulations (loops over heads,
complex numbers for RoPE), so that a test passing is good evidence that your
implementation is right.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def rope_reference(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """RoPE with complex numbers: pair (x[2k], x[2k+1]) is the complex x[2k] + i x[2k+1]."""
    d = x.shape[-1]
    freqs = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    angles = positions.to(torch.float64)[..., None] * freqs  # (..., T, d/2)
    z = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], d // 2, 2).contiguous())
    z = z * torch.polar(torch.ones_like(angles), angles)
    return torch.view_as_real(z).reshape(x.shape).to(x.dtype)


def mha_reference(
    x: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    num_heads: int,
    rope_theta: float | None = None,
) -> Tensor:
    """Causal multi-head self-attention, one head at a time (no reshape tricks)."""
    batch, seq_len, d_model = x.shape
    d_head = d_model // num_heads
    positions = torch.arange(seq_len)
    heads = []
    for h in range(num_heads):
        rows = slice(h * d_head, (h + 1) * d_head)  # rows of W for head h
        q = x @ wq[rows].T  # (batch, seq_len, d_head)
        k = x @ wk[rows].T
        v = x @ wv[rows].T
        if rope_theta is not None:
            q = rope_reference(q, positions, rope_theta)
            k = rope_reference(k, positions, rope_theta)
        heads.append(F.scaled_dot_product_attention(q, k, v, is_causal=True))
    return torch.cat(heads, dim=-1) @ wo.T


def swiglu_reference(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def rmsnorm_reference(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    return F.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps).to(x.dtype)


def block_reference(x: Tensor, sd: dict[str, Tensor], prefix: str, num_heads: int, rope_theta) -> Tensor:
    h = rmsnorm_reference(x, sd[prefix + "ln1.weight"])
    x = x + mha_reference(
        h,
        sd[prefix + "attn.q_proj.weight"],
        sd[prefix + "attn.k_proj.weight"],
        sd[prefix + "attn.v_proj.weight"],
        sd[prefix + "attn.output_proj.weight"],
        num_heads,
        rope_theta,
    )
    h = rmsnorm_reference(x, sd[prefix + "ln2.weight"])
    return x + swiglu_reference(h, sd[prefix + "ffn.w1.weight"], sd[prefix + "ffn.w2.weight"], sd[prefix + "ffn.w3.weight"])


def lm_reference(ids: Tensor, sd: dict[str, Tensor], num_layers: int, num_heads: int, rope_theta) -> Tensor:
    x = F.embedding(ids, sd["token_embeddings.weight"])
    for i in range(num_layers):
        x = block_reference(x, sd, f"layers.{i}.", num_heads, rope_theta)
    x = rmsnorm_reference(x, sd["ln_final.weight"])
    return F.linear(x, sd["lm_head.weight"])


def lm_num_parameters(vocab_size: int, d_model: int, num_layers: int, d_ff: int) -> int:
    per_layer = 4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model
    return 2 * vocab_size * d_model + num_layers * per_layer + d_model


def uniform_loss(vocab_size: int) -> float:
    return math.log(vocab_size)
