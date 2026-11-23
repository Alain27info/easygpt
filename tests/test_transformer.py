"""Tests for docs/part1/transformer.md: silu, RMSNorm, SwiGLU, TransformerBlock, TransformerLM."""

import pytest
import torch
import torch.nn.functional as F
from reference import block_reference, lm_num_parameters, lm_reference, rmsnorm_reference, swiglu_reference

from easygpt.attention import RotaryPositionalEmbedding
from easygpt.functional import silu
from easygpt.layers import RMSNorm, SwiGLU
from easygpt.model import TransformerBlock, TransformerLM


# -- SiLU --------------------------------------------------------------------
def test_silu_matches_pytorch():
    x = torch.linspace(-10, 10, 101)
    torch.testing.assert_close(silu(x), F.silu(x))


# -- RMSNorm -----------------------------------------------------------------
def test_rmsnorm_parameter():
    norm = RMSNorm(12)
    assert list(dict(norm.named_parameters())) == ["weight"]
    assert torch.equal(norm.weight.detach(), torch.ones(12))


def test_rmsnorm_matches_reference():
    norm = RMSNorm(12)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(12))
    x = torch.randn(3, 5, 12)
    torch.testing.assert_close(norm(x), rmsnorm_reference(x, norm.weight))


def test_rmsnorm_output_has_unit_rms():
    x = 5 * torch.randn(100, 64)
    out = RMSNorm(64, eps=0.0)(x)
    torch.testing.assert_close(out.pow(2).mean(dim=-1), torch.ones(100))


def test_rmsnorm_keeps_input_dtype():
    norm = RMSNorm(8)
    x = (300 * torch.randn(2, 8)).to(torch.float16)  # x**2 overflows in float16
    out = norm(x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


# -- SwiGLU ------------------------------------------------------------------
def test_swiglu_parameters():
    ffn = SwiGLU(d_model=16, d_ff=48)
    shapes = {name: tuple(p.shape) for name, p in ffn.named_parameters()}
    assert shapes == {"w1.weight": (48, 16), "w2.weight": (16, 48), "w3.weight": (48, 16)}


def test_swiglu_matches_reference():
    ffn = SwiGLU(d_model=16, d_ff=48)
    x = torch.randn(2, 7, 16)
    torch.testing.assert_close(ffn(x), swiglu_reference(x, ffn.w1.weight, ffn.w2.weight, ffn.w3.weight))


# -- Transformer block -------------------------------------------------------
def test_block_parameter_names():
    block = TransformerBlock(d_model=16, num_heads=4, d_ff=32)
    assert set(block.state_dict()) == {
        "ln1.weight",
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn.output_proj.weight",
        "ln2.weight",
        "ffn.w1.weight",
        "ffn.w2.weight",
        "ffn.w3.weight",
    }


@pytest.mark.parametrize("use_rope", [False, True])
def test_block_matches_reference(use_rope):
    d_model, num_heads = 16, 4
    rope = RotaryPositionalEmbedding(10000.0, d_model // num_heads, 32) if use_rope else None
    block = TransformerBlock(d_model, num_heads, d_ff=32, rope=rope)
    with torch.no_grad():  # non-trivial norm gains
        block.ln1.weight.copy_(torch.rand(d_model) + 0.5)
        block.ln2.weight.copy_(torch.rand(d_model) + 0.5)
    x = torch.randn(2, 8, d_model)
    expected = block_reference(x, block.state_dict(), "", num_heads, 10000.0 if use_rope else None)
    torch.testing.assert_close(block(x), expected)


def test_block_with_zero_output_projections_is_identity():
    block = TransformerBlock(d_model=16, num_heads=4, d_ff=32)
    with torch.no_grad():
        block.attn.output_proj.weight.zero_()
        block.ffn.w2.weight.zero_()
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(block(x), x)


# -- Transformer LM ----------------------------------------------------------
CONFIG = dict(vocab_size=50, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64)


def test_lm_parameter_names_and_count():
    model = TransformerLM(**CONFIG)
    names = set(model.state_dict())
    assert {"token_embeddings.weight", "ln_final.weight", "lm_head.weight"} <= names
    assert "layers.1.ffn.w3.weight" in names and "layers.2.ln1.weight" not in names
    expected = lm_num_parameters(CONFIG["vocab_size"], CONFIG["d_model"], CONFIG["num_layers"], CONFIG["d_ff"])
    assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.parametrize("rope_theta", [10000.0, None])
def test_lm_matches_reference(rope_theta):
    model = TransformerLM(**CONFIG, rope_theta=rope_theta)
    ids = torch.randint(0, CONFIG["vocab_size"], (3, CONFIG["context_length"]))
    logits = model(ids)
    assert logits.shape == (3, CONFIG["context_length"], CONFIG["vocab_size"])
    expected = lm_reference(ids, model.state_dict(), CONFIG["num_layers"], CONFIG["num_heads"], rope_theta)
    torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)


def test_lm_accepts_shorter_sequences():
    model = TransformerLM(**CONFIG)
    ids = torch.randint(0, CONFIG["vocab_size"], (2, 16))
    torch.testing.assert_close(model(ids[:, :5]), model(ids)[:, :5], atol=1e-5, rtol=1e-5)


def test_lm_is_causal():
    model = TransformerLM(**CONFIG)
    ids = torch.randint(0, CONFIG["vocab_size"], (2, 16))
    ids_changed = ids.clone()
    ids_changed[:, 10:] = torch.randint(0, CONFIG["vocab_size"], (2, 6))
    torch.testing.assert_close(model(ids)[:, :10], model(ids_changed)[:, :10])


def test_lm_rope_is_shared_and_not_saved():
    model = TransformerLM(**CONFIG)
    assert not any("cos" in k or "sin" in k for k in model.state_dict())
    ropes = [m for m in model.modules() if isinstance(m, RotaryPositionalEmbedding)]
    assert len(ropes) == 1, "a single RotaryPositionalEmbedding should be shared by all layers"
