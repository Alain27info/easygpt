"""Tests for docs/part1/bigram.md: softmax, cross_entropy, Linear, Embedding, BigramLM."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from easygpt.functional import cross_entropy, softmax
from easygpt.layers import Embedding, Linear
from easygpt.model import BigramLM


# -- softmax -----------------------------------------------------------------
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_matches_pytorch(dim):
    x = torch.randn(4, 5, 6)
    torch.testing.assert_close(softmax(x, dim=dim), torch.softmax(x, dim=dim))


def test_softmax_default_dim_is_last():
    x = torch.randn(3, 7)
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1))


def test_softmax_is_numerically_stable():
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    out = softmax(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


def test_softmax_is_shift_invariant():
    x = torch.randn(10)
    torch.testing.assert_close(softmax(x + 123.0), softmax(x))


# -- cross-entropy -----------------------------------------------------------
def test_cross_entropy_matches_pytorch_2d():
    logits = torch.randn(32, 11)
    targets = torch.randint(0, 11, (32,))
    torch.testing.assert_close(cross_entropy(logits, targets), F.cross_entropy(logits, targets))


def test_cross_entropy_matches_pytorch_3d():
    logits = torch.randn(4, 9, 11)
    targets = torch.randint(0, 11, (4, 9))
    expected = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    out = cross_entropy(logits, targets)
    assert out.shape == ()
    torch.testing.assert_close(out, expected)


def test_cross_entropy_uniform_logits_is_log_vocab_size():
    logits = torch.zeros(5, 65)
    targets = torch.randint(0, 65, (5,))
    assert cross_entropy(logits, targets).item() == pytest.approx(math.log(65))


def test_cross_entropy_is_numerically_stable():
    logits = 1000 * torch.randn(16, 50)
    targets = torch.randint(0, 50, (16,))
    out = cross_entropy(logits, targets)
    assert torch.isfinite(out)
    torch.testing.assert_close(out, F.cross_entropy(logits, targets))


def test_cross_entropy_gradient_matches_pytorch():
    logits = torch.randn(8, 13, requires_grad=True)
    targets = torch.randint(0, 13, (8,))
    cross_entropy(logits, targets).backward()
    grad = logits.grad.clone()
    logits.grad = None
    F.cross_entropy(logits, targets).backward()
    torch.testing.assert_close(grad, logits.grad)


# -- Linear ------------------------------------------------------------------
def test_linear_has_a_single_weight_parameter():
    layer = Linear(7, 3)
    params = dict(layer.named_parameters())
    assert list(params) == ["weight"], "Linear should have a 'weight' parameter and no bias"
    assert isinstance(layer.weight, nn.Parameter)
    assert layer.weight.shape == (3, 7)


def test_linear_forward_matches_pytorch():
    layer = Linear(7, 3)
    x = torch.randn(2, 5, 7)
    out = layer(x)
    assert out.shape == (2, 5, 3)
    torch.testing.assert_close(out, F.linear(x, layer.weight))


def test_linear_initialization():
    d_in, d_out = 512, 256
    w = Linear(d_in, d_out).weight.detach()
    std = math.sqrt(2 / (d_in + d_out))
    assert w.mean().abs() < 0.01 * std * 10
    # truncation at 3 std shrinks the empirical std a bit (by ~1.3%)
    assert w.std().item() == pytest.approx(std, rel=0.05)
    assert w.abs().max() <= 3 * std + 1e-6


def test_linear_respects_dtype():
    layer = Linear(4, 4, dtype=torch.float64)
    assert layer.weight.dtype == torch.float64


# -- Embedding ---------------------------------------------------------------
def test_embedding_parameter_and_lookup():
    emb = Embedding(10, 4)
    assert list(dict(emb.named_parameters())) == ["weight"]
    assert emb.weight.shape == (10, 4)
    ids = torch.tensor([[1, 2, 3], [9, 9, 0]])
    out = emb(ids)
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out, F.embedding(ids, emb.weight))


def test_embedding_initialization():
    w = Embedding(1000, 64).weight.detach()
    assert w.std().item() == pytest.approx(1.0, rel=0.05)
    assert w.abs().max() <= 3.0


def test_embedding_gradient_flows_to_selected_rows_only():
    emb = Embedding(5, 3)
    emb(torch.tensor([1, 1, 3])).sum().backward()
    grad = emb.weight.grad
    assert torch.equal(grad[1], torch.full((3,), 2.0))
    assert torch.equal(grad[3], torch.ones(3))
    assert torch.equal(grad[[0, 2, 4]], torch.zeros(3, 3))


# -- BigramLM ----------------------------------------------------------------
def test_bigram_shapes_and_parameters():
    model = BigramLM(vocab_size=17)
    assert set(model.state_dict()) == {"logits_table.weight"}
    assert sum(p.numel() for p in model.parameters()) == 17 * 17
    ids = torch.randint(0, 17, (3, 8))
    assert model(ids).shape == (3, 8, 17)


def test_bigram_logits_are_rows_of_the_table():
    model = BigramLM(vocab_size=5)
    ids = torch.tensor([[4, 0, 2]])
    logits = model(ids)
    torch.testing.assert_close(logits[0, 0], model.logits_table.weight[4])
    torch.testing.assert_close(logits[0, 2], model.logits_table.weight[2])


def test_bigram_learns_a_deterministic_sequence():
    # "0 1 2 3 0 1 2 3 ...": the next token is a function of the current one.
    vocab_size = 4
    seq = torch.arange(64) % vocab_size
    x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)
    model = BigramLM(vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=0.5)
    for _ in range(100):
        loss = cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
