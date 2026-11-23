"""Tests for docs/part1/sampling.md: top_p_filter, sample_next_token, generate."""

import pytest
import torch
from torch import nn

from easygpt.generate import generate, sample_next_token, top_p_filter


# -- top-p -------------------------------------------------------------------
def test_top_p_keeps_the_smallest_nucleus():
    probs = torch.tensor([0.05, 0.5, 0.15, 0.3])
    out = top_p_filter(probs, 0.75)  # 0.5 + 0.3 = 0.8 >= 0.75
    torch.testing.assert_close(out, torch.tensor([0.0, 0.625, 0.0, 0.375]))


def test_top_p_three_tokens():
    probs = torch.tensor([0.05, 0.5, 0.15, 0.3])
    out = top_p_filter(probs, 0.9)  # 0.5 + 0.3 + 0.15 = 0.95 >= 0.9
    torch.testing.assert_close(out, torch.tensor([0.0, 0.5, 0.15, 0.3]) / 0.95)


def test_top_p_one_keeps_everything():
    probs = torch.softmax(torch.randn(3, 10), dim=-1)
    torch.testing.assert_close(top_p_filter(probs, 1.0), probs)


def test_top_p_tiny_keeps_only_the_argmax():
    probs = torch.softmax(torch.randn(4, 10), dim=-1)
    out = top_p_filter(probs, 1e-6)
    assert torch.equal(out.argmax(dim=-1), probs.argmax(dim=-1))
    torch.testing.assert_close(out.max(dim=-1).values, torch.ones(4))


def test_top_p_is_batched():
    probs = torch.tensor([[0.05, 0.5, 0.15, 0.3], [0.6, 0.05, 0.25, 0.1]])
    out = top_p_filter(probs, 0.75)
    expected = torch.tensor([[0.0, 0.625, 0.0, 0.375], [0.6 / 0.85, 0.0, 0.25 / 0.85, 0.0]])
    torch.testing.assert_close(out, expected)


# -- sampling ----------------------------------------------------------------
def test_greedy_decoding():
    logits = torch.randn(5, 20)
    assert torch.equal(sample_next_token(logits, temperature=0.0), logits.argmax(dim=-1))


def test_sample_shape_and_dtype():
    out = sample_next_token(torch.randn(3, 11))
    assert out.shape == (3,)
    assert out.dtype == torch.long


def test_low_temperature_is_almost_greedy():
    logits = torch.randn(64, 20)
    assert torch.equal(sample_next_token(logits, temperature=1e-3), logits.argmax(dim=-1))


def test_sampling_follows_the_softmax():
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.7]])).expand(20000, 3)
    gen = torch.Generator().manual_seed(0)
    samples = sample_next_token(logits, generator=gen)
    freqs = torch.bincount(samples, minlength=3).float() / len(samples)
    torch.testing.assert_close(freqs, torch.tensor([0.1, 0.2, 0.7]), atol=0.015, rtol=0)


def test_temperature_sharpens_and_flattens():
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.7]])).expand(20000, 3)
    gen = torch.Generator().manual_seed(0)
    hot = torch.bincount(sample_next_token(logits, temperature=2.0, generator=gen), minlength=3) / 20000
    cold = torch.bincount(sample_next_token(logits, temperature=0.5, generator=gen), minlength=3) / 20000
    p_hot = torch.tensor([0.1, 0.2, 0.7]) ** 0.5
    p_cold = torch.tensor([0.1, 0.2, 0.7]) ** 2
    torch.testing.assert_close(hot, p_hot / p_hot.sum(), atol=0.015, rtol=0)
    torch.testing.assert_close(cold, p_cold / p_cold.sum(), atol=0.015, rtol=0)


def test_top_p_sampling_never_picks_outside_the_nucleus():
    logits = torch.log(torch.tensor([[0.05, 0.5, 0.15, 0.3]])).expand(2000, 4)
    samples = sample_next_token(logits, top_p=0.75)
    assert set(samples.tolist()) == {1, 3}


def test_sampling_is_reproducible():
    logits = torch.randn(10, 50)
    a = sample_next_token(logits, generator=torch.Generator().manual_seed(7))
    b = sample_next_token(logits, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


# -- generation --------------------------------------------------------------
class CountingModel(nn.Module):
    """Always predicts ``(last token + 1) % vocab_size`` with high confidence."""

    def __init__(self, vocab_size=10, context_length=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.dummy = nn.Parameter(torch.zeros(()))
        self.seen_lengths = []

    def forward(self, ids):
        batch, seq_len = ids.shape
        assert seq_len <= self.context_length, "the context must be cropped to context_length"
        self.seen_lengths.append(seq_len)
        nxt = (ids + 1) % self.vocab_size
        return 100.0 * torch.nn.functional.one_hot(nxt, self.vocab_size).float() + 0 * self.dummy


def test_generate_appends_the_predicted_tokens():
    model = CountingModel()
    out = generate(model, [3], max_new_tokens=8, context_length=4)
    assert out == [4, 5, 6, 7, 8, 9, 0, 1]


def test_generate_crops_the_context():
    model = CountingModel(context_length=4)
    generate(model, [1, 2, 3], max_new_tokens=6, context_length=4)
    assert model.seen_lengths == [3, 4, 4, 4, 4, 4]


def test_generate_stops_at_eos():
    model = CountingModel()
    out = generate(model, [5], max_new_tokens=100, context_length=4, eos_id=8)
    assert out == [6, 7, 8]


def test_generate_greedy_is_deterministic():
    model = CountingModel()
    assert generate(model, [0], 5, 4, temperature=0.0) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("top_p", [1.0, 0.5])
def test_generate_with_a_real_model(top_p):
    from easygpt.model import TransformerLM

    model = TransformerLM(vocab_size=20, context_length=8, d_model=16, num_layers=1, num_heads=2, d_ff=32)
    out = generate(model, [1, 2, 3], 12, 8, temperature=1.0, top_p=top_p, generator=torch.Generator().manual_seed(0))
    assert len(out) == 12
    assert all(0 <= t < 20 for t in out)
