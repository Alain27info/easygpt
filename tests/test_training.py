"""Tests for docs/part1/training.md: checkpointing, evaluate, train."""

import io
import math

import numpy as np
import pytest
import torch
from torch import nn

from easygpt.config import TrainConfig
from easygpt.data import get_batch
from easygpt.functional import cross_entropy
from easygpt.model import TransformerLM
from easygpt.optim import AdamW, lr_cosine_schedule
from easygpt.train import evaluate, load_checkpoint, read_checkpoint_extra, save_checkpoint, train

TINY = dict(vocab_size=5, context_length=8, d_model=16, num_layers=1, num_heads=2, d_ff=32)


def _steps(model, opt, data, n, seed):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n):
        x, y = get_batch(data, 4, 8, generator=gen)
        loss = cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


@pytest.mark.parametrize("to_file", [False, True])
def test_checkpoint_roundtrip_resumes_training_exactly(tmp_path, to_file):
    data = np.random.default_rng(0).integers(0, 5, 500)
    torch.manual_seed(0)
    model = TransformerLM(**TINY)
    opt = AdamW(model.parameters(), lr=1e-2)
    _steps(model, opt, data, 5, seed=1)

    out = tmp_path / "ckpt.pt" if to_file else io.BytesIO()
    save_checkpoint(model, opt, 5, out, extra={"note": "hello"})
    if not to_file:
        out.seek(0)

    # reference: keep training the original model
    _steps(model, opt, data, 5, seed=2)

    # resumed: fresh model & optimizer restored from the checkpoint
    torch.manual_seed(123)
    model2 = TransformerLM(**TINY)
    opt2 = AdamW(model2.parameters(), lr=1e-2)
    assert load_checkpoint(out, model2, opt2) == 5
    _steps(model2, opt2, data, 5, seed=2)

    for p, q in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p, q)
    if not to_file:
        out.seek(0)
    assert read_checkpoint_extra(out) == {"note": "hello"}


def test_load_checkpoint_without_optimizer(tmp_path):
    model = TransformerLM(**TINY)
    save_checkpoint(model, None, 7, tmp_path / "ckpt.pt")
    model2 = TransformerLM(**TINY)
    assert load_checkpoint(tmp_path / "ckpt.pt", model2) == 7
    for p, q in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p, q)


class UniformModel(nn.Module):
    """Predicts the uniform distribution whatever the input."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, ids):
        return torch.zeros(*ids.shape, self.vocab_size) + 0 * self.dummy


def test_evaluate_uniform_model():
    data = np.arange(100) % 7
    loss = evaluate(UniformModel(7), data, batch_size=4, context_length=8, num_batches=3)
    assert isinstance(loss, float)
    assert loss == pytest.approx(math.log(7), rel=1e-5)


def test_evaluate_does_not_build_a_graph():
    model = TransformerLM(**TINY)
    data = np.arange(100) % 5
    evaluate(model, data, batch_size=2, context_length=8, num_batches=2)
    assert all(p.grad is None for p in model.parameters())


def test_train_learns_a_periodic_sequence():
    torch.manual_seed(0)
    data = (np.arange(2000) % 5).astype(np.uint16)  # 0 1 2 3 4 0 1 2 ...
    model = TransformerLM(**TINY)
    config = TrainConfig(
        batch_size=8, max_steps=60, lr_max=1e-2, lr_min=1e-3, warmup_steps=5, eval_interval=20, eval_batches=2
    )
    history = train(model, data, data, config, context_length=8, log=False)

    assert len(history["train_loss"]) == 60
    assert history["step"] == list(range(60))
    assert history["val_step"][-1] == 60
    assert len(history["val_loss"]) == len(history["val_step"]) == len(history["time"])
    expected_lrs = [lr_cosine_schedule(t, 1e-2, 1e-3, 5, 60) for t in range(60)]
    assert history["lr"] == pytest.approx(expected_lrs)
    assert history["train_loss"][0] > 1.0  # about log(5) = 1.6 at initialization
    assert history["val_loss"][-1] < 0.1
