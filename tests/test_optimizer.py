"""Tests for docs/part1/optimizer.md: AdamW, lr_cosine_schedule, clip_grad_norm."""

import pytest
import torch
from torch import nn

from easygpt.optim import SGD, AdamW, clip_grad_norm, lr_cosine_schedule


def _make_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))


def _run(model: nn.Module, opt: torch.optim.Optimizer, steps: int = 25) -> None:
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randn(16, 6, generator=gen)
        y = torch.randn(16, 3, generator=gen)
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_sgd_is_given_and_works():
    model = _make_model()
    before = [p.detach().clone() for p in model.parameters()]
    _run(model, SGD(model.parameters(), lr=0.1), steps=1)
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters()))


@pytest.mark.parametrize(
    "hyper",
    [
        dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01),
        dict(lr=3e-2, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.1),
        dict(lr=1e-2, betas=(0.8, 0.9), eps=1e-8, weight_decay=0.0),
    ],
)
def test_adamw_matches_pytorch(hyper):
    ours, theirs = _make_model(), _make_model()
    _run(ours, AdamW(ours.parameters(), **hyper))
    _run(theirs, torch.optim.AdamW(theirs.parameters(), **hyper))
    for p, q in zip(ours.parameters(), theirs.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def test_adamw_state():
    model = _make_model()
    opt = AdamW(model.parameters(), lr=1e-3)
    _run(model, opt, steps=3)
    p = next(model.parameters())
    state = opt.state[p]
    assert state["step"] == 3
    assert state["exp_avg"].shape == p.shape
    assert state["exp_avg_sq"].shape == p.shape
    assert (state["exp_avg_sq"] >= 0).all()


def test_adamw_supports_parameter_groups_and_skips_missing_gradients():
    a = nn.Parameter(torch.ones(3))
    b = nn.Parameter(torch.ones(3))
    frozen = nn.Parameter(torch.ones(3))
    opt = AdamW([{"params": [a], "lr": 0.1}, {"params": [b, frozen], "lr": 0.0}], weight_decay=0.0)
    (a.sum() + b.sum()).backward()
    opt.step()
    # first Adam step moves each coordinate by exactly lr (m_hat / sqrt(v_hat) = sign(g))
    torch.testing.assert_close(a.detach(), torch.full((3,), 0.9))
    torch.testing.assert_close(b.detach(), torch.ones(3))
    assert frozen.grad is None and torch.equal(frozen.detach(), torch.ones(3))


def test_adamw_weight_decay_is_decoupled():
    # with a zero gradient, AdamW only shrinks the weights: p <- p (1 - lr wd)
    p = nn.Parameter(torch.full((4,), 2.0))
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros(4)
    opt.step()
    torch.testing.assert_close(p.detach(), torch.full((4,), 2.0 * (1 - 0.1 * 0.5)))


EXPECTED_LRS = [
    0.0, 0.1428571429, 0.2857142857, 0.4285714286, 0.5714285714, 0.7142857143, 0.8571428571,
    1.0, 0.9887175605, 0.9554359906, 0.9018241671, 0.8305704108, 0.7452476826, 0.6501344203,
    0.55, 0.4498655797, 0.3547523174, 0.2694295892, 0.1981758329, 0.1445640094, 0.1112824395,
    0.1, 0.1, 0.1, 0.1,
]  # fmt: skip


def test_lr_cosine_schedule():
    lrs = [lr_cosine_schedule(t, lr_max=1.0, lr_min=0.1, warmup_iters=7, cosine_cycle_iters=21) for t in range(25)]
    assert lrs == pytest.approx(EXPECTED_LRS, abs=1e-9)


def _grads_for(params):
    gen = torch.Generator().manual_seed(3)
    for p in params:
        p.grad = 10 * torch.randn(p.shape, generator=gen)


@pytest.mark.parametrize("max_norm", [0.5, 1.0, 1e4])
def test_clip_grad_norm_matches_pytorch(max_norm):
    ours = [nn.Parameter(torch.randn(5, 4)), nn.Parameter(torch.randn(7)), nn.Parameter(torch.randn(2, 2))]
    theirs = [nn.Parameter(p.detach().clone()) for p in ours]
    _grads_for(ours)
    _grads_for(theirs)
    ours[2].grad = None  # parameters without gradient are ignored
    theirs[2].grad = None
    norm = clip_grad_norm(ours, max_norm)
    expected_norm = torch.nn.utils.clip_grad_norm_(theirs, max_norm)
    assert float(norm) == pytest.approx(expected_norm.item(), rel=1e-5)
    for p, q in zip(ours[:2], theirs[:2]):
        torch.testing.assert_close(p.grad, q.grad)


def test_clip_grad_norm_result_is_just_below_max_norm():
    params = [nn.Parameter(torch.randn(100))]
    _grads_for(params)
    clip_grad_norm(params, 1.0)
    assert params[0].grad.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert params[0].grad.norm().item() <= 1.0 + 1e-6
