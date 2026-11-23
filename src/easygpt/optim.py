"""Optimization: SGD (given), AdamW, cosine learning-rate schedule, gradient clipping.

Docs: docs/part1/optimizer.md
"""

import math
from collections.abc import Callable, Iterable

import torch


class SGD(torch.optim.Optimizer):
    """Plain stochastic gradient descent ``p <- p - lr * grad`` (given).

    It is a template showing the ``torch.optim.Optimizer`` API you will reuse
    for AdamW.
    """

    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, defaults={"lr": lr})

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p -= lr * p.grad
        return loss


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    For each parameter ``p`` with gradient ``g``, at step ``t = 1, 2, ...``::

        p <- p - lr * weight_decay * p
        m <- beta1 * m + (1 - beta1) * g
        v <- beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    The state of each parameter (``self.state[p]``) holds ``"step"``,
    ``"exp_avg"`` (that is ``m``) and ``"exp_avg_sq"`` (that is ``v``).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                raise NotImplementedError("Exercise: AdamW.step (docs/part1/optimizer.md)")
        return loss


def lr_cosine_schedule(
    t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """Learning rate at iteration ``t`` for linear warmup + cosine decay.

    - if ``t < warmup_iters``: ``t / warmup_iters * lr_max``
    - if ``warmup_iters <= t <= cosine_cycle_iters``:
      ``lr_min + 0.5 * (1 + cos(pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (lr_max - lr_min)``
    - if ``t > cosine_cycle_iters``: ``lr_min``
    """
    raise NotImplementedError("Exercise: lr_cosine_schedule (docs/part1/optimizer.md)")


def clip_grad_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> float:
    """Rescale gradients in place so that their global l2 norm is at most ``max_norm``.

    The global norm is ``sqrt(sum over all parameters of sum(grad**2))``
    (parameters without gradient are ignored). If it exceeds ``max_norm``,
    every gradient is multiplied by ``max_norm / (norm + eps)``.

    Returns:
        The global norm *before* clipping, as a Python float.
    """
    raise NotImplementedError("Exercise: clip_grad_norm (docs/part1/optimizer.md)")
