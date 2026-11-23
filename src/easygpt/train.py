"""Training loop, evaluation and checkpointing.

Docs: docs/part1/training.md
"""

import math
import os
import time
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import nn

from easygpt.config import TrainConfig
from easygpt.data import get_batch
from easygpt.functional import cross_entropy
from easygpt.optim import AdamW, clip_grad_norm, lr_cosine_schedule

PathOrFile = str | os.PathLike | BinaryIO | IO[bytes]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    out: PathOrFile,
    extra: dict | None = None,
) -> None:
    """Save the model, the optimizer state and the iteration number to ``out``.

    ``extra`` is an optional dictionary of plain Python objects (configuration,
    tokenizer, ...) stored alongside under the key ``"extra"``.
    """
    raise NotImplementedError("Exercise: save_checkpoint (docs/part1/training.md)")


def load_checkpoint(
    src: PathOrFile,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Restore ``model`` (and ``optimizer`` if given) from ``src``; return the iteration."""
    raise NotImplementedError("Exercise: load_checkpoint (docs/part1/training.md)")


def read_checkpoint_extra(src: PathOrFile) -> dict:
    """Return the ``extra`` dictionary of a checkpoint (given)."""
    return torch.load(src, map_location="cpu")["extra"]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    num_batches: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> float:
    """Average cross-entropy of ``model`` over ``num_batches`` random batches of ``data``."""
    raise NotImplementedError("Exercise: evaluate (docs/part1/training.md)")


def train(
    model: nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    config: TrainConfig,
    context_length: int,
    log: bool = True,
) -> dict[str, list[float]]:
    """Train ``model`` with AdamW, a cosine schedule and gradient clipping.

    For each step ``t = 0, ..., config.max_steps - 1``:

    1. set the learning rate of every parameter group to
       ``lr_cosine_schedule(t, lr_max, lr_min, warmup_steps, max_steps)``;
    2. draw a batch with :func:`get_batch`, compute the logits and the
       cross-entropy loss;
    3. back-propagate, clip the gradient norm to ``config.grad_clip``, take an
       optimizer step;
    4. every ``config.eval_interval`` steps (and after the last step),
       evaluate on ``val_data`` with :func:`evaluate`.

    Returns:
        A history dictionary with lists ``"step"``, ``"train_loss"``, ``"lr"``
        (one entry per step) and ``"val_step"``, ``"val_loss"``, ``"time"``
        (one entry per evaluation, time in seconds since the start).
    """
    device = config.device
    model.to(device)
    model.train()
    data_gen = torch.Generator().manual_seed(config.seed)
    eval_gen = torch.Generator().manual_seed(config.seed + 1)
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr_max,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    history: dict[str, list[float]] = {
        "step": [], "train_loss": [], "lr": [], "val_step": [], "val_loss": [], "time": []
    }
    start = time.time()

    def run_eval(step: int) -> None:
        val_loss = evaluate(
            model, val_data, config.batch_size, context_length, config.eval_batches, device, eval_gen
        )
        history["val_step"].append(step)
        history["val_loss"].append(val_loss)
        history["time"].append(time.time() - start)
        if log:
            print(
                f"step {step:6d} | val loss {val_loss:.4f} | val ppl {math.exp(val_loss):8.2f}"
                f" | {time.time() - start:7.1f}s",
                flush=True,
            )

    raise NotImplementedError("Exercise: train (docs/part1/training.md)")
    return history
