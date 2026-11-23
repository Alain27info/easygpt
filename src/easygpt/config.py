"""Model and training configurations, with ready-made presets (given, not an exercise).

Docs: docs/part1/training.md and docs/part2/tinystories.md
"""

from dataclasses import asdict, dataclass, field, replace


def default_d_ff(d_model: int, multiple_of: int = 64) -> int:
    """``8/3 * d_model`` rounded up to a multiple of ``multiple_of``."""
    d_ff = int(8 * d_model / 3)
    return multiple_of * ((d_ff + multiple_of - 1) // multiple_of)


@dataclass
class ModelConfig:
    vocab_size: int = 65
    context_length: int = 128
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 384
    rope_theta: float | None = 10000.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 2000
    lr_max: float = 3e-3
    lr_min: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_batches: int = 20
    log_interval: int = 50
    seed: int = 0
    device: str = "cpu"


@dataclass
class Preset:
    """Everything needed to launch a run with ``easygpt train --preset NAME``."""

    dataset: str  # "shakespeare" or "tinystories"
    model: ModelConfig
    train: TrainConfig
    description: str = ""
    extra: dict = field(default_factory=dict)


PRESETS: dict[str, Preset] = {
    # ~0.87M parameters, a few minutes on a laptop CPU.
    "shakespeare": Preset(
        dataset="shakespeare",
        description="char-level Shakespeare, laptop CPU (~5-7 min)",
        model=ModelConfig(vocab_size=65, context_length=128, d_model=128, num_layers=4, num_heads=4, d_ff=384),
        train=TrainConfig(batch_size=32, max_steps=2000, lr_max=3e-3, lr_min=3e-4, warmup_steps=100),
    ),
    # ~10M parameters, for a Colab (free tier) T4 GPU.
    "shakespeare-gpu": Preset(
        dataset="shakespeare",
        description="char-level Shakespeare, GPU (e.g. Colab T4; not benchmarked yet)",
        model=ModelConfig(vocab_size=65, context_length=256, d_model=384, num_layers=6, num_heads=6, d_ff=1024),
        train=TrainConfig(
            batch_size=64, max_steps=5000, lr_max=1e-3, lr_min=1e-4, warmup_steps=200, eval_interval=500, device="cuda"
        ),
    ),
    # ~2.6M parameters (vocab 2048), CPU in ~20-25 min.
    "tinystories": Preset(
        dataset="tinystories",
        description="BPE (vocab 2048) TinyStories, laptop CPU (~25 min)",
        model=ModelConfig(vocab_size=2048, context_length=128, d_model=192, num_layers=4, num_heads=6, d_ff=512),
        train=TrainConfig(
            batch_size=32, max_steps=4000, lr_max=3e-3, lr_min=3e-4, warmup_steps=100, eval_interval=250
        ),
    ),
    # Bigger model for a Colab T4 GPU.
    "tinystories-gpu": Preset(
        dataset="tinystories",
        description="BPE (vocab 2048) TinyStories, GPU (e.g. Colab T4; not benchmarked yet)",
        model=ModelConfig(vocab_size=2048, context_length=256, d_model=384, num_layers=6, num_heads=6, d_ff=1024),
        train=TrainConfig(
            batch_size=64, max_steps=8000, lr_max=1e-3, lr_min=1e-4, warmup_steps=200, eval_interval=500, device="cuda"
        ),
    ),
}


def get_preset(name: str, **overrides) -> Preset:
    """Return a copy of ``PRESETS[name]``; ``overrides`` are applied to the train config."""
    preset = PRESETS[name]
    train = replace(preset.train, **{k: v for k, v in overrides.items() if v is not None})
    return replace(preset, model=replace(preset.model), train=train)


__all__ = ["ModelConfig", "TrainConfig", "Preset", "PRESETS", "get_preset", "default_d_ff", "asdict"]
