# easygpt

Build and train a small **modern GPT** (RMSNorm, RoPE, SwiGLU, pre-norm, AdamW, byte-level BPE) from scratch with PyTorch, on a laptop CPU.
You implement each piece yourself in a minimal Python package, guided by tests.

1. **Part I**: a character-level Transformer trained on Tiny Shakespeare (0.9M parameters, ~5 min on a CPU).
2. **Part II**: your own BPE tokenizer (2048 tokens) and a 2.6M-parameter model trained on TinyStories (~20–25 min on a CPU).

The whole course (explanations, exercises, questions) is on the website: **https://svaiter.github.io/easygpt/**

## Exercises

It is recommended to use [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                          # everything fails at first: that's expected
uv run pytest tests/test_data.py -x    # work page by page
```

Each `src/easygpt/*.py` module marks its exercises with `raise NotImplementedError(...)` and a pointer to the docs page that explains the task.
Once the tests pass, the `easygpt` command trains *your* model:

```bash
uv run easygpt download shakespeare
uv run easygpt train --preset shakespeare
uv run easygpt generate checkpoints/shakespeare.pt --prompt "ROMEO:"
```

## Docs site

```bash
uv run zensical serve   # preview the docs site locally
uv run zensical build   # build the static site into site/
```

## Resources

- [Stanford CS336, Assignment 1](https://github.com/stanford-cs336/assignment1-basics): the architecture and most exercises are adapted from it.
- [nanoGPT](https://github.com/karpathy/nanoGPT): the character-level Shakespeare setup.
- [TinyStories](https://arxiv.org/abs/2305.07759): the dataset of Part II.
- [easygrad](https://github.com/svaiter/easygrad): the companion course on automatic differentiation.
