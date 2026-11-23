# Setup

## On your computer (recommended)

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment (install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`).

```bash
git clone https://github.com/svaiter/easygpt
cd easygpt
uv sync
```

`uv sync` creates a virtual environment in `.venv` with PyTorch (the CPU build on Linux and Windows, which is much smaller than the CUDA one), NumPy, `regex`, `pytest`, and the tools to build this website.
Then check that everything is in place:

```bash
uv run pytest
```

Almost every test fails with a `NotImplementedError`: this is expected, since you have not done the exercises yet!
While working on a page, run only its test file, and stop at the first failure with `-x`:

```bash
uv run pytest tests/test_data.py -x
```

Download the first dataset (1.1 MB):

```bash
uv run easygpt download shakespeare
```

!!! tip "Without uv"
    A plain virtual environment works too; then drop the `uv run` prefix from all commands.
    On Linux and Windows, install the CPU build of PyTorch first, otherwise pip downloads the much larger CUDA build:

    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -e . pytest
    ```

## On Google Colab (optional, for a free GPU)

Everything in this course runs on a CPU. If you want to train bigger models, a free Colab GPU (Runtime → Change runtime type → T4 GPU) helps.
In a notebook cell:

```python
!git clone https://github.com/YOUR_USERNAME/easygpt   # your fork, with your code
%cd easygpt
!pip install -e .   # PyTorch is already installed on Colab
!pytest -q
!easygpt train --preset shakespeare-gpu
```

!!! warning
    Colab sessions are temporary: push your code to your fork (or mount your Google Drive) before closing the tab.

## The repository

```text
easygpt/
├── docs/               this website (zensical)
├── src/easygpt/        the package you complete
│   ├── data.py         CharTokenizer, get_batch                     (page 1)
│   ├── functional.py   softmax, cross_entropy, silu                 (pages 2, 5)
│   ├── layers.py       Linear, Embedding, RMSNorm, SwiGLU           (pages 2, 5)
│   ├── attention.py    causal_mask, attention, RoPE, multi-head     (pages 3, 4)
│   ├── model.py        BigramLM, TransformerBlock, TransformerLM    (pages 2, 5)
│   ├── optim.py        SGD (given), AdamW, schedule, clipping       (page 6)
│   ├── train.py        checkpoints, evaluate, train                 (page 7)
│   ├── generate.py     temperature, top-p, generate                 (page 8)
│   ├── bpe.py          BPE training and Tokenizer                   (pages 9, 10)
│   ├── config.py       model/training presets                       (given)
│   ├── datasets.py     downloads                                    (given)
│   └── cli.py          the `easygpt` command                        (given)
└── tests/              one test file per page
```

The command-line tool `easygpt` (defined in `cli.py`) glues your functions together:

```bash
uv run easygpt --help
uv run easygpt presets                        # list the model/training presets
uv run easygpt train --preset shakespeare     # page 7
uv run easygpt generate checkpoints/shakespeare.pt --prompt "ROMEO:"   # page 8
```

## Conventions

- **Shapes.** We write `B` for the batch size, `T` for the sequence length (number of tokens), `D` for the model dimension `d_model`, `H` for the number of heads, `V` for the vocabulary size. Docstrings always give the expected shapes: read them!
- **Row vectors.** PyTorch puts the batch dimensions first and the features last, so a linear layer with weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ maps a batch of row vectors $x \in \mathbb{R}^{B \times d_{\text{in}}}$ to $y = x W^\top \in \mathbb{R}^{B \times d_{\text{out}}}$. In formulas we often write $y = Wx$ for a single column vector: it is the same map.
- **Leading dimensions.** Most functions should work with any number of leading "batch-like" dimensions: `x @ W.T` works for `x` of shape `(D_in,)`, `(B, D_in)` or `(B, T, D_in)`. Reductions should use `dim=-1` and `keepdim=True` rather than hard-coded axes.
- **No einsum.** We use plain PyTorch: `@` (batched matrix product), `view`/`reshape`, `transpose`, broadcasting.

??? note "A five-minute PyTorch refresher"
    ```python
    import torch
    x = torch.randn(2, 3, 4)          # (B, T, D)
    W = torch.randn(5, 4)             # (D_out, D_in)
    y = x @ W.T                       # (2, 3, 5): @ broadcasts over leading dims
    m = x.max(dim=-1, keepdim=True)   # .values has shape (2, 3, 1), .indices too
    z = x - m.values                  # broadcasting (2, 3, 4) - (2, 3, 1)
    h = x.view(2, 3, 2, 2)            # split the last dim into 2 heads of size 2
    h = h.transpose(1, 2)             # (B, H, T, D/H)
    back = h.transpose(1, 2).reshape(2, 3, 4)  # reshape, not view: a transposed tensor is in general not contiguous
    ids = torch.tensor([[0, 2], [1, 1]])
    table = torch.randn(3, 4)
    table[ids].shape                  # (2, 2, 4): integer indexing = embedding lookup
    ```
