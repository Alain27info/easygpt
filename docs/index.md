# Build a small GPT from scratch

Large language models are, at their core, a surprisingly short program: a stack of attention and feed-forward layers trained to predict the next token.
This site invites you to **write that program yourself**, piece by piece, in the minimal PyTorch package [`easygpt`](https://github.com/svaiter/easygpt), and to train it on a laptop CPU.

Rather than the GPT-2 architecture of 2019, we build a *modern* decoder-only Transformer, the kind you find (scaled up) in [Llama](https://arxiv.org/abs/2302.13971), [Qwen](https://arxiv.org/abs/2412.15115) or [Mistral](https://arxiv.org/abs/2310.06825):

| | [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) / [nanoGPT](https://github.com/karpathy/nanoGPT) | **easygpt** (Llama-style) |
|---|---|---|
| normalization | LayerNorm (pre-norm) | **RMSNorm** (pre-norm) |
| positions | learned absolute embeddings | **rotary embeddings (RoPE)** |
| feed-forward | GELU MLP, $d_{ff} = 4d$ | **SwiGLU**, $d_{ff} \approx \frac{8}{3} d$ |
| biases | yes | **none** |
| optimizer | AdamW + cosine schedule with warmup + clipping | same |
| tokenizer | GPT-2 BPE (50k) | characters first, then **your own byte-level BPE** (2k) |

## Two stages

**[Part I: a character-level GPT](part1/data.md).**
To focus on the core (the model and its training), we start with the simplest possible tokenizer: one token per character.
On the 1.1 MB of [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), we train a 0.9M-parameter model that writes pseudo-Shakespeare after **about five minutes on a CPU**.

**[Part II: subwords and TinyStories](part2/bpe-training.md).**
Then we implement a byte-level **BPE tokenizer** (the algorithm behind the GPT-2/3/4 and Llama 3 tokenizers), train it on [TinyStories](https://arxiv.org/abs/2305.07759) with a 2048-token vocabulary, and use it to train a 2.6M-parameter model that writes grammatical short stories after **20–25 minutes on a CPU**.

| Page | You implement | Test file |
|---|---|---|
| [1. Characters as tokens](part1/data.md) | `CharTokenizer`, `get_batch` | `tests/test_data.py` |
| [2. A bigram baseline](part1/bigram.md) | `softmax`, `cross_entropy`, `Linear`, `Embedding`, `BigramLM` | `tests/test_bigram.py` |
| [3. Self-attention](part1/attention.md) | `causal_mask`, `scaled_dot_product_attention`, `MultiHeadSelfAttention` | `tests/test_attention.py` |
| [4. Rotary position embeddings](part1/rope.md) | `RotaryPositionalEmbedding` | `tests/test_rope.py` |
| [5. The Transformer](part1/transformer.md) | `silu`, `RMSNorm`, `SwiGLU`, `TransformerBlock`, `TransformerLM` | `tests/test_transformer.py` |
| [6. AdamW, schedule, clipping](part1/optimizer.md) | `AdamW`, `lr_cosine_schedule`, `clip_grad_norm` | `tests/test_optimizer.py` |
| [7. Training on Shakespeare](part1/training.md) | `save_checkpoint`, `load_checkpoint`, `evaluate`, `train` | `tests/test_training.py` |
| [8. Generating text](part1/sampling.md) | `top_p_filter`, `sample_next_token`, `generate` | `tests/test_sampling.py` |
| [9. Training a BPE tokenizer](part2/bpe-training.md) | `split_on_special_tokens`, `count_pretokens`, `count_pairs`, `merge_pair`, `train_bpe_naive`, `train_bpe` | `tests/test_bpe_training.py` |
| [10. Encoding and decoding](part2/tokenizer.md) | `Tokenizer.encode_pretoken`, `encode`, `decode` | `tests/test_tokenizer.py` |
| [11. Training on TinyStories](part2/tinystories.md) | nothing new: you run your code | |

## How it works: make the tests pass

The `easygpt` package in `src/easygpt/` ships with signatures and docstrings, but the body of each function you have to write raises `NotImplementedError`, with a pointer to the page that explains it:

```python
def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax along dimension ``dim``. ..."""
    raise NotImplementedError("Exercise: softmax (docs/part1/bigram.md)")
```

Your job is to replace each `raise` with an implementation until the test suite goes green.
Each page tells you which tests to run, for example:

```bash
uv run pytest tests/test_bigram.py
```

The tests compare your code with PyTorch's own implementations (`torch.nn.functional`, `torch.optim.AdamW`, ...) or check mathematical properties (causality, invariances), so a green test is a good sign that your implementation is right.
Everything else (command-line interface, dataset download, configuration presets) is given, and uses *your* functions: once the tests pass, `uv run easygpt train` trains *your* model.

!!! warning "The rules"
    The goal is to understand what is inside the black boxes, so in `src/easygpt` you may use
    any PyTorch tensor operation (`@`, `view`, `transpose`, `exp`, `sum`, `max`, `torch.where`, `torch.sigmoid`, ...), autograd (`.backward()`), `nn.Module`, `nn.Parameter`, `nn.ModuleList`, `nn.init` and the `torch.optim.Optimizer` base class,
    **but not** `torch.nn.functional`, the ready-made layers (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.MultiheadAttention`, ...), the built-in `softmax`/`log_softmax`/`logsumexp`, the ready-made optimizers and schedulers, `clip_grad_norm_`, or `einsum`/`einops`.
    `tests/test_rules.py` checks this for you.

## Prerequisites

- Python and the basics of PyTorch: tensors, broadcasting, `nn.Module`, `loss.backward()`. If `x.view(B, T, H, -1).transpose(1, 2)` scares you, read the [PyTorch tensor tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) first.
- Linear algebra, basic probability (softmax, log-likelihood), gradient descent.
- No GPU needed: everything runs on a laptop CPU. A free [Google Colab](https://colab.research.google.com) GPU lets you train bigger models (see [Setup](setup.md)).

## Resources and credits

This course borrows heavily from:

- [Stanford CS336, Assignment 1](https://github.com/stanford-cs336/assignment1-basics): the modern architecture, the BPE tokenizer and many exercises are adapted (and scaled down) from it.
- [nanoGPT](https://github.com/karpathy/nanoGPT) and the video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy: the character-level Shakespeare setup.
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023): small models can write coherent English if the data is simple enough.
- [easygrad](https://svaiter.github.io/easygrad/), the companion course on automatic differentiation (what `loss.backward()` does).

Original papers: [Attention is all you need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
[RMSNorm](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019),
[RoFormer](https://arxiv.org/abs/2104.09864) (Su et al., 2021),
[GLU variants](https://arxiv.org/abs/2002.05202) (Shazeer, 2020),
[AdamW](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2019),
[BPE for NMT](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016),
[LLaMA](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023).

Visual explanations of the whole architecture, useful at any point of the course:

- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): the data flow of attention and of a decoder-only language model, drawn step by step.
- Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/): the original paper interleaved with a line-by-line PyTorch implementation.
- 3Blue1Brown, [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt), [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention) and [How might LLMs store facts](https://www.3blue1brown.com/lessons/mlp): animated videos on embeddings, attention and feed-forward layers.
- Polo Club (Georgia Tech), [Transformer Explainer](https://poloclub.github.io/transformer-explainer/): runs GPT-2 in your browser on your own text and lets you inspect every intermediate step.
- Brendan Bycroft, [LLM Visualization](https://bbycroft.net/llm): a 3D walk through every operation of a small GPT.
