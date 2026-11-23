# 5. The Transformer

**Objective.** Assemble the full model: RMSNorm, the SwiGLU feed-forward network, the pre-norm Transformer block, and the Transformer language model.

## Architecture

<figure style="margin:1em 0">
<svg viewBox="0 0 760 536" width="760" height="536" role="img" aria-label="Architecture of the Transformer language model. Left: token ids go through the embedding, num_layers Transformer blocks, a final RMSNorm and the LM head to give logits. Right: inside one pre-norm block, the residual stream goes straight from input to output; a first branch applies RMSNorm then causal multi-head attention with RoPE and adds its result to the stream, a second branch applies RMSNorm then the SwiGLU feed-forward network and adds its result." style="max-width:100%;height:auto;font-size:13px" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><defs><marker id="eg-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker><marker id="eg-arr-res" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#2f7fe0"/></marker></defs><text x="140" y="90" text-anchor="middle">logits <tspan font-size="11" opacity="0.75">(B, T, V)</tspan></text><line x1="140" y1="112" x2="140" y2="104" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><rect x="50" y="112" width="180" height="36" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.3"/><text x="140.0" y="134.5" text-anchor="middle">Linear (LM head)</text><line x1="140" y1="180" x2="140" y2="150" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><rect x="50" y="180" width="180" height="36" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.3"/><text x="140.0" y="202.5" text-anchor="middle">RMSNorm</text><line x1="140" y1="250" x2="140" y2="218" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><text x="148" y="238" font-size="11" opacity="0.75">(B, T, D)</text><rect x="40" y="250" width="200" height="100" rx="8" fill="currentColor" fill-opacity="0.10" stroke="currentColor" stroke-width="1.6"/><text x="140" y="294" text-anchor="middle" font-weight="600">Transformer block</text><text x="140" y="316" text-anchor="middle" font-size="12" opacity="0.8">× num_layers</text><line x1="140" y1="384" x2="140" y2="352" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><text x="148" y="372" font-size="11" opacity="0.75">(B, T, D)</text><rect x="50" y="384" width="180" height="36" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.3"/><text x="140.0" y="406.5" text-anchor="middle">Embedding</text><line x1="140" y1="446" x2="140" y2="422" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><text x="140" y="462" text-anchor="middle">token ids <tspan font-size="11" opacity="0.75">(B, T)</tspan></text><line x1="240" y1="250" x2="320" y2="12" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4" opacity="0.4"/><line x1="240" y1="350" x2="320" y2="522" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4" opacity="0.4"/><rect x="320" y="12" width="428" height="510" rx="10" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/><text x="736" y="36" text-anchor="end" font-weight="600">one block (pre-norm)</text><line x1="430" y1="496" x2="430" y2="311" stroke="#2f7fe0" stroke-width="3"/><line x1="430" y1="289" x2="430" y2="121" stroke="#2f7fe0" stroke-width="3"/><line x1="430" y1="99" x2="430" y2="58" stroke="#2f7fe0" stroke-width="3" marker-end="url(#eg-arr-res)"/><circle cx="430" cy="300" r="11" fill="none" stroke="#2f7fe0" stroke-width="2"/><line x1="424" y1="300" x2="436" y2="300" stroke="#2f7fe0" stroke-width="2"/><line x1="430" y1="294" x2="430" y2="306" stroke="#2f7fe0" stroke-width="2"/><circle cx="430" cy="110" r="11" fill="none" stroke="#2f7fe0" stroke-width="2"/><line x1="424" y1="110" x2="436" y2="110" stroke="#2f7fe0" stroke-width="2"/><line x1="430" y1="104" x2="430" y2="116" stroke="#2f7fe0" stroke-width="2"/><text x="430" y="44" text-anchor="middle">output <tspan font-size="11" opacity="0.75">(B, T, D)</tspan></text><text x="430" y="512" text-anchor="middle">input <tspan font-size="11" opacity="0.75">(B, T, D)</tspan></text><text x="0" y="0" transform="translate(418 400) rotate(-90)" text-anchor="middle" font-size="12" fill="#2f7fe0">residual stream</text><text x="0" y="0" transform="translate(418 205) rotate(-90)" text-anchor="middle" font-size="12" fill="#2f7fe0">residual stream</text><circle cx="430" cy="474" r="3.5" fill="#2f7fe0"/><polyline points="430,474 600,474 600,448" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><rect x="540" y="410" width="120" height="36" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.3"/><text x="600.0" y="432.5" text-anchor="middle">RMSNorm</text><line x1="600" y1="410" x2="600" y2="382" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><rect x="495" y="330" width="210" height="50" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.3"/><text x="600.0" y="352.0" text-anchor="middle">causal multi-head attention</text><text x="600.0" y="369.0" text-anchor="middle" font-size="11" opacity="0.75">RoPE on queries and keys</text><polyline points="600,330 600,300 443,300" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><text x="450" y="292" font-size="12" opacity="0.8">positions communicate</text><circle cx="430" cy="270" r="3.5" fill="#2f7fe0"/><polyline points="430,270 600,270 600,244" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><rect x="540" y="206" width="120" height="36" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.3"/><text x="600.0" y="228.5" text-anchor="middle">RMSNorm</text><line x1="600" y1="206" x2="600" y2="178" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><rect x="510" y="140" width="180" height="36" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.3"/><text x="600.0" y="162.5" text-anchor="middle">SwiGLU feed-forward</text><polyline points="600,140 600,110 443,110" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#eg-arr)"/><text x="450" y="102" font-size="12" opacity="0.8">each position on its own</text></svg>
</figure>

*Left: the whole model, from token ids to logits. Right: one pre-norm block. The residual stream (blue) runs from the embedding to the output untouched; each sub-layer reads a normalized copy of it and adds its result back.*

The token ids are embedded into vectors of dimension $d$ (`d_model`).
This sequence of vectors, the **residual stream**, then goes through `num_layers` identical blocks.
Each block *adds* two contributions to the stream:

$$
\begin{aligned}
z &= x + \operatorname{MHA}(\operatorname{RMSNorm}(x)) && \text{positions communicate,}\\
y &= z + \operatorname{FFN}(\operatorname{RMSNorm}(z)) && \text{each position computes on its own.}
\end{aligned}
$$

A final RMSNorm and a linear layer (the **LM head**, $W \in \mathbb{R}^{V \times d}$) then produce the logits.

This is the **pre-norm** arrangement: the normalization is applied to the *input* of each sub-layer, and the residual stream itself is never normalized (except at the very end).
The [original Transformer](https://arxiv.org/abs/1706.03762) was post-norm, $y = \operatorname{Norm}(x + \operatorname{Sublayer}(x))$; pre-norm keeps an identity path from the embeddings to the output, which makes deep networks much easier to train ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745); GPT-2 and most recent LLMs use it).

!!! tip "Visual explanations"
    - Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): a decoder-only Transformer, block by block (with the GPT-2 components: LayerNorm, learned positions, GELU).
    - Brendan Bycroft, [LLM Visualization](https://bbycroft.net/llm): a 3D, step-by-step walk through every operation of a small GPT, including the residual stream, the norms and the MLP.
    - 3Blue1Brown, [How might LLMs store facts](https://www.3blue1brown.com/lessons/mlp): what the feed-forward layers compute.

## RMSNorm

[Layer normalization](https://arxiv.org/abs/1607.06450) rescales each activation vector, so that every sub-layer receives inputs of a controlled scale.
RMSNorm ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) is a simpler variant, with neither centering nor bias, used by Llama: for $a \in \mathbb{R}^d$,

$$
\operatorname{RMSNorm}(a)_i = \frac{a_i}{\operatorname{RMS}(a)}\, g_i, \qquad \operatorname{RMS}(a) = \sqrt{\frac{1}{d}\sum_{j=1}^d a_j^2 + \varepsilon},
$$

with a learnable gain $g \in \mathbb{R}^d$ initialized to ones and $\varepsilon = 10^{-5}$.
In low precision (float16 or bfloat16, common on GPUs), squares can overflow (float16) or lose accuracy, so the computation is done in float32 and the result is cast back to the input dtype (this part is given).

## SwiGLU feed-forward

The original Transformer feed-forward network is $W_2\, \operatorname{ReLU}(W_1 x)$ with a hidden dimension $d_\text{ff} = 4d$ (GPT-2 uses GELU instead of ReLU).
Modern LLMs use a **gated** variant ([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083); [Shazeer, 2020](https://arxiv.org/abs/2002.05202)). With the SiLU activation ([Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415); [Elfwing et al., 2017](https://arxiv.org/abs/1702.03118); also called Swish) $\operatorname{SiLU}(x) = x\, \sigma(x) = \frac{x}{1 + e^{-x}}$, a smooth version of ReLU,

$$
\operatorname{FFN}(x) = W_2 \big(\operatorname{SiLU}(W_1 x) \odot W_3 x\big), \qquad W_1, W_3 \in \mathbb{R}^{d_\text{ff} \times d},\ W_2 \in \mathbb{R}^{d \times d_\text{ff}},
$$

where $\odot$ is the element-wise product: the branch $\operatorname{SiLU}(W_1 x)$ acts as a soft, input-dependent gate on the linear branch $W_3 x$.
With three matrices instead of two, keeping the parameter count of the $4d$ MLP requires $d_\text{ff} \approx \frac{8}{3} d$, which we round up to a multiple of 64 for hardware efficiency (`easygpt.config.default_d_ff`).
Why does it work better? "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence" ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)). See [Going further](../experiments.md) to measure it yourself.

!!! question "Exercise: `silu`, `RMSNorm`, `SwiGLU`"
    - `silu(x)` in [`src/easygpt/functional.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/functional.py) (you may use `torch.sigmoid`).
    - `RMSNorm` in [`src/easygpt/layers.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/layers.py): the gain is `self.weight`, of shape `(d_model,)`; normalize over the last dimension.
    - `SwiGLU` in [`src/easygpt/layers.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/layers.py): three `Linear` layers named `w1`, `w2`, `w3` as in the formula.

## Putting it together

!!! question "Exercise: `TransformerBlock` and `TransformerLM`"
    In [`src/easygpt/model.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/model.py):

    - `TransformerBlock(d_model, num_heads, d_ff, rope)`: sub-modules `ln1` and `ln2` (`RMSNorm`), `attn` (`MultiHeadSelfAttention`, which receives `rope`) and `ffn` (`SwiGLU`); `forward` computes the two residual updates above.
    - `TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)`: sub-modules `token_embeddings`, `layers` (an `nn.ModuleList` of blocks), `ln_final` and `lm_head`.
      Create **one** `RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)` and pass it to every block; if `rope_theta` is `None`, pass `None` (no positional information at all, for an ablation).

    The sub-module names matter: they define the keys of the `state_dict`, used by the tests and by checkpoints.

Validate with `uv run pytest tests/test_transformer.py`.
The tests compare your model with an independent implementation built from `torch.nn.functional`, and check that it is causal.

## Counting parameters

For the CPU Shakespeare preset ($V = 65$, $d = 128$, 4 layers, 4 heads, $d_\text{ff} = 384$, $T = 128$):

```python
>>> import torch
>>> from easygpt.model import TransformerLM, count_parameters
>>> model = TransformerLM(vocab_size=65, context_length=128, d_model=128, num_layers=4, num_heads=4, d_ff=384)
>>> count_parameters(model)
869760
>>> x = torch.randint(0, 65, (2, 128))
>>> model(x).shape
torch.Size([2, 128, 65])
```

The model has 0.87M parameters, almost all in the blocks: per block, $4d^2$ for attention, $3 d\, d_\text{ff}$ for the feed-forward and $2d$ for the norms.

Before training, a sanity check: with random weights, the predictions should be roughly uniform, so the loss should be close to $\log 65 \approx 4.17$.

```python
>>> from easygpt.functional import cross_entropy
>>> cross_entropy(model(x[:, :-1]), x[:, 1:])
tensor(4.8..., grad_fn=<MeanBackward0>)
```

It is a bit larger because the initial logits are not zero: the final RMSNorm outputs vectors with unit RMS, and the LM head weights have standard deviation $\sqrt{2 / (128 + 65)} \approx 0.1$, so each logit has standard deviation about $\sqrt{128} \times 0.1 \approx 1.15$ (see question 1 of [page 2](bigram.md)). A loss of 10 or 50 at initialization would signal a bug.

## Questions

1. Write the number of parameters of `TransformerLM` as a function of $V$, $d$, $L$ (layers) and $d_\text{ff}$. Check it against `count_parameters` for the two Shakespeare presets (`uv run easygpt presets`).
2. **FLOPs.** A matrix product between an $m \times n$ and an $n \times p$ matrix costs $2mnp$ floating-point operations. List the matrix products of a forward pass for a sequence of $T$ tokens, and count their FLOPs. Which part dominates for the CPU Shakespeare preset? For the TinyStories preset ($V = 2048$, $d = 192$, $T = 128$)? For a GPT-2 XL-sized model with our architecture ($V = 50\,257$, $T = 1024$, $L = 48$, $d = 1600$, $H = 25$, $d_\text{ff} = 4288$)?
3. A rule of thumb says that training costs about $6N$ FLOPs per token for a model with $N$ parameters ($2N$ forward, $4N$ backward). Where does it come from, and what does it neglect?
4. Why is the final `RMSNorm` needed in a pre-norm Transformer but not in a post-norm one?
