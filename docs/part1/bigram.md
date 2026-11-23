# 2. A bigram baseline

**Objective.** Build and train the simplest possible neural language model, end to end: logits, softmax, cross-entropy, a training loop, and sampling. Along the way, implement the `Linear` and `Embedding` layers that the Transformer will reuse.

## Logits and softmax

A language model outputs, for each position, a vector of **logits** $o \in \mathbb{R}^V$: one unnormalized score per token of the vocabulary.
The **softmax** turns it into a probability distribution:

$$
\operatorname{softmax}(o)_i = \frac{\exp(o_i)}{\sum_{j=1}^{V} \exp(o_j)}.
$$

Computed naively, $\exp(o_i)$ overflows to `inf` as soon as $o_i \gtrsim 89$ in float32 (and `inf / inf = nan`).
The fix relies on the shift invariance of the softmax, $\operatorname{softmax}(o) = \operatorname{softmax}(o - c)$ for any constant $c$: subtract $c = \max_j o_j$ first, so that the largest term is $\exp(0) = 1$.

!!! question "Exercise: `softmax`"
    Implement `softmax(x, dim=-1)` in [`src/easygpt/functional.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/functional.py).
    It must work along any dimension `dim` of a tensor of any shape, and be stable for large inputs.
    Use `x.max(dim=dim, keepdim=True).values`, `torch.exp` and `.sum(dim=dim, keepdim=True)`, not `torch.softmax`.

## Cross-entropy

Given logits $o$ and the index $y$ of the true next token, the loss is the **negative log-likelihood** of $y$, also called the **cross-entropy**:

$$
\ell(o, y) = -\log \operatorname{softmax}(o)_y = \underbrace{\log \sum_{j=1}^{V} \exp(o_j)}_{\text{log-sum-exp}} - o_y.
$$

Compute it with the right-hand side: never take the `log` of a softmax that may have underflowed to 0.
The log-sum-exp is stabilized with the same trick, $\log \sum_j \exp(o_j) = c + \log \sum_j \exp(o_j - c)$ with $c = \max_j o_j$.

Over a batch of $B$ sequences of $T$ tokens, we average the loss over the $B \times T$ predictions.
The exponential of the average loss is the **perplexity**: a model with perplexity $k$ is "as uncertain as" a uniform choice among $k$ tokens.
A model that knows nothing predicts the uniform distribution, with loss $\log V$ (perplexity $V$): for our 65 characters, $\log 65 \approx 4.17$.

!!! tip "Visual explanation"
    Chris Olah's [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) explains entropy and cross-entropy with pictures: the cross-entropy is the average number of bits needed to encode the true data with a code optimized for the model's distribution.

!!! question "Exercise: `cross_entropy`"
    Implement `cross_entropy(logits, targets)` in [`src/easygpt/functional.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/functional.py).
    `logits` has shape `(..., V)` and `targets` has shape `(...)` (integers); return the scalar mean loss.

    To pick the logit of the target at every position, use `logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)` (or advanced indexing).

## Linear and Embedding layers

Two layers are enough for the bigram model and almost everything else.

**Linear.** A linear map without bias, $y = W x$, i.e. `y = x @ W.T` for a batch of row vectors, where $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ is stored as `self.weight` (same convention as `torch.nn.Linear`).
Modern LLMs ([PaLM](https://arxiv.org/abs/2204.02311), [Llama](https://arxiv.org/abs/2302.13971), ...) drop biases: they do not help at scale, and PaLM reports better training stability without them.
Initialization matters: we draw $W_{ij} \sim \mathcal{N}(0, \sigma^2)$ with $\sigma^2 = \frac{2}{d_\text{in} + d_\text{out}}$ ([Glorot initialization](https://proceedings.mlr.press/v9/glorot10a.html): a compromise between preserving the variance of the activations in the forward pass and that of the gradients in the backward pass), truncated at $\pm 3\sigma$.

**Embedding.** A lookup table $E \in \mathbb{R}^{V \times d}$ stored as `self.weight`: token $i$ is mapped to the row $E_i$.
It is a linear layer applied to one-hot vectors ($\mathbb{1}_i^\top E = E_i$), but indexing `self.weight[token_ids]` is much cheaper than a matrix product.
We initialize $E_{ij} \sim \mathcal{N}(0, 1)$ truncated at $\pm 3$.

!!! question "Exercise: `Linear` and `Embedding`"
    Implement `Linear` and `Embedding` in [`src/easygpt/layers.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/layers.py).

    - The skeleton (subclass of `nn.Module`, call to `super().__init__()`) is given: wrap the weight in an `nn.Parameter` and store it as `self.weight` (the name matters: tests and checkpoints rely on it).
    - Create the tensor with `torch.empty(..., device=device, dtype=dtype)` and fill it with `nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)`.
    - `forward` must accept any number of leading dimensions.

## The bigram model

The simplest language model only looks at the *current* token to predict the next one:
$p(x_{t+1} \mid x_{1:t}) = p(x_{t+1} \mid x_t)$.
A table of $V \times V$ logits is enough: row $a$ holds the logits of the token that follows $a$.
That is exactly an `Embedding(V, V)`!

!!! tip "Visual explanation"
    Andrej Karpathy's video [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) builds exactly this model on a dataset of first names: first by counting the pairs of consecutive characters (and plotting the table of counts), then as a one-layer neural network trained by gradient descent.

!!! question "Exercise: `BigramLM`"
    Implement `BigramLM` in [`src/easygpt/model.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/model.py): an `Embedding(vocab_size, vocab_size)` stored as `self.logits_table`, whose `forward` maps ids of shape `(B, T)` to logits of shape `(B, T, V)`.

Validate everything with `uv run pytest tests/test_bigram.py`.

## Train it

We have all the pieces of a training loop.
We do not have an optimizer yet (that is [page 6](optimizer.md)), so for now we borrow PyTorch's `torch.optim.AdamW`.
Run this in a notebook, a Python console or a script:

```python
import numpy as np
import torch
from easygpt.data import CharTokenizer, get_batch
from easygpt.functional import cross_entropy, softmax
from easygpt.model import BigramLM

text = open("data/shakespeare.txt").read()
tok = CharTokenizer(text)
ids = np.array(tok.encode(text))
n = int(0.9 * len(ids))
train_data, val_data = ids[:n], ids[n:]

torch.manual_seed(0)
model = BigramLM(tok.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)  # temporary: page 6 replaces it

for step in range(1501):
    x, y = get_batch(train_data, batch_size=32, context_length=64)
    loss = cross_entropy(model(x), y)
    optimizer.zero_grad()  # reset the gradients accumulated by the previous step
    loss.backward()        # compute d loss / d parameters
    optimizer.step()       # update the parameters
    if step % 500 == 0:
        with torch.no_grad():
            xv, yv = get_batch(val_data, batch_size=256, context_length=64)
            print(f"{step:5d} | train {loss.item():.3f} | val {cross_entropy(model(xv), yv).item():.3f}")
```

```text
    0 | train 4.497 | val 4.395
  500 | train 2.488 | val 2.486
 1000 | train 2.455 | val 2.501
 1500 | train 2.523 | val 2.481
```

The loss drops from about 4.5 to about 2.5 in a few seconds, and then stops improving: the model has learned the bigram statistics of the text, and cannot do better.
Now let us **sample** from it: start from a newline, and repeatedly draw the next character from $\operatorname{softmax}(o)$.

```python
g = torch.Generator().manual_seed(0)
generated = tok.encode("\n")
for _ in range(300):
    logits = model(torch.tensor([generated[-1:]]))[0, -1]  # logits after the last token
    probs = softmax(logits)
    generated.append(torch.multinomial(probs, 1, generator=g).item())
print(tok.decode(generated))
```

```text title="(your sample will differ)"
th remofad hexpselteredeca meaman youd INAlste grelofuroy hifotherere hath thig berelisseanongilost atho y or w ke e alcot idarthofr nd; my f tsoo we w y t f aim whioervoto,
Pre co n s uzed
I s h. fo INVeprf pifie-wily mpoennd ssay ed tcodeld fllome igle ind t he t,
As toinobe y tonoon gst st hayowa
```

Not quite Shakespeare, but the "words" have plausible lengths, there are capitals after newlines, and common pairs such as `th` or `he` show up.
To do better, a token must be able to look at *more* than one previous token: this is what attention does.

## Questions

1. At initialization the loss is about 4.5, larger than $\log 65 \approx 4.17$. Why? (Hint: the initial logits are i.i.d. $\mathcal{N}(0, 1)$; what is $\mathbb{E}\left[\log \sum_j e^{o_j}\right] - \mathbb{E}[o_y]$ roughly equal to when $V$ is large?)
2. The bigram model can be fitted *without* gradient descent: count the pairs $(a, b)$ of consecutive characters in the training set and set $p(b \mid a) = \frac{\text{count}(a, b)}{\text{count}(a)}$. Compute the training loss of this count-based model (you should find about 2.45). Why is it the best any bigram model can do on the training set?
3. How many parameters does the bigram model have? How many would a "trigram" table $p(x_{t+1} \mid x_{t-1}, x_t)$ need? And an $n$-gram table? Why is this approach hopeless for long contexts?
