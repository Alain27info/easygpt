# 1. Characters as tokens

**Objective.** Turn a text into a sequence of integers, and sample random training examples from it.

## Language modeling

A language model is a probability distribution over sequences of tokens $x_1, \dots, x_n$.
By the chain rule of probability, it factorizes as a product of *next-token* distributions:

$$
p(x_1, \dots, x_n) = \prod_{t=1}^{n} p(x_t \mid x_1, \dots, x_{t-1}).
$$

A GPT-like model is a neural network $f_\theta$ that reads a prefix $x_{1:t}$ and outputs a vector of **logits** $o_t \in \mathbb{R}^V$ (one score per vocabulary token), which a softmax turns into probabilities: $p_\theta(\cdot \mid x_{1:t}) = \operatorname{softmax}(o_t)$.
It is trained by maximizing the likelihood of a large text, i.e. minimizing the average **negative log-likelihood** of each token given the previous ones.
Once trained, it generates text by sampling one token at a time and feeding it back as input.

Before any of that, we need to decide what a *token* is.
In this first part we choose the simplest option: **a token is a character**.

## The data

[Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) is a 1.1 MB text file of excerpts from Shakespeare's plays:

```text
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
```

It has 1,115,394 characters and only 65 distinct ones: the newline, the space, ten punctuation symbols, the digit `3`, and the 52 upper- and lower-case letters.
Download it with `uv run easygpt download shakespeare` (it is saved as `data/shakespeare.txt`).
The dataset was popularized by Andrej Karpathy's blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), which trains character-level models (recurrent networks, not Transformers) on it and visualizes what they learn.

## Exercise: a character tokenizer

A tokenizer maps text to a list of integer **ids** (`encode`) and back (`decode`).
For characters, the vocabulary is just the set of characters of the training text.
To make the ids reproducible, we sort the characters: id 0 is the character with the smallest Unicode code point (here `"\n"`), id 1 the next one (`" "`), and so on.

Implement `CharTokenizer` in [`src/easygpt/data.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/data.py):

- `__init__(text)` builds `self.chars`, the sorted list of distinct characters of `text` (the given `vocab_size` property relies on it), and whatever lookup table you need for encoding (a `dict` from character to id is the natural choice);
- `encode(text)` returns a `list[int]` (an unknown character raises a `KeyError`);
- `decode(ids)` returns the string.

```python
>>> from easygpt.data import CharTokenizer
>>> text = open("data/shakespeare.txt").read()
>>> tok = CharTokenizer(text)
>>> tok.vocab_size
65
>>> tok.encode("ROMEO:")
[30, 27, 25, 17, 27, 10]
>>> tok.decode([30, 27, 25, 17, 27, 10])
'ROMEO:'
```

!!! note "Characters vs bytes"
    A Python `str` is a sequence of Unicode code points, so our "characters" are code points.
    This works for Tiny Shakespeare, which uses only 65 of them, but on real text (accents, other scripts, emojis) the vocabulary would explode.
    We fix this in [Part II](../part2/bpe-training.md) by working with UTF-8 **bytes** and merging frequent byte sequences.

## Training examples

The whole text becomes one long array of ids, which we split into a training set (the first 90%) and a validation set (the last 10%).
How do we get training examples from it?

Take a window of $T + 1$ consecutive tokens starting at a position $s$.
Its first $T$ tokens are the input $x = (x_s, \dots, x_{s+T-1})$ and its last $T$ tokens, shifted by one, are the **targets** $y = (x_{s+1}, \dots, x_{s+T})$.
Numbering the positions of the window from 1, a single window gives $T$ prediction problems at once: predict $y_1$ from $x_1$, $y_2$ from $(x_1, x_2)$, ..., $y_T$ from $(x_1, \dots, x_T)$.
For instance with $T = 8$:

| input $x$ (prefix) | F | Fi | Fir | Firs | First | First␣ | First␣C | First␣Ci |
|---|---|---|---|---|---|---|---|---|
| target $y$ | i | r | s | t | ␣ | C | i | t |

The Transformer makes all 8 predictions in a single forward pass, while ensuring that the prediction at position $t$ only depends on $x_1, \dots, x_t$ (this is the *causal mask* of [page 3](attention.md)).
$T$ is the **context length**: the model never sees more than $T$ tokens at once.

A mini-batch is made of $B$ such windows, with start positions drawn uniformly at random: no need for padding, epochs, or shuffling.

## Exercise: random batches

Implement `get_batch(data, batch_size, context_length, device, generator)` in [`src/easygpt/data.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/data.py).
It returns a pair `(x, y)` of `torch.long` tensors of shape `(batch_size, context_length)`, on `device`: with `T = context_length`, `x[b] = data[s : s + T]` and `y[b] = data[s + 1 : s + T + 1]`, for a start position `s` drawn at random for each row `b`.

- `data` is a 1D NumPy array of ids (in Part II it will be a memory-mapped `uint16` array, so do not convert the *whole* array to a tensor, only the windows you sample).
- All valid start positions $s$ (those with $s + T + 1 \leq$ `len(data)`) must be equally likely.
- Draw the start positions with `torch.randint(..., generator=generator)` so that runs are reproducible.

```python
>>> import numpy as np
>>> from easygpt.data import get_batch
>>> x, y = get_batch(np.arange(100), batch_size=2, context_length=5)
>>> x  # random start positions: yours will differ
tensor([[44, 45, 46, 47, 48],
        [39, 40, 41, 42, 43]])
>>> y
tensor([[45, 46, 47, 48, 49],
        [40, 41, 42, 43, 44]])
```

Validate with `uv run pytest tests/test_data.py`.

## Questions

1. With $T = 128$, how many *different* training windows does the training set of Tiny Shakespeare contain? How many prediction problems?
2. Why is it convenient that all the windows have the same length?
3. Some characters are rare: count the occurrences of each character in the text (`collections.Counter`). Which are the rarest? What problem could that cause?
