# 9. Training a BPE tokenizer

**Objective.** Implement byte-level byte-pair encoding (BPE), the tokenization algorithm of GPT-2, GPT-4 and Llama 3, and train it on TinyStories.

## Why not characters?

Characters worked for Shakespeare, but they have two problems.

- **Long sequences.** A word takes 5–6 tokens. The cost of attention grows quadratically with the sequence length, and a context of $T = 128$ characters only covers about 20–25 words.
- **Open vocabulary.** Real text contains accents, other scripts, emojis... There are about 160,000 Unicode characters: too many, and most are very rare.

Word-level tokenizers have the opposite problems: a huge vocabulary, and unknown words at test time.
**Subword** tokenization is the middle ground: frequent words are single tokens, rare words are split into pieces, and any string can be encoded.

## Unicode and UTF-8

A Python `str` is a sequence of Unicode **code points** (`ord("é") == 233`, `ord("牛") == 29275`).
The **UTF-8** encoding turns each code point into 1 to 4 **bytes** (integers in $[0, 255]$): ASCII characters take 1 byte, `é` takes 2, `牛` takes 3, most emojis take 4.

```python
>>> s = "héllo 牛!"
>>> list(s.encode("utf-8"))
[104, 195, 169, 108, 108, 111, 32, 231, 137, 155, 33]
>>> len(s), len(s.encode("utf-8"))
(8, 11)
>>> bytes([195, 169]).decode("utf-8")
'é'
```

**Byte-level** BPE starts from the 256 possible bytes: any text can be encoded, and there is no unknown token.
It then *merges* frequent sequences of bytes into new tokens.

!!! tip "Further reading"
    Nikita Prokopov, [The Absolute Minimum Every Software Developer Must Know About Unicode in 2023](https://tonsky.me/blog/unicode/): code points, UTF-8 and grapheme clusters, with many examples.

!!! question "Questions: bytes"
    1. Why train on UTF-8 bytes rather than on UTF-16 or UTF-32 code units? Compare `len(s.encode(...))` for an English sentence.
    2. Why is the following function wrong? Find an input on which it fails.
        ```python
        def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
            return "".join([bytes([b]).decode("utf-8") for b in bytestring])
        ```
    3. Find a two-byte sequence that is not valid UTF-8.

## The BPE algorithm

BPE started as a data compression algorithm (Gage, 1994), was adapted to subword tokenization for machine translation by [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909), and made byte-level by [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

**Vocabulary initialization.** The 256 single bytes are the tokens 0 to 255.

**Merges.** Repeatedly: count all pairs of adjacent tokens in the corpus, take the most frequent pair $(A, B)$, create a new token $AB$ and replace every occurrence of $A, B$ with $AB$.
After $k$ merges, the vocabulary has $256 + k$ tokens. The ordered list of merges *is* the tokenizer.

**Pre-tokenization.** Counting pairs over the raw corpus at every merge is slow, and would create tokens like `"dog."` or `"g! T"` that cross word boundaries.
So we first split the text into **pre-tokens** (roughly words, with their leading space) with GPT-2's regular expression, and never merge across pre-tokens.
Since the same word occurs many times, we only need the **count of each distinct pre-token**: `{" the": 41290, " cat": 312, ...}`.
The pattern and the function `pretokenize` are given in [`src/easygpt/bpe.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/bpe.py):

```python
>>> from easygpt.bpe import pretokenize
>>> pretokenize("Once upon a time, there were 2 cats. They'll play!")
['Once', ' upon', ' a', ' time', ',', ' there', ' were', ' 2', ' cats', '.', ' They', "'ll", ' play', '!']
```

We represent a pre-token as a tuple of `bytes` objects, one per current token: `" cat"` starts as `(b' ', b'c', b'a', b't')`. (Python has no "byte" type: a single byte is a `bytes` of length 1.)

**Special tokens.** Strings like `<|endoftext|>` mark the boundary between two documents. They must never be split into bytes or merged with their neighbors: we cut the text on special tokens *before* pre-tokenization, drop them while counting, and add them to the vocabulary at the end.

**Ties.** When several pairs are equally frequent, we take the **lexicographically greatest** pair, comparing `bytes` objects (`max` on tuples does that: `(b"s", b"t") > (b"e", b"s")`). This makes the result deterministic.

### A worked example

Take the corpus `"low low low low low lower lower widest widest widest newest newest newest newest newest newest"`.
Pre-tokenization gives the counts
`{"low": 1, " low": 4, " lower": 2, " widest": 3, " newest": 6}`.

1. Pair counts: `(e, s)` and `(s, t)` appear 3 + 6 = 9 times, more than any other pair. Tie: `(s, t)` is greater. Merge `s t` → `st`: `" newest"` becomes `( , n, e, w, e, st)`.
2. Now `(e, st)` has count 9: merge into `est`.
3. `(o, w)` and `(l, o)` both have count 1 + 4 + 2 = 7: merge `o w`, then `l ow`.
4. ... and so on: the first ten merges are
   `s t`, `e st`, `o w`, `l ow`, `w est`, `n e`, `ne west`, `␣ newest`, `␣ low`, `w i`.

The vocabulary now contains ` newest` and ` low` as single tokens: frequent words become tokens, with their leading space.

!!! tip "Visual explanations"
    - Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (video) and its companion code [minbpe](https://github.com/karpathy/minbpe): byte-level BPE from scratch, with the GPT-2/GPT-4 regular expressions and special tokens.
    - Hugging Face LLM course, [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5): the merges of a toy corpus, step by step.

## Exercises

All in [`src/easygpt/bpe.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/bpe.py); see the docstrings for examples.

!!! question "Exercise: the building blocks"
    - `split_on_special_tokens(text, special_tokens)`: split around special tokens, keeping them as separate items. Build a regex `"(tok1|tok2|...)"` with `regex.escape` (`|` is a regex metacharacter!), sort the special tokens by decreasing length so that the longest wins when they overlap, and use `regex.split` (a capturing group keeps the separators). Drop empty strings. With no special tokens, return the text as a single item (an empty pattern would split it between every character).
    - `count_pretokens(text, special_tokens)`: a dictionary mapping each pre-token, as a tuple of single bytes, to its count; special tokens are boundaries and are not counted. Count strings first (`collections.Counter`) and convert to bytes at the end: it is much faster.
    - `count_pairs(word_counts)`: the count of each adjacent pair, weighted by the count of the word.
    - `merge_pair(word, pair)`: replace the occurrences of `pair` in `word` (a tuple of bytes), scanning from left to right: `merge_pair((a, a, a), (a, a)) == (aa, a)`.

!!! question "Exercise: `train_bpe_naive`"
    Implement the algorithm above literally: at each merge, recount all the pairs with `count_pairs`, take the best one (`max(pair_counts, key=lambda p: (pair_counts[p], p))`), record it, add the new token to the vocabulary, and apply `merge_pair` to every word. Stop early if no pair is left.
    The vocabulary layout is fixed: ids 0–255 are the bytes, then one id per merge in order, then the special tokens (given).

Validate with `uv run pytest tests/test_bpe_training.py -k "not fast"` (this deselects the tests of the fast version below).

### Faster training

The naive version recounts *all* the pairs at every merge, although a merge only changes the pairs of the words that contain the merged pair.
On 10 MB of TinyStories with a 2048-token vocabulary, the naive training takes about 40 seconds (the incremental one 3 seconds); with larger corpora and vocabularies it would take hours.

The incremental version keeps two data structures across merges:

- `pair_counts`: the count of every pair (computed once at the beginning);
- `pair_to_words`: for each pair, the set of (indices of) words that contain it.

When merging the best pair, only loop over `pair_to_words[best]`: for each such word, *subtract* the counts of its old pairs, merge, *add* the counts of its new pairs (and register the word for its new pairs).

!!! question "Exercise: `train_bpe`"
    Implement the incremental version in `train_bpe`. It must return exactly the same merges as `train_bpe_naive`, and be much faster: the test requires a 2× speed-up on a small corpus; you should get a lot more on TinyStories.
    Careful with pairs whose count drops to zero: delete them, otherwise `max` may pick them.

Validate with `uv run pytest tests/test_bpe_training.py`.

## Train a tokenizer on TinyStories

[TinyStories](https://arxiv.org/abs/2305.07759) is a dataset of short stories generated by GPT-4 with the vocabulary of a 3- to 4-year-old child, separated by `<|endoftext|>`.
The full training set is 2.2 GB; we only download its first 40 MB (about 50,000 stories) and 4 MB of the validation set:

```bash
uv run easygpt download tinystories
```

```text
Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. [...]
<|endoftext|>
```

Then train a 2048-token BPE on the first 10 MB and encode both files into arrays of token ids (next page):

```bash
uv run easygpt tokenize --vocab-size 2048
```

```text
training BPE (vocab 2048) on 10.0M characters...
  done in 2.9s
  longest token: b'<|endoftext|>'
  train: 10,840,924 tokens (3.69 bytes/token)
  valid: 1,084,034 tokens (3.69 bytes/token)
```

(The encoding step needs the `Tokenizer` of the [next page](tokenizer.md).)

## Questions

1. What are the longest tokens of your vocabulary (besides `<|endoftext|>`)? Do they make sense? (`sorted(tok.vocab.values(), key=len)[-10:]` after `tok = Tokenizer.load("data/tinystories/tokenizer.json")`.)
2. Profile the training (`python -m cProfile -s cumtime ...`): which part takes the most time, the pre-tokenization or the merges? In the naive and in the incremental version?
3. Look at the first 50 merges. What do they capture?
4. The `max` over all pairs at each merge is still linear in the number of distinct pairs. How could you make it faster? (Hint: a heap with lazy deletion.)
