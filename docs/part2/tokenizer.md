# 10. Encoding and decoding

**Objective.** Use the learned merges to encode any text into token ids, and decode ids back into text.

## Encoding

Encoding mirrors training:

1. split the text on special tokens; each special token is mapped to its own id;
2. pre-tokenize the other chunks with the same regular expression;
3. encode each pre-token **independently**: start from its UTF-8 bytes and apply the merges.

How do we "apply the merges"? Exactly as during training, in the same order: the first merge learned is applied first.
An efficient way to do it: give each merge its **rank** (its position in the list of merges); then, as long as possible, find the adjacent pair of the current pre-token with the **lowest rank** and merge it (all its occurrences, left to right).
One can show that this gives the same result as applying all the merges one after the other in order of creation (the test `test_encode_pretoken_matches_training_segmentation` checks this).

### Example

Vocabulary `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`, merges (in order) `[(t, h), (␣, c), (␣, a), (th, e), (␣a, t)]`, text `"the cat ate"`.

- Pre-tokens: `"the"`, `" cat"`, `" ate"`.
- `"the"` = `(t, h, e)`. Pairs `(t, h)` (rank 0) and `(h, e)` (no rank): merge `(t, h)` → `(th, e)`; then `(th, e)` (rank 3) → `(the)`. Ids `[9]`.
- `" cat"` = `(␣, c, a, t)`: `(␣, c)` has rank 1, the others none → `(␣c, a, t)`. No pair is left to merge. Ids `[7, 1, 5]`.
- `" ate"` = `(␣, a, t, e)`: `(␣, a)` rank 2 → `(␣a, t, e)`; `(␣a, t)` rank 4 → `(␣at, e)`. Ids `[10, 3]`.

Result: `[9, 7, 1, 5, 10, 3]`.

**Caching.** Encoding a large corpus pre-token by pre-token is slow in Python, but the same pre-tokens (`" the"`, `" was"`, ...) occur over and over: keep a dictionary `pre-token → ids` (`self._cache`, given) and encode each distinct pre-token only once.

## Decoding

Decoding is easy: concatenate the bytes of the tokens and decode them as UTF-8.
One subtlety: an arbitrary sequence of ids (e.g. sampled by a model) is not always valid UTF-8 (think of a multi-byte character cut in the middle).
Use `bytes.decode("utf-8", errors="replace")`, which replaces invalid bytes with the replacement character `U+FFFD` (�).

!!! question "Exercise: `Tokenizer`"
    In [`src/easygpt/bpe.py`](https://github.com/svaiter/easygpt/blob/main/src/easygpt/bpe.py), the constructor of `Tokenizer` is given: it builds `self.vocab` (id → bytes, with the special tokens appended if missing), `self.token_to_id` (bytes → id), `self.merge_ranks` (pair → rank) and an empty `self._cache`. Implement:

    - `encode_pretoken(pretoken)`: the merge loop above for a single pre-token (a `str`), returning a list of ids;
    - `encode(text)`: split on special tokens, pre-tokenize, encode each pre-token (through the cache);
    - `decode(ids)`.

    `encode_iterable` (encode a file lazily, line by line) and the serialization methods `save`/`load` are given.

Validate with `uv run pytest tests/test_tokenizer.py`.

Now you can run (or re-run) the tokenization of TinyStories:

```bash
uv run easygpt tokenize --vocab-size 2048
```

It trains the BPE, saves it to `data/tinystories/tokenizer.json`, and encodes the training and validation texts into `data/tinystories/train.npy` and `valid.npy`: NumPy arrays of `uint16`, read back with `np.load(..., mmap_mode="r")`: they are memory-mapped, so they are read from disk on demand instead of being loaded into memory (this is why `get_batch` only converts the windows it samples).

```python
>>> from easygpt.bpe import Tokenizer
>>> tok = Tokenizer.load("data/tinystories/tokenizer.json")
>>> ids = tok.encode("Once upon a time, there was a little dog named Max.")
>>> [tok.decode([i]) for i in ids]
['Once', ' upon', ' a', ' time', ',', ' there', ' was', ' a', ' little', ' dog', ' named', ' Max', '.']
```

!!! tip "See it"
    [Tiktokenizer](https://tiktokenizer.vercel.app/) colors, token by token, how the tokenizers of GPT-4o and other models split the text you type.

## Questions

1. What is the **compression ratio** (bytes per token) of your tokenizer on the TinyStories validation set? On Tiny Shakespeare? On a page of Wikipedia? Explain the differences.
2. Encode `"The Higgs boson was discovered at CERN in 2012."`. How are the rare words split?
3. Estimate the throughput of your tokenizer (MB/s). How long would it take to encode the 2.2 GB of the full TinyStories? The 825 GB of the Pile? What could you do to go faster?
4. Why is `uint16` an appropriate type to store the token ids? Up to which vocabulary size?
5. Encoding a file line by line with `encode_iterable` can give a slightly different result from encoding it in one go. Find an example. (Hint: whitespace at the end of a line.) Why is cutting the text right after `<|endoftext|>`, as `easygpt tokenize` does, safe?
