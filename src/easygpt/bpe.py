"""Byte-level byte-pair encoding (BPE): training and a Tokenizer.

Docs: docs/part2/bpe-training.md (split_on_special_tokens, count_pretokens,
count_pairs, merge_pair, train_bpe) and docs/part2/tokenizer.md (Tokenizer).
"""

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable

import regex

# GPT-2 pre-tokenization pattern (the version from tiktoken). Requires the
# third-party `regex` module for the \p{L} (letter) and \p{N} (number) classes.
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_GPT2_RE = regex.compile(GPT2_PATTERN)

Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into pre-tokens with the GPT-2 regular expression (given)."""
    return _GPT2_RE.findall(text)


def split_on_special_tokens(text: str, special_tokens: Iterable[str]) -> list[str]:
    """Split ``text`` around occurrences of special tokens.

    The special tokens themselves are kept as separate items of the output,
    and empty strings are dropped. When special tokens overlap, the longest
    one wins.

    >>> split_on_special_tokens("a<|eot|>b", ["<|eot|>"])
    ['a', '<|eot|>', 'b']
    """
    raise NotImplementedError("Exercise: split_on_special_tokens (docs/part2/bpe-training.md)")


def count_pretokens(text: str, special_tokens: Iterable[str] = ()) -> dict[Word, int]:
    """Count the pre-tokens of ``text``, each represented as a tuple of single bytes.

    Special tokens act as hard boundaries and are *not* counted.

    >>> count_pretokens("hi hi")
    {(b'h', b'i'): 1, (b' ', b'h', b'i'): 1}
    """
    special_tokens = list(special_tokens)
    raise NotImplementedError("Exercise: count_pretokens (docs/part2/bpe-training.md)")


def count_pairs(word_counts: dict[Word, int]) -> dict[Pair, int]:
    """Count adjacent pairs of tokens, weighted by the count of each word.

    >>> count_pairs({(b'a', b'b', b'a', b'b'): 2})
    {(b'a', b'b'): 4, (b'b', b'a'): 2}
    """
    raise NotImplementedError("Exercise: count_pairs (docs/part2/bpe-training.md)")


def merge_pair(word: Word, pair: Pair) -> Word:
    """Replace every occurrence of ``pair`` in ``word`` (scanning left to right) by its concatenation.

    >>> merge_pair((b'a', b'a', b'a'), (b'a', b'a'))
    (b'aa', b'a')
    """
    raise NotImplementedError("Exercise: merge_pair (docs/part2/bpe-training.md)")


def train_bpe(
    text: str, vocab_size: int, special_tokens: Iterable[str] = ()
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a byte-level BPE tokenizer on ``text``.

    Returns:
        ``(vocab, merges)`` where ``vocab`` maps ids to tokens (as bytes):
        ids ``0..255`` are the single bytes (``vocab[i] == bytes([i])``), then
        one id per merge in order of creation, then the special tokens; and
        ``merges`` is the ordered list of merged pairs. ``len(vocab) ==
        vocab_size`` unless the text runs out of pairs to merge. Ties between
        equally frequent pairs are broken by taking the lexicographically
        greatest pair.
    """
    special_tokens = list(special_tokens)
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size is too small for 256 bytes + special tokens")
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[Pair] = []
    word_counts = count_pretokens(text, special_tokens)
    raise NotImplementedError("Exercise: train_bpe (docs/part2/bpe-training.md)")
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    return vocab, merges


def train_bpe_naive(
    text: str, vocab_size: int, special_tokens: Iterable[str] = ()
) -> tuple[dict[int, bytes], list[Pair]]:
    """Straightforward (slow) BPE training, recounting all pairs after each merge.

    Same contract as :func:`train_bpe`.
    """
    special_tokens = list(special_tokens)
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size is too small for 256 bytes + special tokens")
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[Pair] = []
    word_counts = count_pretokens(text, special_tokens)
    raise NotImplementedError("Exercise: train_bpe_naive (docs/part2/bpe-training.md)")
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    return vocab, merges


class Tokenizer:
    """Encode text into BPE token ids and decode them back.

    Attributes (built in ``__init__``):
        vocab: ``dict[int, bytes]``, id -> token.
        merges: ``list[tuple[bytes, bytes]]``, the merges in order of creation.
        special_tokens: list of special tokens (strings); missing ones are
            appended to the vocabulary.
        token_to_id: ``dict[bytes, int]``, the inverse of ``vocab``.
        merge_ranks: ``dict[tuple[bytes, bytes], int]``, position of each pair
            in ``merges`` (lower rank = merged earlier = higher priority).
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        for tok in self.special_tokens:
            if tok.encode("utf-8") not in self.vocab.values():
                self.vocab[max(self.vocab) + 1] = tok.encode("utf-8")
        self.token_to_id = {token: i for i, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._cache: dict[str, list[int]] = {}  # pre-token -> ids

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_pretoken(self, pretoken: str) -> list[int]:
        """Encode a single pre-token (no special token inside).

        Start from its UTF-8 bytes and repeatedly merge the adjacent pair with
        the lowest rank in ``self.merge_ranks`` until no pair can be merged.
        """
        raise NotImplementedError("Exercise: Tokenizer.encode_pretoken (docs/part2/tokenizer.md)")

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token ids.

        Split on special tokens (mapped to their own id), pre-tokenize the
        other chunks and encode each pre-token with :meth:`encode_pretoken`
        (use ``self._cache`` to avoid re-encoding frequent pre-tokens).
        """
        raise NotImplementedError("Exercise: Tokenizer.encode (docs/part2/tokenizer.md)")

    def decode(self, ids: list[int]) -> str:
        """Concatenate the bytes of the tokens and decode them as UTF-8.

        Invalid byte sequences are replaced by U+FFFD (``errors="replace"``).
        """
        raise NotImplementedError("Exercise: Tokenizer.decode (docs/part2/tokenizer.md)")

    def encode_iterable(self, iterable: Iterable[str]):
        """Lazily encode an iterable of strings (e.g. a file, line by line) (given)."""
        for text in iterable:
            yield from self.encode(text)

    # -- serialization (given) ----------------------------------------------
    def to_dict(self) -> dict:
        return {
            "type": "bpe",
            "vocab": {str(i): token.hex() for i, token in self.vocab.items()},
            "merges": [[a.hex(), b.hex()] for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tokenizer":
        vocab = {int(i): bytes.fromhex(h) for i, h in d["vocab"].items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in d["merges"]]
        return cls(vocab, merges, d.get("special_tokens"))

    def save(self, path: str | os.PathLike) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
