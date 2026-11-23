"""Tests for docs/part2/bpe-training.md."""

import time

import pytest

from easygpt.bpe import (
    count_pairs,
    count_pretokens,
    merge_pair,
    pretokenize,
    split_on_special_tokens,
    train_bpe,
    train_bpe_naive,
)

EOT = "<|endoftext|>"
SENNRICH = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"


def B(*parts: str) -> tuple[bytes, ...]:
    return tuple(p.encode() for p in parts)


def test_pretokenize_is_given():
    assert pretokenize("some text that i'll pre-tokenize") == [
        "some", " text", " that", " i", "'ll", " pre", "-", "tokenize"
    ]  # fmt: skip


# -- special tokens ----------------------------------------------------------
def test_split_on_special_tokens():
    assert split_on_special_tokens(f"Hello{EOT}World", [EOT]) == ["Hello", EOT, "World"]
    assert split_on_special_tokens(f"{EOT}a{EOT}{EOT}", [EOT]) == [EOT, "a", EOT, EOT]
    assert split_on_special_tokens("no special here", [EOT]) == ["no special here"]
    assert split_on_special_tokens("abc", []) == ["abc"]
    assert split_on_special_tokens("", [EOT]) == []


def test_split_on_overlapping_special_tokens_prefers_the_longest():
    specials = [EOT, EOT + EOT]
    assert split_on_special_tokens(f"a{EOT}{EOT}b{EOT}", specials) == ["a", EOT + EOT, "b", EOT]


def test_split_escapes_regex_characters():
    assert split_on_special_tokens("a|b.c", ["|", "."]) == ["a", "|", "b", ".", "c"]


# -- counting ----------------------------------------------------------------
def test_count_pretokens_small():
    assert count_pretokens("hi hi") == {B("h", "i"): 1, B(" ", "h", "i"): 1}


def test_count_pretokens_sennrich():
    counts = count_pretokens(SENNRICH)
    assert counts[B(" ", "l", "o", "w")] == 4
    assert counts[B("l", "o", "w")] == 1
    assert counts[B(" ", "n", "e", "w", "e", "s", "t")] == 6
    assert sum(counts.values()) == 16


def test_count_pretokens_uses_utf8_bytes():
    counts = count_pretokens("été")
    assert counts == {(b"\xc3", b"\xa9", b"t", b"\xc3", b"\xa9"): 1}


def test_count_pretokens_drops_special_tokens_and_splits_on_them():
    counts = count_pretokens(f"ab{EOT}ab", [EOT])
    assert counts == {B("a", "b"): 2}  # "ab" twice, and no pre-token spans the boundary


def test_count_pairs():
    assert count_pairs({B("a", "b", "a", "b"): 2, B("b", "a"): 1}) == {B("a", "b"): 4, B("b", "a"): 3}


@pytest.mark.parametrize(
    "word,pair,expected",
    [
        (B("a", "b", "c"), B("a", "b"), B("ab", "c")),
        (B("a", "b", "a", "b"), B("a", "b"), B("ab", "ab")),
        (B("a", "a", "a"), B("a", "a"), B("aa", "a")),
        (B("a", "a", "a", "a"), B("a", "a"), B("aa", "aa")),
        (B("x", "y"), B("a", "b"), B("x", "y")),
        (B("e", "st", "s", "t"), B("e", "st"), B("est", "s", "t")),
    ],
)
def test_merge_pair(word, pair, expected):
    assert merge_pair(word, pair) == expected


# -- training ----------------------------------------------------------------
# Pre-tokens: "low" x1, " low" x4, " lower" x2, " widest" x3, " newest" x6.
# Ties are broken by the greatest pair, e.g. (s, t) > (e, s) for the first merge.
SENNRICH_MERGES = [
    (b"s", b"t"),
    (b"e", b"st"),
    (b"o", b"w"),
    (b"l", b"ow"),
    (b"w", b"est"),
    (b"n", b"e"),
    (b"ne", b"west"),
    (b" ", b"newest"),
    (b" ", b"low"),
    (b"w", b"i"),
]


TRAINERS = pytest.mark.parametrize("trainer", [train_bpe_naive, train_bpe], ids=["naive", "fast"])


@TRAINERS
def test_train_bpe_sennrich_example(trainer):
    vocab, merges = trainer(SENNRICH, vocab_size=256 + len(SENNRICH_MERGES))
    assert merges == SENNRICH_MERGES
    assert len(vocab) == 256 + len(SENNRICH_MERGES)


@TRAINERS
def test_train_bpe_vocabulary_layout(trainer, shakespeare_sample):
    vocab, merges = trainer(shakespeare_sample, vocab_size=300, special_tokens=[EOT])
    assert len(vocab) == 300
    assert len(merges) == 300 - 256 - 1
    assert all(vocab[i] == bytes([i]) for i in range(256))
    for i, (a, b) in enumerate(merges):
        assert vocab[256 + i] == a + b
    assert vocab[299] == EOT.encode()


@TRAINERS
def test_train_bpe_breaks_ties_with_the_greatest_pair(trainer):
    # (a,b) and (c,d) both occur twice: (c,d) is lexicographically greater
    _, merges = trainer("ab cd ab cd", vocab_size=257)
    assert merges == [(b"c", b"d")]


@TRAINERS
def test_train_bpe_never_merges_across_special_tokens(trainer, shakespeare_sample):
    paragraphs = shakespeare_sample.split("\n\n")
    text = EOT.join(paragraphs)
    vocab, _ = trainer(text, vocab_size=400, special_tokens=[EOT])
    for token in vocab.values():
        if token != EOT.encode():
            assert b"<|" not in token and b"|>" not in token


@TRAINERS
def test_train_bpe_stops_when_nothing_is_left_to_merge(trainer):
    vocab, merges = trainer("aaaa", vocab_size=1000)
    assert merges == [(b"a", b"a"), (b"aa", b"aa")]
    assert len(vocab) == 258


def test_fast_and_naive_training_agree(shakespeare_sample):
    _, merges_naive = train_bpe_naive(shakespeare_sample, vocab_size=400, special_tokens=[EOT])
    _, merges_fast = train_bpe(shakespeare_sample, vocab_size=400, special_tokens=[EOT])
    assert merges_fast == merges_naive


def test_train_bpe_is_faster_than_naive(shakespeare_sample):
    text = shakespeare_sample * 2
    start = time.perf_counter()
    train_bpe_naive(text, vocab_size=700)
    naive = time.perf_counter() - start
    start = time.perf_counter()
    train_bpe(text, vocab_size=700)
    fast = time.perf_counter() - start
    assert fast < naive / 2, f"train_bpe took {fast:.2f}s, train_bpe_naive {naive:.2f}s"
