"""Tests for docs/part1/data.md: CharTokenizer and get_batch."""

import numpy as np
import pytest
import torch

from easygpt.data import CharTokenizer, get_batch

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n"


def test_vocabulary_is_sorted_distinct_characters():
    tok = CharTokenizer(TEXT)
    assert tok.chars == sorted(set(TEXT))
    assert tok.vocab_size == len(set(TEXT))


def test_encode_small_example():
    tok = CharTokenizer("abcabc")
    assert tok.encode("cab") == [2, 0, 1]
    assert tok.encode("") == []


def test_encode_returns_python_ints():
    tok = CharTokenizer(TEXT)
    ids = tok.encode(TEXT)
    assert isinstance(ids, list)
    assert all(type(i) is int for i in ids)
    assert len(ids) == len(TEXT)
    assert min(ids) == 0 and max(ids) == tok.vocab_size - 1


def test_decode_inverts_encode(shakespeare_sample):
    tok = CharTokenizer(shakespeare_sample)
    assert tok.decode(tok.encode(shakespeare_sample)) == shakespeare_sample
    assert tok.decode(tok.encode("ROMEO:")) == "ROMEO:"


def test_unknown_character_raises():
    tok = CharTokenizer("abc")
    with pytest.raises(KeyError):
        tok.encode("abd")


def test_get_batch_shapes_and_dtype():
    data = np.arange(1000)
    x, y = get_batch(data, batch_size=8, context_length=16)
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert x.dtype == torch.long and y.dtype == torch.long
    assert x.device.type == "cpu"


def test_get_batch_targets_are_shifted_inputs():
    data = np.arange(1000)
    x, y = get_batch(data, batch_size=16, context_length=10)
    # rows are contiguous windows of data...
    assert torch.equal(x[:, 1:], x[:, :-1] + 1)
    # ... and targets are the inputs shifted by one position
    assert torch.equal(y, x + 1)


def test_get_batch_covers_every_valid_start():
    data = np.arange(10)
    context_length = 3
    starts = set()
    for _ in range(300):
        x, y = get_batch(data, batch_size=4, context_length=context_length)
        starts.update(x[:, 0].tolist())
        assert int(y.max()) <= 9  # never reads past the end
    # s + context_length + 1 <= len(data)  =>  s in {0, ..., 6}
    assert starts == set(range(len(data) - context_length))


def test_get_batch_is_reproducible_with_a_generator():
    data = np.arange(500)
    x1, _ = get_batch(data, 4, 8, generator=torch.Generator().manual_seed(42))
    x2, _ = get_batch(data, 4, 8, generator=torch.Generator().manual_seed(42))
    assert torch.equal(x1, x2)


def test_get_batch_on_a_uint16_memmap(tmp_path):
    ids = (np.arange(5000) % 2048).astype(np.uint16)
    np.save(tmp_path / "ids.npy", ids)
    data = np.load(tmp_path / "ids.npy", mmap_mode="r")
    x, y = get_batch(data, batch_size=4, context_length=32)
    assert x.dtype == torch.long
    assert torch.equal(y[:, :-1], x[:, 1:])
