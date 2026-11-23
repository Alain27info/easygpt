"""Tests for docs/part2/tokenizer.md: the BPE Tokenizer."""

import pytest

from easygpt.bpe import Tokenizer, merge_pair, pretokenize, train_bpe

EOT = "<|endoftext|>"


def _cat_tokenizer(special_tokens=None):
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    return Tokenizer(vocab, merges, special_tokens)


def test_handout_example():
    tok = _cat_tokenizer()
    assert tok.encode_pretoken("the") == [9]
    assert tok.encode_pretoken(" cat") == [7, 1, 5]
    assert tok.encode("the cat ate") == [9, 7, 1, 5, 10, 3]
    assert tok.decode([9, 7, 1, 5, 10, 3]) == "the cat ate"


def test_lookup_tables():
    tok = _cat_tokenizer()
    assert tok.token_to_id[b"the"] == 9
    assert tok.merge_ranks[(b" ", b"a")] == 2


def test_special_tokens_are_added_to_the_vocabulary():
    tok = _cat_tokenizer([EOT])
    assert tok.vocab_size == 12
    assert tok.vocab[11] == EOT.encode()
    assert tok.encode(f"the{EOT} cat") == [9, 11, 7, 1, 5]
    assert tok.decode([9, 11]) == f"the{EOT}"


def test_existing_special_token_is_not_duplicated():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = EOT.encode()
    tok = Tokenizer(vocab, [], [EOT])
    assert tok.vocab_size == 257
    assert tok.encode(EOT) == [256]


def test_overlapping_special_tokens():
    vocab = {i: bytes([i]) for i in range(256)}
    tok = Tokenizer(vocab, [], [EOT, EOT + EOT])
    ids = tok.encode(f"a{EOT}{EOT}b{EOT}")
    assert ids == [97, tok.token_to_id[(EOT + EOT).encode()], 98, tok.token_to_id[EOT.encode()]]


def test_decode_replaces_invalid_utf8():
    vocab = {i: bytes([i]) for i in range(256)}
    tok = Tokenizer(vocab, [])
    assert tok.decode([0xC3]) == "�"
    assert tok.decode([0xC3, 0xA9]) == "é"


@pytest.fixture(scope="module")
def trained(request):
    from pathlib import Path

    text = (Path(__file__).parent / "fixtures" / "shakespeare_sample.txt").read_text(encoding="utf-8")
    vocab, merges = train_bpe(text, vocab_size=500, special_tokens=[EOT])
    return Tokenizer(vocab, merges, [EOT]), text, merges


def _sequential_reference(pretoken: str, merges) -> list[bytes]:
    """Apply every merge, in order of creation, to the whole pre-token (training-time view)."""
    word = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
    for pair in merges:
        word = merge_pair(word, pair)
    return list(word)


def test_encode_pretoken_matches_training_segmentation(trained):
    tok, text, merges = trained
    for pretoken in sorted(set(pretokenize(text)))[:300]:
        ids = tok.encode_pretoken(pretoken)
        assert [tok.vocab[i] for i in ids] == _sequential_reference(pretoken, merges), pretoken


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, world!",
        "ROMEO:\nBut, soft! what light through yonder window breaks?\n\n",
        "Ünïcödé, 中文 and emojis 🙂🙃 are fine too.",
        f"one{EOT}two{EOT}",
        "   lots   of   spaces   \t\n\n\n",
    ],
)
def test_roundtrip(trained, text):
    tok, _, _ = trained
    ids = tok.encode(text)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == text


def test_compression(trained):
    tok, text, _ = trained
    ids = tok.encode(text)
    ratio = len(text.encode("utf-8")) / len(ids)
    assert ratio > 2.0, f"only {ratio:.2f} bytes per token"


def test_encode_iterable(trained):
    tok, text, _ = trained
    lines = text.splitlines(keepends=True)[:200]
    assert list(tok.encode_iterable(lines)) == [i for line in lines for i in tok.encode(line)]


def test_save_and_load(trained, tmp_path):
    tok, text, _ = trained
    tok.save(tmp_path / "tok.json")
    tok2 = Tokenizer.load(tmp_path / "tok.json")
    assert tok2.vocab == tok.vocab and tok2.merges == tok.merges
    assert tok2.encode(text[:2000]) == tok.encode(text[:2000])
