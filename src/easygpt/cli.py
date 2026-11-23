"""Command-line interface (given, not an exercise).

    easygpt download shakespeare|tinystories
    easygpt tokenize --vocab-size 2048          # TinyStories: BPE + encode to .npy
    easygpt train --preset shakespeare          # see `easygpt presets`
    easygpt generate checkpoints/shakespeare.pt --prompt "ROMEO:"

Docs: docs/part1/training.md, docs/part1/sampling.md, docs/part2/tinystories.md
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from easygpt import datasets
from easygpt.config import PRESETS, ModelConfig, get_preset

EOT = "<|endoftext|>"


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------
def load_shakespeare(data_dir: Path):
    from easygpt.data import CharTokenizer

    text = datasets.download_shakespeare(data_dir).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    n = int(0.9 * len(ids))
    return tokenizer, ids[:n], ids[n:]


def load_tinystories(data_dir: Path):
    from easygpt.bpe import Tokenizer

    folder = Path(data_dir) / "tinystories"
    tok_path = folder / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit("TinyStories is not tokenized yet: run `easygpt tokenize` first.")
    tokenizer = Tokenizer.load(tok_path)
    train = np.load(folder / "train.npy", mmap_mode="r")
    valid = np.load(folder / "valid.npy", mmap_mode="r")
    return tokenizer, train, valid


def encode_file(tokenizer, path: Path, chunk_chars: int = 1 << 20) -> np.ndarray:
    """Encode a large text file chunk by chunk, cutting only right after ``<|endoftext|>``."""
    ids: list[int] = []
    buffer = ""
    read, total = 0, path.stat().st_size
    start = time.time()
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            read += len(chunk)
            buffer += chunk
            cut = buffer.rfind(EOT) + len(EOT) if chunk else len(buffer)
            if cut >= len(EOT) or not chunk:  # a story boundary was found (or end of file)
                ids.extend(tokenizer.encode(buffer[:cut]))
                buffer = buffer[cut:]
            rate = read / 1e6 / max(time.time() - start, 1e-9)
            print(f"\r  encoding {path.name}: {min(100, 100 * read / total):5.1f}% ({rate:.2f} MB/s)", end="", flush=True)
            if not chunk:
                break
    print()
    return np.array(ids, dtype=np.uint16)


# -----------------------------------------------------------------------------
# commands
# -----------------------------------------------------------------------------
def cmd_download(args):
    if args.dataset == "shakespeare":
        print(datasets.download_shakespeare(args.data_dir))
    else:
        for p in datasets.download_tinystories(args.data_dir, max_mb=args.max_mb, valid_mb=args.valid_mb):
            print(p)


def cmd_tokenize(args):
    from easygpt.bpe import Tokenizer, train_bpe

    train_path, valid_path = datasets.download_tinystories(args.data_dir)
    folder = train_path.parent
    with open(train_path, encoding="utf-8") as f:
        sample = f.read(int(args.bpe_mb * 1e6))
    print(f"training BPE (vocab {args.vocab_size}) on {len(sample) / 1e6:.1f}M characters...")
    start = time.time()
    vocab, merges = train_bpe(sample, args.vocab_size, [EOT])
    print(f"  done in {time.time() - start:.1f}s")
    tokenizer = Tokenizer(vocab, merges, [EOT])
    tokenizer.save(folder / "tokenizer.json")
    longest = max(tokenizer.vocab.values(), key=len)
    print(f"  longest token: {longest!r}")
    for split, path in (("train", train_path), ("valid", valid_path)):
        ids = encode_file(tokenizer, path)
        np.save(folder / f"{split}.npy", ids)
        print(f"  {split}: {len(ids):,} tokens ({path.stat().st_size / len(ids):.2f} bytes/token)")


def cmd_presets(args):
    for name, preset in PRESETS.items():
        print(f"{name:18s} {preset.description}")
        print(f"{'':18s} model: {asdict(preset.model)}")
        print(f"{'':18s} train: {asdict(preset.train)}")


def build_model(model_config: ModelConfig):
    from easygpt.model import TransformerLM

    return TransformerLM(**asdict(model_config))


def cmd_train(args):
    from easygpt.model import count_parameters
    from easygpt.train import save_checkpoint, train

    preset = get_preset(
        args.preset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr_max=args.lr_max,
        device=args.device,
        seed=args.seed,
    )
    torch.manual_seed(preset.train.seed)
    if preset.dataset == "shakespeare":
        tokenizer, train_data, val_data = load_shakespeare(args.data_dir)
    else:
        tokenizer, train_data, val_data = load_tinystories(args.data_dir)
    preset.model.vocab_size = tokenizer.vocab_size
    if args.no_rope:
        preset.model.rope_theta = None
    model = build_model(preset.model)
    if args.compile:
        model = torch.compile(model)
    print(f"model: {asdict(preset.model)}")
    print(f"parameters: {count_parameters(model):,} ({count_parameters(model, exclude_embeddings=True):,} non-embedding)")
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")
    history = train(model, train_data, val_data, preset.train, preset.model.context_length)

    out = Path(args.out or f"checkpoints/{args.preset}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {"model_config": asdict(preset.model), "tokenizer": tokenizer.to_dict(), "history": history}
    model_to_save = getattr(model, "_orig_mod", model)  # undo torch.compile wrapper
    save_checkpoint(model_to_save, None, preset.train.max_steps, out, extra=extra)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"preset": args.preset, "model": asdict(preset.model), "train": asdict(preset.train), "history": history}, f)
    print(f"saved {out}")

    from easygpt.generate import generate

    prompt = "\n" if preset.dataset == "shakespeare" else "Once upon a time"
    ids = generate(model_to_save, tokenizer.encode(prompt), 300, preset.model.context_length, temperature=0.8)
    print("-" * 80)
    print(prompt + tokenizer.decode(ids))


def load_tokenizer(d: dict):
    if d["type"] == "char":
        from easygpt.data import CharTokenizer

        return CharTokenizer.from_dict(d)
    from easygpt.bpe import Tokenizer

    return Tokenizer.from_dict(d)


def cmd_generate(args):
    from easygpt.generate import generate
    from easygpt.train import load_checkpoint, read_checkpoint_extra

    extra = read_checkpoint_extra(args.checkpoint)
    model_config = ModelConfig(**extra["model_config"])
    model = build_model(model_config)
    load_checkpoint(args.checkpoint, model)
    model.to(args.device).eval()
    tokenizer = load_tokenizer(extra["tokenizer"])
    eos_id = None
    if getattr(tokenizer, "special_tokens", None):
        eos_id = tokenizer.token_to_id[EOT.encode("utf-8")]
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    for i in range(args.num_samples):
        ids = generate(
            model,
            tokenizer.encode(args.prompt),
            args.max_new_tokens,
            model_config.context_length,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_id=eos_id,
            generator=generator,
        )
        if i > 0:
            print("-" * 80)
        print(args.prompt + tokenizer.decode(ids))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="easygpt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=datasets.DATA_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download a dataset")
    p.add_argument("dataset", choices=["shakespeare", "tinystories"])
    p.add_argument("--max-mb", type=float, default=40, help="TinyStories: MB of training text")
    p.add_argument("--valid-mb", type=float, default=4, help="TinyStories: MB of validation text")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("tokenize", help="train a BPE tokenizer on TinyStories and encode it")
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--bpe-mb", type=float, default=10, help="MB of text used to train the BPE")
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("presets", help="list the training presets")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("train", help="train a Transformer LM")
    p.add_argument("--preset", default="shakespeare", choices=list(PRESETS))
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr-max", type=float)
    p.add_argument("--device", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-rope", action="store_true", help="ablation: no position embedding (NoPE)")
    p.add_argument("--compile", action="store_true", help="use torch.compile (faster, slow start)")
    p.add_argument("--out", type=str, help="checkpoint path (default checkpoints/PRESET.pt)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("generate", help="sample from a trained checkpoint")
    p.add_argument("checkpoint", type=str)
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.set_defaults(func=cmd_generate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
