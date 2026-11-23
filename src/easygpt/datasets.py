"""Download the datasets used in the course (given, not an exercise).

- Tiny Shakespeare (1.1 MB): the concatenation of Shakespeare plays used by
  Karpathy's char-rnn and nanoGPT.
- TinyStories (Eldan & Li, 2023), GPT-4 generated version: only the first
  ``max_mb`` megabytes of the training file are downloaded, cut at a story
  boundary, plus the beginning of the validation file.

Usage: ``easygpt download shakespeare`` or ``easygpt download tinystories``.
"""

import os
import urllib.request
from pathlib import Path

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"
EOT = "<|endoftext|>"

DATA_DIR = Path(os.environ.get("EASYGPT_DATA", "data"))


def _download(url: str, dest: Path, max_bytes: int | None = None) -> None:
    """Stream ``url`` to ``dest``, stopping after ``max_bytes`` bytes if given."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        while max_bytes is None or written < max_bytes:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            if max_bytes is not None:
                chunk = chunk[: max_bytes - written]
            f.write(chunk)
            written += len(chunk)
            print(f"\r  {dest.name}: {written / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def download_shakespeare(data_dir: Path = DATA_DIR) -> Path:
    path = Path(data_dir) / "shakespeare.txt"
    if not path.exists():
        _download(SHAKESPEARE_URL, path)
    return path


def _cut_at_last_eot(path: Path) -> None:
    """Truncate a partial TinyStories file right after its last complete story."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    end = text.rfind(EOT)
    if end != -1:
        text = text[: end + len(EOT)] + "\n"
    path.write_text(text, encoding="utf-8")


def download_tinystories(data_dir: Path = DATA_DIR, max_mb: float = 40, valid_mb: float = 4) -> tuple[Path, Path]:
    data_dir = Path(data_dir) / "tinystories"
    paths = []
    for split, mb in (("train", max_mb), ("valid", valid_mb)):
        path = data_dir / f"{split}.txt"
        if not path.exists():
            _download(TINYSTORIES_URL.format(split=split), path, max_bytes=int(mb * 1e6))
            _cut_at_last_eot(path)
        paths.append(path)
    return paths[0], paths[1]
