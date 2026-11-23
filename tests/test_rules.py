"""The rules of the game: build things from scratch.

Your code in ``src/easygpt`` may use any tensor operation of PyTorch, autograd,
``nn.Module``, ``nn.Parameter``, ``nn.ModuleList``, ``torch.optim.Optimizer``
and ``nn.init``, but not the ready-made layers, losses and optimizers that you
are asked to implement, nor einsum-style helpers.
"""

import io
import re
import tokenize
from pathlib import Path

import pytest

import easygpt

PACKAGE_DIR = Path(easygpt.__file__).parent

FORBIDDEN = [
    (r"\bnn\.functional\b|(?<![\w.])F\.", "torch.nn.functional"),
    (
        r"\bnn\s*\.\s*(Linear|Bilinear|Embedding|EmbeddingBag|LayerNorm|RMSNorm|MultiheadAttention|"
        r"Transformer\w*|CrossEntropyLoss|NLLLoss|Softmax|LogSoftmax|SiLU|GELU|ReLU)\b",
        "ready-made torch.nn layers",
    ),
    # `softmax(` alone is a call to our own function; `torch.softmax(` / `x.softmax(` are not
    (r"\.softmax\(|\blog_softmax\b|\blogsumexp\b", "built-in (log-)softmax / logsumexp"),
    (r"\beinsum\b|\beinops\b|\beinx\b", "einsum-style helpers (use @, view, transpose, ...)"),
    (r"\boptim\s*\.\s*(SGD|Adam|AdamW|lr_scheduler)\b", "ready-made optimizers and schedulers"),
    (r"\bclip_grad_norm_\b|\bclip_grad_value_\b", "torch.nn.utils.clip_grad_norm_"),
]


def _code_without_strings_and_comments(path: Path) -> str:
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
    kept = [tok.string for tok in tokens if tok.type not in (tokenize.STRING, tokenize.COMMENT)]
    return " ".join(kept)


@pytest.mark.parametrize("path", sorted(PACKAGE_DIR.glob("*.py")), ids=lambda p: p.name)
def test_no_forbidden_shortcuts(path):
    code = _code_without_strings_and_comments(path)
    code = code.replace(" . ", ".").replace(" (", "(")
    for pattern, what in FORBIDDEN:
        match = re.search(pattern, code)
        if match:
            snippet = code[max(0, match.start() - 30) : match.end() + 30]
            pytest.fail(f"{path.name} uses {what}: ...{snippet}...")
