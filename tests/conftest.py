"""Shared pytest configuration.

By default the tests run against your code in ``src/easygpt``. Instructors can
run them against the reference solutions with ``pytest --solutions``.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption(
        "--solutions",
        action="store_true",
        default=False,
        help="run the tests against solutions/easygpt instead of src/easygpt",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "solutions_only: only meaningful with --solutions")
    if config.getoption("--solutions"):
        solutions = ROOT / "solutions"
        if not solutions.exists():
            raise pytest.UsageError("--solutions given but there is no solutions/ directory")
        sys.path.insert(0, str(solutions))
        for name in list(sys.modules):
            if name == "easygpt" or name.startswith("easygpt."):
                del sys.modules[name]


def pytest_collection_modifyitems(config, items):
    if config.getoption("--solutions"):
        return
    skip = pytest.mark.skip(reason="instructor check, run with --solutions")
    for item in items:
        if "solutions_only" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def shakespeare_sample() -> str:
    return (FIXTURES / "shakespeare_sample.txt").read_text(encoding="utf-8")
