from __future__ import annotations

from importlib import import_module


def test_package_imports() -> None:
    module = import_module("stegmark")
    assert module.__version__
