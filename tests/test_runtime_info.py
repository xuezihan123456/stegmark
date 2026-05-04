from __future__ import annotations

import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import stegmark


def test_available_engines_lists_native() -> None:
    engines = stegmark.available_engines()

    assert "native" in engines
    assert engines["native"] is True


def test_is_available_accepts_known_engines() -> None:
    assert stegmark.is_available("native") is True
    assert isinstance(stegmark.is_available("hidden"), bool)


def test_device_info_returns_dict() -> None:
    info = stegmark.device_info()

    assert isinstance(info, dict)
    assert "python" in info
    assert "engines" in info


def test_device_info_reports_hidden_provider_strategy(
    tmp_path: Path, monkeypatch
) -> None:
    fake_ort = ModuleType("onnxruntime")
    fake_ort.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", loader=None)
    fake_ort.get_available_providers = lambda: [  # type: ignore[attr-defined]
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        "stegmark.core.weights.load_config",
        lambda: SimpleNamespace(
            model_dir=tmp_path,
            hidden_model_dir=tmp_path / "hidden",
            hidden_providers=("CUDAExecutionProvider",),
        ),
    )

    info = stegmark.device_info()

    assert info["hidden_model_dir"] == str(tmp_path / "hidden")
    assert info["hidden"] == {
        "model_dir": str(tmp_path / "hidden"),
        "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "preferred_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }


def test_registered_engines_lists_builtin_backends() -> None:
    engines = stegmark.registered_engines()

    assert "native" in engines
    assert "hidden" in engines
    assert "trustmark" in engines
