from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from stegmark.config import load_config, save_config_value
from stegmark.core.hidden import HiddenEngine


def test_config_round_trips_nested_hidden_model_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    hidden_dir = tmp_path / "models" / "hidden-custom"

    save_config_value(config_path, "engines.hidden.model_dir", str(hidden_dir))
    config = load_config(config_path=config_path)

    assert config.hidden_model_dir == hidden_dir


def test_config_writer_escapes_windows_like_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    raw = r"C:\Users\A B\hidden\models"

    save_config_value(config_path, "engines.hidden.model_dir", raw)
    config = load_config(config_path=config_path)

    assert str(config.hidden_model_dir) == raw


def test_hidden_engine_initializes_session_only_once_under_concurrency(monkeypatch) -> None:
    pytest.importorskip("onnxruntime")
    engine = HiddenEngine(model_dir=Path("models"))
    calls: list[Path] = []

    def fake_create_session(path: Path) -> object:
        calls.append(path)
        return object()

    monkeypatch.setattr(engine, "_create_session", fake_create_session)
    monkeypatch.setattr(engine, "_ensure_available", lambda: None)

    with ThreadPoolExecutor(max_workers=8) as executor:
        sessions = list(executor.map(lambda _: engine._encoder, range(16)))

    assert len({id(item) for item in sessions}) == 1
    assert calls == [engine.encoder_path]


def test_hidden_engine_prefers_configured_providers(tmp_path: Path, monkeypatch) -> None:
    providers_seen: list[list[str]] = []

    class FakeOrt:
        @staticmethod
        def InferenceSession(path: str, *, providers: list[str]) -> object:
            providers_seen.append(providers)
            return object()

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)
    config_path = tmp_path / "config.toml"
    save_config_value(config_path, "engines.hidden.providers", "CUDAExecutionProvider,CPUExecutionProvider")
    monkeypatch.setenv("STEGMARK_CONFIG", str(config_path))

    engine = HiddenEngine(model_dir=tmp_path)
    engine._create_session(tmp_path / "encoder.onnx")

    assert providers_seen == [["CUDAExecutionProvider", "CPUExecutionProvider"]]
