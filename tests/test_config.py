from __future__ import annotations

import json
from pathlib import Path

from stegmark.config import (
    default_config_path,
    load_config,
    reset_config_file,
    save_config_value,
)


def test_save_and_load_config_value(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    save_config_value(config_path, "engine", "native")
    config = load_config(config_path=config_path)

    assert config.engine == "native"


def test_env_overrides_file_value(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    save_config_value(config_path, "engine", "native")
    monkeypatch.setenv("STEGMARK_ENGINE", "hidden")

    config = load_config(config_path=config_path)

    assert config.engine == "hidden"


def test_reset_config_file_removes_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    save_config_value(config_path, "engine", "native")

    reset_config_file(config_path)

    assert not config_path.exists()


def test_default_config_path_honors_env_override(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "custom.toml"
    monkeypatch.setenv("STEGMARK_CONFIG", str(config_path))

    assert default_config_path() == config_path


def test_load_config_supports_nested_hidden_table(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    hidden_model_dir = tmp_path / 'models "hidden"'
    config_path.write_text(
        "\n".join(
            [
                'engine = "hidden"',
                "[hidden]",
                f"model_dir = {json.dumps(str(hidden_model_dir))}",
                'providers = ["CUDAExecutionProvider"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path=config_path)

    assert config.engine == "hidden"
    assert config.hidden.model_dir == hidden_model_dir
    assert config.hidden_model_dir == hidden_model_dir
    assert config.hidden.providers == ("CUDAExecutionProvider",)


def test_save_config_value_writes_nested_hidden_config_safely(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    hidden_model_dir = tmp_path / 'quoted "hidden" dir'

    save_config_value(config_path, "hidden.model_dir", str(hidden_model_dir))
    config = load_config(config_path=config_path)

    assert config.hidden.model_dir == hidden_model_dir
    assert config.hidden_model_dir == hidden_model_dir


def test_load_config_supports_legacy_hidden_model_dir_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    hidden_model_dir = tmp_path / "legacy-hidden"
    config_path.write_text(
        f"hidden_model_dir = {json.dumps(str(hidden_model_dir))}\n",
        encoding="utf-8",
    )

    config = load_config(config_path=config_path)

    assert config.hidden_model_dir == hidden_model_dir
