from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from stegmark.cli import main


def test_config_show_outputs_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STEGMARK_CONFIG", str(tmp_path / "config.toml"))
    runner = CliRunner()

    set_result = runner.invoke(main, ["config", "set", "engine", "native"])
    show_result = runner.invoke(main, ["config", "show", "--json"])

    assert set_result.exit_code == 0
    assert show_result.exit_code == 0
    assert '"engine": "native"' in show_result.output


def test_config_reset_removes_file(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("STEGMARK_CONFIG", str(config_path))
    runner = CliRunner()

    runner.invoke(main, ["config", "set", "engine", "native"])
    reset_result = runner.invoke(main, ["config", "reset", "--yes"])

    assert reset_result.exit_code == 0
    assert not config_path.exists()


def test_config_set_supports_nested_hidden_keys(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    hidden_model_dir = tmp_path / 'models "hidden"'
    monkeypatch.setenv("STEGMARK_CONFIG", str(config_path))
    runner = CliRunner()

    set_model_dir = runner.invoke(main, ["config", "set", "hidden.model_dir", str(hidden_model_dir)])
    set_providers = runner.invoke(main, ["config", "set", "hidden.providers", "CUDAExecutionProvider,CPUExecutionProvider"])
    show_result = runner.invoke(main, ["config", "show", "--json"])

    payload = json.loads(show_result.output)

    assert set_model_dir.exit_code == 0
    assert set_providers.exit_code == 0
    assert payload["hidden"]["model_dir"] == str(hidden_model_dir)
    assert payload["hidden"]["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert payload["hidden_model_dir"] == str(hidden_model_dir)
