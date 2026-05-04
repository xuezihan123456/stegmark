from __future__ import annotations

from pathlib import Path

import tomllib
from click.testing import CliRunner

from stegmark.cli import main


def test_cli_help_lists_core_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "embed" in result.output
    assert "extract" in result.output
    assert "verify" in result.output
    assert "info" in result.output


def test_pyproject_declares_console_script() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["project"]["scripts"]["stegmark"] == "stegmark.cli:main"
