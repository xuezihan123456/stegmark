from __future__ import annotations

from pathlib import Path

import tomli as tomllib

import stegmark


def test_package_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["project"]["version"] == stegmark.__version__


def test_changelog_mentions_current_version() -> None:
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{stegmark.__version__}]" in changelog
    assert "## [Unreleased]" in changelog
