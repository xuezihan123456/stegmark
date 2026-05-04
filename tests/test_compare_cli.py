from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from stegmark.cli import main


def test_embed_compare_outputs_report(
    sample_image_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output.png"

    result = CliRunner().invoke(
        main,
        [
            "embed",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "-o",
            str(output),
            "-e",
            "native",
            "--compare",
        ],
    )

    assert result.exit_code == 0
    assert "compare" in result.output.lower()
