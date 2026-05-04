from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from stegmark.cli import main


def test_embed_command_supports_overwrite(
    sample_image_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output.png"
    output.write_bytes(sample_image_path.read_bytes())

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
            "--overwrite",
        ],
    )

    assert result.exit_code == 0


def test_embed_command_supports_format_and_quality(
    sample_image_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output.jpg"

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
            "--format",
            "jpeg",
            "--quality",
            "90",
        ],
    )

    assert result.exit_code == 0
    assert Image.open(output).format == "JPEG"
