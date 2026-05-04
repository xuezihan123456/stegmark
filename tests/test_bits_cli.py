from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from stegmark.cli import main


def test_embed_bits_and_extract_bits(
    sample_image_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "bits-output.png"
    runner = CliRunner()

    embed_result = runner.invoke(
        main,
        [
            "embed",
            str(sample_image_path),
            "--bits",
            "deadbeef",
            "-o",
            str(output),
            "-e",
            "native",
        ],
    )
    extract_result = runner.invoke(
        main,
        [
            "extract",
            str(output),
            "-e",
            "native",
            "--mode",
            "bits",
        ],
    )

    assert embed_result.exit_code == 0
    assert extract_result.exit_code == 0
    assert "deadbeef" in extract_result.output.lower()


def test_embed_requires_message_or_bits(sample_image_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "embed",
            str(sample_image_path),
            "-e",
            "native",
        ],
    )

    assert result.exit_code != 0
