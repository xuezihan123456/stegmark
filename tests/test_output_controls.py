from __future__ import annotations

from pathlib import Path

from PIL import Image

from stegmark.service import embed_file


def test_embed_file_can_force_jpeg_output(
    sample_image_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output.jpg"

    result = embed_file(
        sample_image_path,
        output,
        message="Alice 2026",
        engine="native",
        output_format="jpeg",
    )

    assert result.output_path == output
    assert output.exists()
    assert Image.open(output).format == "JPEG"


def test_embed_file_respects_overwrite_false(
    sample_image_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output.png"
    output.write_bytes(sample_image_path.read_bytes())

    try:
        embed_file(
            sample_image_path,
            output,
            message="Alice 2026",
            engine="native",
            overwrite=False,
        )
    except FileExistsError:
        assert True
    else:
        raise AssertionError("expected FileExistsError")
