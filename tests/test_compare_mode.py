from __future__ import annotations

from pathlib import Path

from stegmark.service import embed_file


def test_embed_file_compare_generates_report(
    sample_image_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output.png"

    result = embed_file(
        sample_image_path,
        output,
        message="Alice 2026",
        engine="native",
        compare=True,
    )

    assert result.psnr is not None
    assert result.compare_report is not None
    assert result.compare_report.exists()
