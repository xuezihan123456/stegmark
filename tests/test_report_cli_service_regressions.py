from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import stegmark
from stegmark.cli import main
from stegmark.core import image_io
from stegmark.exceptions import InvalidInputError
from stegmark.service import embed_directory, embed_file
from stegmark.types import BatchResult, EmbedResult, ImageMetadata


def _fake_embed_result() -> EmbedResult:
    import numpy as np

    return EmbedResult(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        engine="native",
        message="ok",
        bits=(1, 0, 1),
        metadata=ImageMetadata(format="PNG", mode="RGB"),
        output_path=Path("out.png"),
    )


def test_embed_cli_preserves_explicit_zero_strength(sample_image_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_embed(*args, **kwargs):
        captured.update(kwargs)
        return _fake_embed_result()

    monkeypatch.setattr(stegmark, "embed", fake_embed)

    result = CliRunner().invoke(
        main,
        ["embed", str(sample_image_path), "--message", "hello", "--strength", "0.0"],
    )

    assert result.exit_code == 0
    assert captured["strength"] == 0.0


def test_embed_directory_cli_preserves_explicit_zero_workers(sample_image_path: Path, tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / "a.png").write_bytes(sample_image_path.read_bytes())
    captured: dict[str, object] = {}

    def fake_embed_directory(*args, **kwargs):
        captured.update(kwargs)
        return BatchResult(items=())

    monkeypatch.setattr(stegmark, "embed_directory", fake_embed_directory)

    result = CliRunner().invoke(
        main,
        ["embed", str(source_dir), "--message", "hello", "--workers", "0"],
    )

    assert result.exit_code == 0
    assert captured["workers"] == 0


def test_embed_cli_allows_empty_message(sample_image_path: Path, monkeypatch) -> None:
    def fake_embed(*args, **kwargs):
        return _fake_embed_result()

    monkeypatch.setattr(stegmark, "embed", fake_embed)

    result = CliRunner().invoke(
        main,
        ["embed", str(sample_image_path), "--message", ""],
    )

    assert result.exit_code == 0


def test_embed_file_accepts_bits_with_hex_prefix(sample_image_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "prefixed.png"

    result = embed_file(sample_image_path, output, bits_hex="0xdeadbeef", engine="native")

    assert result.payload == bytes.fromhex("deadbeef")


def test_embed_directory_progress_callback_reports_all_items(sample_image_path: Path, tmp_path: Path) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    for name in ("a.png", "b.png"):
        (source_dir / name).write_bytes(sample_image_path.read_bytes())
    seen: list[tuple[int, int, str]] = []

    def progress(current: int, total: int, path: Path) -> None:
        seen.append((current, total, path.name))

    result = embed_directory(source_dir, message="hello", engine="native", progress=progress)

    assert result.succeeded == 2
    assert seen == [(1, 2, "a.png"), (2, 2, "b.png")]


def test_embed_directory_rejects_output_outside_allowed_root(sample_image_path: Path, tmp_path: Path) -> None:
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "safe"
    source_dir.mkdir()
    (source_dir / "a.png").write_bytes(sample_image_path.read_bytes())

    with pytest.raises(InvalidInputError):
        embed_directory(
            source_dir,
            message="hello",
            engine="native",
            output_dir=output_dir,
            allowed_output_root=tmp_path / "different-root",
        )


def test_load_image_rejects_oversized_file(sample_image_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(image_io, "MAX_FILE_SIZE_BYTES", 1)

    with pytest.raises(InvalidInputError):
        image_io.load_image(sample_image_path)


def test_load_image_rejects_excessive_pixels(sample_image_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(image_io, "MAX_IMAGE_PIXELS", 32)

    with pytest.raises(InvalidInputError):
        image_io.load_image(sample_image_path)
