from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from stegmark.cli import main


def test_embed_directory_command(sample_image_path: Path, tmp_path: Path) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    for name in ("a.png", "b.png"):
        (source_dir / name).write_bytes(sample_image_path.read_bytes())

    result = CliRunner().invoke(
        main,
        [
            "embed",
            str(source_dir),
            "-m",
            "Alice 2026",
            "-e",
            "native",
            "-w",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert (source_dir / "a_wm.png").exists()
    assert (source_dir / "b_wm.png").exists()


def test_embed_directory_can_target_output_dir(
    sample_image_path: Path, tmp_path: Path
) -> None:
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "protected"
    nested = source_dir / "nested"
    nested.mkdir(parents=True)
    for path in (source_dir / "a.png", nested / "b.png"):
        path.write_bytes(sample_image_path.read_bytes())

    result = CliRunner().invoke(
        main,
        [
            "embed",
            str(source_dir),
            "-m",
            "Alice 2026",
            "-o",
            str(output_dir),
            "-e",
            "native",
            "-r",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "a_wm.png").exists()
    assert (output_dir / "nested" / "b_wm.png").exists()


def test_extract_directory_recursive_command(
    sample_image_path: Path, tmp_path: Path
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    watermarked = nested / "sample_wm.png"

    embed_result = CliRunner().invoke(
        main,
        [
            "embed",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "-o",
            str(watermarked),
            "-e",
            "native",
        ],
    )
    assert embed_result.exit_code == 0

    extract_result = CliRunner().invoke(
        main,
        [
            "extract",
            str(tmp_path),
            "-e",
            "native",
            "-r",
            "--json",
        ],
    )

    assert extract_result.exit_code == 0
    assert "Alice 2026" in extract_result.output


def test_verify_directory_command(sample_image_path: Path, tmp_path: Path) -> None:
    watermarked = tmp_path / "sample_wm.png"

    embed_result = CliRunner().invoke(
        main,
        [
            "embed",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "-o",
            str(watermarked),
            "-e",
            "native",
        ],
    )
    assert embed_result.exit_code == 0

    verify_result = CliRunner().invoke(
        main,
        [
            "verify",
            str(tmp_path),
            "-m",
            "Alice 2026",
            "-e",
            "native",
            "--json",
        ],
    )

    assert verify_result.exit_code == 1
    assert '"matched": true' in verify_result.output


def test_info_directory_command(sample_image_path: Path, tmp_path: Path) -> None:
    watermarked = tmp_path / "sample_wm.png"

    embed_result = CliRunner().invoke(
        main,
        [
            "embed",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "-o",
            str(watermarked),
            "-e",
            "native",
        ],
    )
    assert embed_result.exit_code == 0

    info_result = CliRunner().invoke(
        main,
        [
            "info",
            str(tmp_path),
            "-e",
            "native",
            "--json",
        ],
    )

    assert info_result.exit_code == 0
    assert '"found": true' in info_result.output


def test_embed_directory_command_preserves_zero_workers(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / "a.png").write_bytes(sample_image_path.read_bytes())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "stegmark.cli.load_config",
        lambda: SimpleNamespace(engine="native", strength=1.5, workers=9),
    )

    def fake_embed_directory(input_path: Path, **kwargs: object) -> SimpleNamespace:
        captured["input_path"] = input_path
        captured.update(kwargs)
        return SimpleNamespace(total=1, succeeded=1, failed=0, items=())

    monkeypatch.setattr("stegmark.cli.stegmark.embed_directory", fake_embed_directory)

    result = CliRunner().invoke(
        main,
        [
            "embed",
            str(source_dir),
            "-m",
            "Alice 2026",
            "-w",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert captured["workers"] == 0
