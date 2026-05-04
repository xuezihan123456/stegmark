from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from stegmark.cli import main


def test_embed_command_outputs_file(sample_image_path: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "cli-output.png"

    result = runner.invoke(
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
        ],
    )

    assert result.exit_code == 0
    assert output.exists()


def test_extract_and_verify_commands(sample_image_path: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "cli-roundtrip.png"
    runner.invoke(
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
        ],
    )

    extract_result = runner.invoke(main, ["extract", str(output), "-e", "native"])
    verify_result = runner.invoke(main, ["verify", str(output), "-m", "Alice 2026", "-e", "native"])

    assert extract_result.exit_code == 0
    assert "Alice 2026" in extract_result.output
    assert verify_result.exit_code == 0


def test_embed_command_preserves_zero_strength(
    sample_image_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    output = sample_image_path.parent / "zero-strength.png"

    monkeypatch.setattr(
        "stegmark.cli.load_config",
        lambda: SimpleNamespace(engine="native", strength=2.5, workers=7),
    )

    def fake_embed(input_path: Path, message: str | None, **kwargs: object) -> SimpleNamespace:
        captured["input_path"] = input_path
        captured["message"] = message
        captured.update(kwargs)
        return SimpleNamespace(
            output_path=kwargs["output"],
            engine="native",
            message=message,
            psnr=None,
            compare_report=None,
        )

    monkeypatch.setattr("stegmark.cli.stegmark.embed", fake_embed)

    result = CliRunner().invoke(
        main,
        [
            "embed",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "-o",
            str(output),
            "--strength",
            "0.0",
        ],
    )

    assert result.exit_code == 0
    assert captured["strength"] == 0.0


def test_embed_command_accepts_empty_message(
    sample_image_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    output = sample_image_path.parent / "empty-message.png"

    monkeypatch.setattr(
        "stegmark.cli.load_config",
        lambda: SimpleNamespace(engine="native", strength=1.0, workers=4),
    )

    def fake_embed(input_path: Path, message: str | None, **kwargs: object) -> SimpleNamespace:
        captured["input_path"] = input_path
        captured["message"] = message
        captured.update(kwargs)
        return SimpleNamespace(
            output_path=kwargs["output"],
            engine="native",
            message=message,
            psnr=None,
            compare_report=None,
        )

    monkeypatch.setattr("stegmark.cli.stegmark.embed", fake_embed)

    result = CliRunner().invoke(
        main,
        [
            "embed",
            str(sample_image_path),
            "-m",
            "",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert captured["message"] == ""


def test_embed_command_rejects_message_and_empty_bits_together(
    sample_image_path: Path,
    monkeypatch,
) -> None:
    called = False

    monkeypatch.setattr(
        "stegmark.cli.load_config",
        lambda: SimpleNamespace(engine="native", strength=1.0, workers=4),
    )

    def fake_embed(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal called
        called = True
        return SimpleNamespace(
            output_path=Path("ignored.png"),
            engine="native",
            message="ignored",
            psnr=None,
            compare_report=None,
        )

    monkeypatch.setattr("stegmark.cli.stegmark.embed", fake_embed)

    result = CliRunner().invoke(
        main,
        [
            "embed",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "-b",
            "",
            "-o",
            str(sample_image_path.parent / "ignored.png"),
        ],
    )

    assert result.exit_code != 0
    assert "Provide exactly one of --message or --bits." in result.output
    assert called is False
