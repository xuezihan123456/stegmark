from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from stegmark.cli import main


def test_benchmark_command_outputs_json(sample_image_path: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "--attacks",
            "jpeg_q90,brightness_1.3",
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "attack_results" in result.output


def test_benchmark_command_writes_csv_report(sample_image_path: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "--attacks",
            "jpeg_q90",
            "--output-dir",
            str(tmp_path),
            "--report-format",
            "csv",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "benchmark.csv").exists()


def test_benchmark_command_prints_table_output(sample_image_path: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "--attacks",
            "jpeg_q90",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Attack" in result.output
    assert "Summary" in result.output


def test_benchmark_compare_engines_outputs_json(sample_image_path: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "--engines",
            "native,hidden",
            "--attacks",
            "jpeg_q90",
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"engine_results"' in result.output
    assert '"native"' in result.output
    assert '"hidden"' in result.output


def test_benchmark_compare_engines_prints_table_output(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "--engines",
            "native,hidden",
            "--attacks",
            "jpeg_q90",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Engine" in result.output
    assert "Summary" in result.output


def test_benchmark_compare_engines_returns_nonzero_when_gate_fails(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "--engines",
            "native,hidden",
            "--attacks",
            "jpeg_q90",
            "--output-dir",
            str(tmp_path),
            "--min-average-bit-accuracy",
            "1.1",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert '"gate"' in result.output


def test_benchmark_command_returns_nonzero_when_gate_fails(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            str(sample_image_path),
            "-m",
            "Alice 2026",
            "--attacks",
            "jpeg_q90",
            "--output-dir",
            str(tmp_path),
            "--min-average-bit-accuracy",
            "1.1",
        ],
    )

    assert result.exit_code == 1
    assert "Gate failed" in result.output
