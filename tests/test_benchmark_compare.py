from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stegmark.evaluation.benchmark import benchmark_compare_engines


def test_benchmark_compare_engines_reports_available_and_unavailable_backends(
    sample_image_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_model_dir = tmp_path / "missing-hidden"
    monkeypatch.setattr(
        "stegmark.core.hidden.load_config",
        lambda: SimpleNamespace(hidden_model_dir=missing_model_dir),
    )

    result = benchmark_compare_engines(
        sample_image_path,
        message="Alice 2026",
        engines=["native", "hidden"],
        attacks=["brightness_1.3"],
    )

    assert set(result.engines) == {"native", "hidden"}
    assert result.engines["native"].available is True
    assert result.engines["native"].result is not None
    assert result.engines["hidden"].available is False
    assert result.engines["hidden"].result is None
    assert "hidden engine weights are missing" in (result.engines["hidden"].error or "")
    assert result.summary.best_engine_by_average_bit_accuracy == "native"


def test_benchmark_compare_engines_can_write_csv_report(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = benchmark_compare_engines(
        sample_image_path,
        message="Alice 2026",
        engines=["native"],
        attacks=["jpeg_q90"],
        output_dir=tmp_path,
        report_format="csv",
    )

    report = tmp_path / "benchmark_compare.csv"

    assert result.output_report == report
    assert report.exists()
    assert "engine,available,average_bit_accuracy,average_psnr,worst_attack,error" in report.read_text(
        encoding="utf-8"
    )


def test_benchmark_compare_engines_can_write_html_report(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = benchmark_compare_engines(
        sample_image_path,
        message="Alice 2026",
        engines=["native"],
        attacks=["jpeg_q90"],
        output_dir=tmp_path,
        report_format="html",
    )

    report = tmp_path / "benchmark_compare.html"

    assert result.output_report == report
    assert report.exists()
    assert "<table>" in report.read_text(encoding="utf-8")


def test_benchmark_compare_terminal_text_contains_summary(
    sample_image_path: Path,
) -> None:
    result = benchmark_compare_engines(
        sample_image_path,
        message="Alice 2026",
        engines=["native"],
        attacks=["jpeg_q90"],
    )

    text = result.to_terminal_text()

    assert "StegMark Benchmark Compare" in text
    assert "Summary:" in text


def test_benchmark_compare_can_evaluate_gate(sample_image_path: Path) -> None:
    result = benchmark_compare_engines(
        sample_image_path,
        message="Alice 2026",
        engines=["native"],
        attacks=["jpeg_q90"],
    )

    passed = result.evaluate_gate(min_average_bit_accuracy=0.9)
    failed = result.evaluate_gate(min_average_bit_accuracy=1.1)

    assert passed.passed is True
    assert failed.passed is False
