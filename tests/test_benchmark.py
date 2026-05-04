from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich.table import Table

from stegmark.evaluation.attacks import apply_attack
from stegmark.evaluation.benchmark import benchmark_file
from stegmark.evaluation.metrics import compute_bit_accuracy


def test_apply_attack_changes_image(sample_image) -> None:
    attacked = apply_attack(sample_image, "brightness_1.3")

    assert attacked.shape == sample_image.shape


def test_gaussian_noise_attack_accepts_optional_seed(sample_image) -> None:
    seeded = apply_attack(sample_image, "gaussian_noise_0.03", seed=123)
    repeated = apply_attack(sample_image, "gaussian_noise_0.03", seed=123)
    different = apply_attack(sample_image, "gaussian_noise_0.03", seed=456)
    default_seed = apply_attack(sample_image, "gaussian_noise_0.03")
    explicit_default = apply_attack(sample_image, "gaussian_noise_0.03", seed=0)

    assert np.array_equal(seeded, repeated)
    assert not np.array_equal(seeded, different)
    assert np.array_equal(default_seed, explicit_default)


def test_compute_bit_accuracy_penalizes_actual_length_mismatches() -> None:
    assert compute_bit_accuracy((1, 0), (1, 0, 1, 1)) == 0.5
    assert compute_bit_accuracy((), (1,)) == 0.0


def test_benchmark_file_returns_attack_results(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = benchmark_file(
        sample_image_path,
        message="Alice 2026",
        engine="native",
        attacks=["jpeg_q90", "brightness_1.3"],
        output_dir=tmp_path,
    )

    assert result.engine == "native"
    assert set(result.attack_results) == {"jpeg_q90", "brightness_1.3"}
    report = tmp_path / "benchmark.json"
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["engine"] == "native"
    assert payload["summary"]["attack_count"] == 2


def test_benchmark_result_exposes_summary_and_table(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = benchmark_file(
        sample_image_path,
        message="Alice 2026",
        engine="native",
        attacks=["jpeg_q90", "brightness_1.3"],
        output_dir=tmp_path,
    )

    assert result.summary.attack_count == 2
    assert 0.0 <= result.summary.average_bit_accuracy <= 1.0
    assert result.summary.worst_attack in {"jpeg_q90", "brightness_1.3"}
    assert isinstance(result.to_table(), Table)
    assert result.to_table().columns[0].header == "Attack"


def test_benchmark_file_can_write_csv_report(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = benchmark_file(
        sample_image_path,
        message="Alice 2026",
        engine="native",
        attacks=["jpeg_q90"],
        output_dir=tmp_path,
        report_format="csv",
    )

    report = tmp_path / "benchmark.csv"

    assert result.output_report == report
    assert report.exists()
    assert "attack,message_match,extracted_message,found,bit_accuracy,psnr,elapsed,error" in report.read_text(
        encoding="utf-8"
    )


def test_benchmark_file_can_write_html_report(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = benchmark_file(
        sample_image_path,
        message="Alice 2026",
        engine="native",
        attacks=["brightness_1.3"],
        output_dir=tmp_path,
        report_format="html",
    )

    report = tmp_path / "benchmark.html"

    assert result.output_report == report
    assert report.exists()
    assert "<table>" in report.read_text(encoding="utf-8")


def test_benchmark_result_can_evaluate_quality_gate(
    sample_image_path: Path, tmp_path: Path
) -> None:
    result = benchmark_file(
        sample_image_path,
        message="Alice 2026",
        engine="native",
        attacks=["jpeg_q90", "brightness_1.3"],
        output_dir=tmp_path,
    )

    passed = result.evaluate_gate(min_average_bit_accuracy=0.9)
    failed = result.evaluate_gate(min_average_bit_accuracy=1.1)

    assert passed.passed is True
    assert failed.passed is False
    assert failed.failures
