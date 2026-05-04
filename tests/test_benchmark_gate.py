from __future__ import annotations

import json
from pathlib import Path

from stegmark.evaluation.gates import GateCheckResult, evaluate_benchmark_report_file


def test_evaluate_single_benchmark_report_passes(tmp_path: Path) -> None:
    report = tmp_path / "benchmark.json"
    report.write_text(
        json.dumps(
            {
                "engine": "native",
                "summary": {
                    "attack_count": 2,
                    "message_match_count": 2,
                    "found_count": 2,
                    "average_bit_accuracy": 0.95,
                    "average_psnr": 38.0,
                    "worst_attack": "jpeg_q90",
                },
                "attack_results": {},
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_benchmark_report_file(report, min_average_bit_accuracy=0.9)

    assert isinstance(result, GateCheckResult)
    assert result.passed is True
    assert result.scope == "single"


def test_evaluate_single_benchmark_report_fails(tmp_path: Path) -> None:
    report = tmp_path / "benchmark.json"
    report.write_text(
        json.dumps(
            {
                "engine": "native",
                "summary": {
                    "attack_count": 2,
                    "message_match_count": 1,
                    "found_count": 2,
                    "average_bit_accuracy": 0.75,
                    "average_psnr": 34.0,
                    "worst_attack": "jpeg_q50",
                },
                "attack_results": {},
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_benchmark_report_file(
        report,
        min_average_bit_accuracy=0.9,
        require_all_matches=True,
    )

    assert result.passed is False
    assert result.failures


def test_evaluate_compare_report_passes_when_one_engine_passes(tmp_path: Path) -> None:
    report = tmp_path / "benchmark_compare.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "total_engines": 2,
                    "available_engines": 1,
                    "unavailable_engines": 1,
                    "best_engine_by_average_bit_accuracy": "native",
                },
                "engine_results": {
                    "native": {
                        "engine": "native",
                        "available": True,
                        "error": None,
                        "result": {
                            "engine": "native",
                            "summary": {
                                "attack_count": 1,
                                "message_match_count": 1,
                                "found_count": 1,
                                "average_bit_accuracy": 1.0,
                                "average_psnr": 40.0,
                                "worst_attack": "jpeg_q90",
                            },
                            "attack_results": {},
                        },
                    },
                    "hidden": {
                        "engine": "hidden",
                        "available": False,
                        "error": "hidden engine weights are missing",
                        "result": None,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_benchmark_report_file(report, min_average_bit_accuracy=0.9)

    assert result.passed is True
    assert result.scope == "comparison"


def test_evaluate_compare_report_fails_when_no_engine_passes(tmp_path: Path) -> None:
    report = tmp_path / "benchmark_compare.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "total_engines": 2,
                    "available_engines": 1,
                    "unavailable_engines": 1,
                    "best_engine_by_average_bit_accuracy": "native",
                },
                "engine_results": {
                    "native": {
                        "engine": "native",
                        "available": True,
                        "error": None,
                        "result": {
                            "engine": "native",
                            "summary": {
                                "attack_count": 1,
                                "message_match_count": 0,
                                "found_count": 1,
                                "average_bit_accuracy": 0.2,
                                "average_psnr": 20.0,
                                "worst_attack": "jpeg_q50",
                            },
                            "attack_results": {},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_benchmark_report_file(
        report,
        min_average_bit_accuracy=0.9,
        require_all_matches=True,
    )

    assert result.passed is False
    assert result.failures
