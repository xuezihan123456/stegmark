"""Evaluation and benchmark helpers for StegMark."""

from stegmark.evaluation.attacks import AVAILABLE_ATTACKS, apply_attack
from stegmark.evaluation.benchmark import (
    BenchmarkAttackResult,
    BenchmarkComparisonResult,
    BenchmarkEngineResult,
    BenchmarkResult,
    BenchmarkSummary,
    benchmark_compare_engines,
    benchmark_file,
)
from stegmark.evaluation.gates import GateCheckResult, evaluate_benchmark_report, evaluate_benchmark_report_file

__all__ = [
    "AVAILABLE_ATTACKS",
    "BenchmarkAttackResult",
    "BenchmarkComparisonResult",
    "BenchmarkEngineResult",
    "GateCheckResult",
    "BenchmarkResult",
    "BenchmarkSummary",
    "apply_attack",
    "benchmark_compare_engines",
    "benchmark_file",
    "evaluate_benchmark_report",
    "evaluate_benchmark_report_file",
]
