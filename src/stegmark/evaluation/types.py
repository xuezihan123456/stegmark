from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rich.table import Table

from stegmark.evaluation.reports import (
    benchmark_csv_text,
    benchmark_html,
    benchmark_table,
    benchmark_terminal_text,
    comparison_terminal_text,
)


@dataclass(frozen=True)
class BenchmarkAttackResult:
    attack: str
    message_match: bool
    extracted_message: str | None
    found: bool
    bit_accuracy: float
    psnr: float
    elapsed: float
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    attack_count: int
    message_match_count: int
    found_count: int
    average_bit_accuracy: float
    average_psnr: float
    worst_attack: str | None


@dataclass(frozen=True)
class BenchmarkEngineResult:
    engine: str
    available: bool
    result: BenchmarkResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkComparisonSummary:
    total_engines: int
    available_engines: int
    unavailable_engines: int
    best_engine_by_average_bit_accuracy: str | None


@dataclass(frozen=True)
class BenchmarkGateEvaluation:
    passed: bool
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkResult:
    engine: str
    message: str
    attack_results: dict[str, BenchmarkAttackResult]
    output_report: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "message": self.message,
            "summary": asdict(self.summary),
            "attack_results": {name: asdict(result) for name, result in self.attack_results.items()},
            "output_report": str(self.output_report) if self.output_report else None,
        }

    @property
    def summary(self) -> BenchmarkSummary:
        results = list(self.attack_results.values())
        if not results:
            return BenchmarkSummary(
                attack_count=0,
                message_match_count=0,
                found_count=0,
                average_bit_accuracy=0.0,
                average_psnr=0.0,
                worst_attack=None,
            )
        average_bit_accuracy = sum(result.bit_accuracy for result in results) / len(results)
        average_psnr = sum(result.psnr for result in results) / len(results)
        worst = min(results, key=lambda result: result.bit_accuracy)
        return BenchmarkSummary(
            attack_count=len(results),
            message_match_count=sum(int(result.message_match) for result in results),
            found_count=sum(int(result.found) for result in results),
            average_bit_accuracy=average_bit_accuracy,
            average_psnr=average_psnr,
            worst_attack=worst.attack,
        )

    def to_rows(self) -> list[dict[str, object]]:
        return [asdict(result) for result in self.attack_results.values()]

    def to_csv(self) -> str:
        return benchmark_csv_text(self)

    def to_html(self) -> str:
        return benchmark_html(self)

    def to_table(self) -> Table:
        return benchmark_table(self)

    def to_terminal_text(self) -> str:
        return benchmark_terminal_text(self)

    def evaluate_gate(
        self,
        *,
        min_average_bit_accuracy: float | None = None,
        min_average_psnr: float | None = None,
        require_all_matches: bool = False,
        require_all_found: bool = False,
    ) -> BenchmarkGateEvaluation:
        failures: list[str] = []
        summary = self.summary
        if min_average_bit_accuracy is not None and summary.average_bit_accuracy < min_average_bit_accuracy:
            failures.append(
                "average_bit_accuracy "
                f"{summary.average_bit_accuracy:.3f} < required {min_average_bit_accuracy:.3f}"
            )
        if min_average_psnr is not None and summary.average_psnr < min_average_psnr:
            failures.append(
                f"average_psnr {summary.average_psnr:.2f} < required {min_average_psnr:.2f}"
            )
        if require_all_matches and summary.message_match_count != summary.attack_count:
            failures.append(
                "message matches "
                f"{summary.message_match_count}/{summary.attack_count} "
                "did not meet all-match requirement"
            )
        if require_all_found and summary.found_count != summary.attack_count:
            failures.append(
                "found count "
                f"{summary.found_count}/{summary.attack_count} "
                "did not meet all-found requirement"
            )
        return BenchmarkGateEvaluation(passed=not failures, failures=tuple(failures))


@dataclass(frozen=True)
class BenchmarkComparisonResult:
    message: str
    engines: dict[str, BenchmarkEngineResult]
    output_report: Path | None = None

    @property
    def summary(self) -> BenchmarkComparisonSummary:
        available_results = [
            item for item in self.engines.values() if item.available and item.result is not None
        ]
        best_engine: str | None = None
        if available_results:
            best_engine = max(
                available_results,
                key=lambda item: item.result.summary.average_bit_accuracy if item.result is not None else -1.0,
            ).engine
        return BenchmarkComparisonSummary(
            total_engines=len(self.engines),
            available_engines=len(available_results),
            unavailable_engines=len(self.engines) - len(available_results),
            best_engine_by_average_bit_accuracy=best_engine,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "message": self.message,
            "summary": asdict(self.summary),
            "engine_results": {
                engine: {
                    "engine": result.engine,
                    "available": result.available,
                    "error": result.error,
                    "result": result.result.to_dict() if result.result is not None else None,
                }
                for engine, result in self.engines.items()
            },
            "output_report": str(self.output_report) if self.output_report else None,
        }

    def to_terminal_text(self) -> str:
        return comparison_terminal_text(self)

    def evaluate_gate(
        self,
        *,
        min_average_bit_accuracy: float | None = None,
        min_average_psnr: float | None = None,
        require_all_matches: bool = False,
        require_all_found: bool = False,
    ) -> BenchmarkGateEvaluation:
        available_results = [
            item for item in self.engines.values() if item.available and item.result is not None
        ]
        if not available_results:
            return BenchmarkGateEvaluation(
                passed=False,
                failures=("no available engines for comparison",),
            )
        passing_engines: list[str] = []
        failure_lines: list[str] = []
        for engine_result in available_results:
            benchmark_result = engine_result.result
            if benchmark_result is None:
                continue
            gate = benchmark_result.evaluate_gate(
                min_average_bit_accuracy=min_average_bit_accuracy,
                min_average_psnr=min_average_psnr,
                require_all_matches=require_all_matches,
                require_all_found=require_all_found,
            )
            if gate.passed:
                passing_engines.append(engine_result.engine)
            else:
                failure_lines.append(f"{engine_result.engine}: {'; '.join(gate.failures)}")
        if passing_engines:
            return BenchmarkGateEvaluation(passed=True)
        return BenchmarkGateEvaluation(passed=False, failures=tuple(failure_lines))
