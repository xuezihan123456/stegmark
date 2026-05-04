from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from stegmark.core.codec import encode_text
from stegmark.core.image_io import load_image
from stegmark.core.registry import get_engine
from stegmark.evaluation.attacks import AVAILABLE_ATTACKS, apply_attack
from stegmark.evaluation.metrics import compute_bit_accuracy, compute_psnr
from stegmark.evaluation.reports import (
    benchmark_csv_text,
    comparison_csv_text,
    comparison_html,
    render_benchmark_comparison_report,
    render_benchmark_report,
)
from stegmark.evaluation.types import (
    BenchmarkAttackResult,
    BenchmarkComparisonResult,
    BenchmarkComparisonSummary,
    BenchmarkEngineResult,
    BenchmarkGateEvaluation,
    BenchmarkResult,
    BenchmarkSummary,
)
from stegmark.exceptions import StegMarkError
from stegmark.types import ImageArray, PathLike

DEFAULT_ATTACKS = ("jpeg_q90", "jpeg_q75", "brightness_1.3", "gaussian_noise_0.03")

__all__ = [
    "DEFAULT_ATTACKS",
    "BenchmarkAttackResult",
    "BenchmarkComparisonResult",
    "BenchmarkComparisonSummary",
    "BenchmarkEngineResult",
    "BenchmarkGateEvaluation",
    "BenchmarkResult",
    "BenchmarkSummary",
    "benchmark_compare_engines",
    "benchmark_file",
    "list_attacks",
]


def benchmark_file(
    input_path: PathLike,
    *,
    message: str,
    engine: str = "native",
    attacks: list[str] | tuple[str, ...] | None = None,
    output_dir: PathLike | None = None,
    report_format: str = "json",
) -> BenchmarkResult:
    loaded = load_image(input_path)
    backend = get_engine(engine)
    encoded = backend.encode(loaded.array, message)
    expected_bits = tuple(encode_text(message))
    attack_names = tuple(attacks or DEFAULT_ATTACKS)
    result = BenchmarkResult(
        engine=backend.name,
        message=message,
        attack_results=_run_benchmark_attacks(
            backend=backend,
            encoded=encoded,
            message=message,
            expected_bits=expected_bits,
            attack_names=attack_names,
        ),
    )
    return _with_report(
        result=result,
        output_dir=output_dir,
        report_name="benchmark",
        report_format=report_format,
        renderer=_render_report,
    )


def benchmark_compare_engines(
    input_path: PathLike,
    *,
    message: str,
    engines: list[str] | tuple[str, ...],
    attacks: list[str] | tuple[str, ...] | None = None,
    output_dir: PathLike | None = None,
    report_format: str = "json",
) -> BenchmarkComparisonResult:
    engine_results: dict[str, BenchmarkEngineResult] = {}
    for engine_name in engines:
        try:
            result = benchmark_file(
                input_path,
                message=message,
                engine=engine_name,
                attacks=attacks,
                output_dir=None,
                report_format=report_format,
            )
        except StegMarkError as exc:
            engine_results[engine_name] = BenchmarkEngineResult(
                engine=engine_name,
                available=False,
                result=None,
                error=str(exc),
            )
        else:
            engine_results[engine_name] = BenchmarkEngineResult(
                engine=engine_name,
                available=True,
                result=result,
            )

    comparison_result = BenchmarkComparisonResult(
        message=message,
        engines=engine_results,
    )
    return _with_report(
        result=comparison_result,
        output_dir=output_dir,
        report_name="benchmark_compare",
        report_format=report_format,
        renderer=_render_comparison_report,
    )


def list_attacks() -> tuple[str, ...]:
    return tuple(sorted(AVAILABLE_ATTACKS))


def _run_benchmark_attacks(
    *,
    backend: Any,
    encoded: ImageArray,
    message: str,
    expected_bits: tuple[int, ...],
    attack_names: tuple[str, ...],
) -> dict[str, BenchmarkAttackResult]:
    if not attack_names:
        return {}
    max_workers = min(32, len(attack_names)) or 1
    results: dict[str, BenchmarkAttackResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[Future[BenchmarkAttackResult]] = [
            executor.submit(
                _benchmark_single_attack,
                backend=backend,
                encoded=encoded,
                message=message,
                expected_bits=expected_bits,
                attack_name=attack_name,
            )
            for attack_name in attack_names
        ]
        for index, (attack_name, future) in enumerate(zip(attack_names, futures, strict=False)):
            try:
                results[attack_name] = future.result()
            except BaseException:
                for pending in futures[index + 1 :]:
                    pending.cancel()
                raise
    return results


def _benchmark_single_attack(
    *,
    backend: Any,
    encoded: ImageArray,
    message: str,
    expected_bits: tuple[int, ...],
    attack_name: str,
) -> BenchmarkAttackResult:
    started = perf_counter()
    attacked = apply_attack(encoded, attack_name)
    extracted = backend.decode(attacked)
    elapsed = perf_counter() - started
    return BenchmarkAttackResult(
        attack=attack_name,
        message_match=extracted.message == message,
        extracted_message=extracted.message,
        found=extracted.found,
        bit_accuracy=_bit_accuracy(expected_bits, extracted.bits),
        psnr=_psnr(encoded, attacked),
        elapsed=elapsed,
        error=extracted.error,
    )


def _with_report(
    *,
    result: BenchmarkResult | BenchmarkComparisonResult,
    output_dir: PathLike | None,
    report_name: str,
    report_format: str,
    renderer: Any,
) -> BenchmarkResult | BenchmarkComparisonResult:
    report_path: Path | None = None
    if output_dir is not None:
        report_path = Path(output_dir) / f"{report_name}.{report_format}"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(renderer(result, report_format), encoding="utf-8")
    if isinstance(result, BenchmarkResult):
        return BenchmarkResult(
            engine=result.engine,
            message=result.message,
            attack_results=result.attack_results,
            output_report=report_path,
        )
    return BenchmarkComparisonResult(
        message=result.message,
        engines=result.engines,
        output_report=report_path,
    )


def _render_report(result: BenchmarkResult, report_format: str) -> str:
    return render_benchmark_report(result, report_format)


def _render_comparison_report(result: BenchmarkComparisonResult, report_format: str) -> str:
    return render_benchmark_comparison_report(result, report_format)



def _bit_accuracy(expected: tuple[int, ...], actual: tuple[int, ...]) -> float:
    correct = sum(a == e for a, e in zip(actual, expected))
    return correct / max(len(actual), len(expected))


def _psnr(reference: ImageArray, candidate: ImageArray) -> float:
    return compute_psnr(reference, candidate)
