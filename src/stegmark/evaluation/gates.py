from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stegmark.types import PathLike


@dataclass(frozen=True)
class GateCheckResult:
    passed: bool
    scope: str
    failures: tuple[str, ...] = ()


def evaluate_benchmark_report_file(
    path: PathLike,
    *,
    min_average_bit_accuracy: float | None = None,
    min_average_psnr: float | None = None,
    require_all_matches: bool = False,
    require_all_found: bool = False,
) -> GateCheckResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return evaluate_benchmark_report(
        payload,
        min_average_bit_accuracy=min_average_bit_accuracy,
        min_average_psnr=min_average_psnr,
        require_all_matches=require_all_matches,
        require_all_found=require_all_found,
    )


def evaluate_benchmark_report(
    payload: dict[str, Any],
    *,
    min_average_bit_accuracy: float | None = None,
    min_average_psnr: float | None = None,
    require_all_matches: bool = False,
    require_all_found: bool = False,
) -> GateCheckResult:
    if "engine_results" in payload:
        return _evaluate_comparison_report(
            payload,
            min_average_bit_accuracy=min_average_bit_accuracy,
            min_average_psnr=min_average_psnr,
            require_all_matches=require_all_matches,
            require_all_found=require_all_found,
        )
    return _evaluate_single_report(
        payload,
        min_average_bit_accuracy=min_average_bit_accuracy,
        min_average_psnr=min_average_psnr,
        require_all_matches=require_all_matches,
        require_all_found=require_all_found,
    )


def _evaluate_single_report(
    payload: dict[str, Any],
    *,
    min_average_bit_accuracy: float | None,
    min_average_psnr: float | None,
    require_all_matches: bool,
    require_all_found: bool,
) -> GateCheckResult:
    summary = payload["summary"]
    failures = _summary_failures(
        summary,
        min_average_bit_accuracy=min_average_bit_accuracy,
        min_average_psnr=min_average_psnr,
        require_all_matches=require_all_matches,
        require_all_found=require_all_found,
    )
    return GateCheckResult(
        passed=not failures,
        scope="single",
        failures=tuple(failures),
    )


def _evaluate_comparison_report(
    payload: dict[str, Any],
    *,
    min_average_bit_accuracy: float | None,
    min_average_psnr: float | None,
    require_all_matches: bool,
    require_all_found: bool,
) -> GateCheckResult:
    engine_results = payload["engine_results"]
    available_payloads = [
        result_payload["result"]
        for result_payload in engine_results.values()
        if result_payload.get("available") and result_payload.get("result") is not None
    ]
    if not available_payloads:
        return GateCheckResult(
            passed=False,
            scope="comparison",
            failures=("no available engines for comparison",),
        )

    passing = False
    failures: list[str] = []
    for result_payload in available_payloads:
        summary = result_payload["summary"]
        summary_failures = _summary_failures(
            summary,
            min_average_bit_accuracy=min_average_bit_accuracy,
            min_average_psnr=min_average_psnr,
            require_all_matches=require_all_matches,
            require_all_found=require_all_found,
        )
        if not summary_failures:
            passing = True
        else:
            failures.append(
                f"{result_payload['engine']}: {'; '.join(summary_failures)}"
            )
    return GateCheckResult(
        passed=passing,
        scope="comparison",
        failures=tuple(failures if not passing else ()),
    )


def _summary_failures(
    summary: dict[str, Any],
    *,
    min_average_bit_accuracy: float | None,
    min_average_psnr: float | None,
    require_all_matches: bool,
    require_all_found: bool,
) -> list[str]:
    failures: list[str] = []
    average_bit_accuracy = float(summary["average_bit_accuracy"])
    average_psnr = float(summary["average_psnr"])
    attack_count = int(summary["attack_count"])
    message_match_count = int(summary["message_match_count"])
    found_count = int(summary["found_count"])
    if min_average_bit_accuracy is not None and average_bit_accuracy < min_average_bit_accuracy:
        failures.append(
            f"average_bit_accuracy {average_bit_accuracy:.3f} < required {min_average_bit_accuracy:.3f}"
        )
    if min_average_psnr is not None and average_psnr < min_average_psnr:
        failures.append(
            f"average_psnr {average_psnr:.2f} < required {min_average_psnr:.2f}"
        )
    if require_all_matches and message_match_count != attack_count:
        failures.append(
            f"message matches {message_match_count}/{attack_count} did not meet all-match requirement"
        )
    if require_all_found and found_count != attack_count:
        failures.append(
            f"found count {found_count}/{attack_count} did not meet all-found requirement"
        )
    return failures

