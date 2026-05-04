from __future__ import annotations

import argparse
from pathlib import Path

from stegmark.evaluation.gates import evaluate_benchmark_report_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a StegMark benchmark report against quality gates.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-average-bit-accuracy", type=float, default=None)
    parser.add_argument("--min-average-psnr", type=float, default=None)
    parser.add_argument("--require-all-matches", action="store_true")
    parser.add_argument("--require-all-found", action="store_true")
    args = parser.parse_args()

    result = evaluate_benchmark_report_file(
        args.report,
        min_average_bit_accuracy=args.min_average_bit_accuracy,
        min_average_psnr=args.min_average_psnr,
        require_all_matches=args.require_all_matches,
        require_all_found=args.require_all_found,
    )
    if result.passed:
        print(f"Gate passed ({result.scope})")
        return 0
    print(f"Gate failed ({result.scope})")
    for failure in result.failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

