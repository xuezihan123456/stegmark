from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from stegmark.types import ImageArray, PathLike


def compute_bit_accuracy(expected: Sequence[int], actual: Sequence[int]) -> float:
    if not expected and not actual:
        return 1.0
    total = max(len(expected), len(actual))
    if total == 0:
        return 1.0
    matched = sum(int(left == right) for left, right in zip(expected, actual, strict=False))
    return matched / total


def compute_psnr(reference: ImageArray, candidate: ImageArray) -> float:
    reference_f = reference.astype(np.float32)
    candidate_f = candidate.astype(np.float32)
    mse = float(np.mean((reference_f - candidate_f) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def save_diff_image(
    reference: ImageArray,
    candidate: ImageArray,
    output_path: PathLike,
) -> Path:
    output = Path(output_path)
    diff = np.abs(reference.astype(np.int16) - candidate.astype(np.int16)).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(diff, mode="RGB").save(output)
    return output


def save_compare_report(
    *,
    input_path: PathLike,
    output_path: PathLike,
    psnr: float,
    diff_image_path: PathLike,
    report_path: PathLike,
) -> Path:
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "psnr": psnr,
                "diff_image": str(diff_image_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return report
