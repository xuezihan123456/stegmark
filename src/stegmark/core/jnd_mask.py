"""Watson DCT-based Just-Noticeable-Difference (JND) perceptual mask.

Replaces the fixed ``BASE_DELTA`` strength in :mod:`stegmark.core.native`
with per-block adaptive strength derived from human visual sensitivity.

References
----------
Watson, A. B. (1993). "DCT quantization matrices visually optimized for
individual images." Human Vision, Visual Processing, and Digital Display IV.

The model produces a per-coefficient JND threshold matrix, modulated by:
  * Luminance masking (eye is less sensitive in bright regions)
  * Contrast masking (textured regions hide perturbations)

Pure-numpy; no torch / Pillow dependency.
"""
from __future__ import annotations

from typing import cast

import numpy as np

BLOCK_SIZE = 8

# Watson 1993 base JND threshold table (Table 1, normalised).
# Values calibrated for an average viewing distance with monitor luminance ~127.
_WATSON_BASIC: np.ndarray = np.array(
    [
        [1.40, 1.01, 1.16, 1.66, 2.40, 3.43, 4.79, 6.56],
        [1.01, 1.45, 1.32, 1.52, 2.00, 2.71, 3.67, 4.93],
        [1.16, 1.32, 2.24, 2.59, 2.98, 3.64, 4.60, 5.88],
        [1.66, 1.52, 2.59, 3.77, 4.55, 5.30, 6.28, 7.60],
        [2.40, 2.00, 2.98, 4.55, 6.15, 7.46, 8.71, 10.17],
        [3.43, 2.71, 3.64, 5.30, 7.46, 9.62, 11.58, 13.51],
        [4.79, 3.67, 4.60, 6.28, 8.71, 11.58, 14.50, 17.29],
        [6.56, 4.93, 5.88, 7.60, 10.17, 13.51, 17.29, 21.15],
    ],
    dtype=np.float32,
)

_MEAN_LUMINANCE_REF = 127.5
_LUMINANCE_EXPONENT = 0.649
_CONTRAST_EXPONENT = 0.7


def watson_dct_jnd(block_dct: np.ndarray, luminance: float) -> np.ndarray:
    """Return per-coefficient JND threshold matrix for an 8x8 DCT block.

    Parameters
    ----------
    block_dct:
        DCT coefficients of an 8x8 luminance block (float).
    luminance:
        Mean luminance of the block in the 0-255 range.
    """
    if block_dct.shape != (BLOCK_SIZE, BLOCK_SIZE):
        raise ValueError(f"block_dct must be {BLOCK_SIZE}x{BLOCK_SIZE}, got {block_dct.shape}")

    luminance = float(max(1.0, luminance))
    luminance_factor = (luminance / _MEAN_LUMINANCE_REF) ** _LUMINANCE_EXPONENT
    thresholds = _WATSON_BASIC * luminance_factor

    # Contrast masking: each AC coefficient raises its own threshold.
    coeff_abs = np.abs(block_dct).astype(np.float32, copy=False)
    masked = np.maximum(thresholds, thresholds * (coeff_abs / np.maximum(thresholds, 1e-6)) ** _CONTRAST_EXPONENT)
    # DC term (0, 0) is kept at the base threshold to avoid over-amplifying flat regions.
    masked[0, 0] = thresholds[0, 0]
    return masked.astype(np.float32, copy=False)


def compute_block_strength(
    block_dct: np.ndarray,
    base_delta: float,
    min_scale: float = 0.5,
    max_scale: float = 3.0,
) -> float:
    """Return adaptive strength multiplier for a single 8x8 DCT block.

    The geometric mean of the AC JND thresholds is compared against an anchor
    value, then clipped to ``[min_scale, max_scale]``.
    """
    if base_delta <= 0:
        raise ValueError("base_delta must be positive")
    if min_scale <= 0 or max_scale < min_scale:
        raise ValueError("scale bounds must satisfy 0 < min_scale <= max_scale")

    luminance = float(np.clip(block_dct[0, 0] / 8.0 + 128.0, 0.0, 255.0))
    jnd = watson_dct_jnd(block_dct, luminance)
    baseline = watson_dct_jnd(np.zeros_like(block_dct), luminance)
    ac_mask = np.ones_like(jnd, dtype=bool)
    ac_mask[0, 0] = False
    actual = float(np.exp(np.mean(np.log(np.maximum(jnd[ac_mask], 1e-6)))))
    base = float(np.exp(np.mean(np.log(np.maximum(baseline[ac_mask], 1e-6)))))
    scale = actual / max(base, 1e-6)
    return float(np.clip(scale, min_scale, max_scale))


def compute_strength_map(
    luminance_channel: np.ndarray,
    base_delta: float,
    min_scale: float = 0.5,
    max_scale: float = 3.0,
) -> np.ndarray:
    """Return per-block adaptive strength map for a Y-channel image.

    The image is split into 8x8 blocks; blocks that fall off the right/bottom
    edges are ignored. Output shape is ``(H // 8, W // 8)``.
    """
    if luminance_channel.ndim != 2:
        raise ValueError("luminance_channel must be 2-D")

    height, width = luminance_channel.shape
    usable_h = height - (height % BLOCK_SIZE)
    usable_w = width - (width % BLOCK_SIZE)
    block_rows = usable_h // BLOCK_SIZE
    block_cols = usable_w // BLOCK_SIZE
    if block_rows == 0 or block_cols == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Build the DCT matrix once.
    matrix = _dct_matrix()
    trimmed = luminance_channel[:usable_h, :usable_w].astype(np.float32, copy=False)
    # Reshape to (block_rows, BLOCK_SIZE, block_cols, BLOCK_SIZE) then transpose.
    blocks = trimmed.reshape(block_rows, BLOCK_SIZE, block_cols, BLOCK_SIZE).transpose(0, 2, 1, 3)
    coeffs = np.einsum("ij,rcjk,kl->rcil", matrix, blocks, matrix.T)

    strength = np.empty((block_rows, block_cols), dtype=np.float32)
    for r in range(block_rows):
        for c in range(block_cols):
            strength[r, c] = compute_block_strength(coeffs[r, c], base_delta, min_scale, max_scale)
    return strength


def _dct_matrix(size: int = BLOCK_SIZE) -> np.ndarray:
    rows = np.arange(size, dtype=np.float32)[:, None]
    cols = np.arange(size, dtype=np.float32)[None, :]
    alpha = np.full(size, np.sqrt(2.0 / size), dtype=np.float32)
    alpha[0] = np.sqrt(1.0 / size)
    matrix = alpha[:, None] * np.cos((np.pi * (2.0 * cols + 1.0) * rows) / (2.0 * size))
    return cast(np.ndarray, matrix.astype(np.float32, copy=False))


__all__ = [
    "BLOCK_SIZE",
    "compute_block_strength",
    "compute_strength_map",
    "watson_dct_jnd",
]
