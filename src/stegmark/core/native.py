from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import cast

import numpy as np
from numpy.lib.stride_tricks import as_strided

from stegmark.core.codec import bits_to_bytes, decode_bitstream, resolve_payload_bits
from stegmark.core.engine import WatermarkEngine
from stegmark.exceptions import MessageTooLongError
from stegmark.types import ExtractResult, ImageArray

BLOCK_SIZE = 8
COEFF_A = (3, 2)
COEFF_B = (2, 3)
BASE_DELTA = 12.0
FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32]]


class NativeEngine(WatermarkEngine):
    name = "native"

    def encode(
        self,
        image: ImageArray,
        message: str | None = None,
        *,
        payload_bits: Sequence[int] | None = None,
        strength: float = 1.0,
    ) -> ImageArray:
        bits = resolve_payload_bits(message, payload_bits)
        y_channel, cb_channel, cr_channel = _rgb_to_ycbcr(image)
        encoded_y = y_channel.copy()
        encoded_blocks = _block_view(encoded_y)
        block_rows, block_cols = encoded_blocks.shape[:2]
        capacity = block_rows * block_cols
        if len(bits) > capacity:
            raise MessageTooLongError(
                "message exceeds the native engine capacity for this image",
                hint="Use a shorter message or a larger image.",
            )
        if not bits:
            return _ycbcr_to_rgb(encoded_y, cb_channel, cr_channel)

        delta = max(4.0, BASE_DELTA * strength)
        row_indices, col_indices = _block_indices(len(bits), block_cols)
        coeffs = _dct2(encoded_blocks[row_indices, col_indices])
        midpoint = (coeffs[:, COEFF_A[0], COEFF_A[1]] + coeffs[:, COEFF_B[0], COEFF_B[1]]) * 0.5
        offsets = np.where(np.asarray(bits, dtype=bool), delta / 2.0, -delta / 2.0).astype(np.float32, copy=False)
        coeffs[:, COEFF_A[0], COEFF_A[1]] = midpoint + offsets
        coeffs[:, COEFF_B[0], COEFF_B[1]] = midpoint - offsets
        encoded_blocks[row_indices, col_indices] = _idct2(coeffs)

        return _ycbcr_to_rgb(encoded_y, cb_channel, cr_channel)

    def decode(self, image: ImageArray) -> ExtractResult:
        y_channel, _, _ = _rgb_to_ycbcr(image)
        raw_bits, confidence = self._extract_bits(y_channel)
        if len(raw_bits) < 24:
            return ExtractResult(
                found=False,
                engine=self.name,
                bits=tuple(raw_bits),
                confidence=0.0,
                error="insufficient_header",
            )

        header = bits_to_bytes(raw_bits[:24])
        payload_length = int.from_bytes(header[1:3], "big")
        total_bits = (1 + 2 + payload_length + 4) * 8
        if total_bits > len(raw_bits):
            return ExtractResult(
                found=False,
                engine=self.name,
                bits=tuple(raw_bits),
                confidence=confidence,
                error="insufficient_capacity",
            )

        decoded = decode_bitstream(raw_bits[:total_bits])
        return ExtractResult(
            found=decoded.valid,
            engine=self.name,
            bits=decoded.bits,
            payload=decoded.payload,
            message=decoded.message,
            confidence=confidence,
            error=decoded.error,
        )

    def _extract_bits(self, y_channel: FloatArray) -> tuple[list[int], float]:
        blocks = _block_view(y_channel)
        if blocks.size == 0:
            return [], 0.0

        coeffs = _dct2(blocks)
        diff = coeffs[..., COEFF_A[0], COEFF_A[1]] - coeffs[..., COEFF_B[0], COEFF_B[1]]
        bits = np.greater_equal(diff.reshape(-1), 0.0).astype(np.uint8, copy=False).tolist()
        average_margin = float(np.mean(np.abs(diff)))
        confidence = min(1.0, average_margin / BASE_DELTA) if average_margin else 0.0
        return bits, confidence


def _iter_blocks(height: int, width: int) -> Iterator[tuple[int, int]]:
    usable_height, usable_width = _usable_dimensions(height, width)
    for row in range(0, usable_height, BLOCK_SIZE):
        for col in range(0, usable_width, BLOCK_SIZE):
            yield row, col


def _block_view(channel: FloatArray) -> FloatArray:
    usable_height, usable_width = _usable_dimensions(channel.shape[0], channel.shape[1])
    if usable_height == 0 or usable_width == 0:
        return cast(FloatArray, np.empty((0, 0, BLOCK_SIZE, BLOCK_SIZE), dtype=channel.dtype))

    trimmed = channel[:usable_height, :usable_width]
    block_rows = usable_height // BLOCK_SIZE
    block_cols = usable_width // BLOCK_SIZE
    return cast(
        FloatArray,
        as_strided(
            trimmed,
            shape=(block_rows, block_cols, BLOCK_SIZE, BLOCK_SIZE),
            strides=(
                trimmed.strides[0] * BLOCK_SIZE,
                trimmed.strides[1] * BLOCK_SIZE,
                trimmed.strides[0],
                trimmed.strides[1],
            ),
            writeable=trimmed.flags.writeable,
        ),
    )


def _usable_dimensions(height: int, width: int) -> tuple[int, int]:
    return height - (height % BLOCK_SIZE), width - (width % BLOCK_SIZE)


def _block_indices(
    count: int, block_cols: int
) -> tuple[np.ndarray[tuple[int], np.dtype[np.intp]], np.ndarray[tuple[int], np.dtype[np.intp]]]:
    indices = np.arange(count, dtype=np.intp)
    return indices // block_cols, indices % block_cols


def _rgb_to_ycbcr(image: ImageArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    rgb = np.asarray(image, dtype=np.float32)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    y_channel = 0.299 * red + 0.587 * green + 0.114 * blue
    cb_channel = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr_channel = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return y_channel, cb_channel, cr_channel


def _ycbcr_to_rgb(y_channel: FloatArray, cb_channel: FloatArray, cr_channel: FloatArray) -> ImageArray:
    red = y_channel + 1.402 * (cr_channel - 128.0)
    green = y_channel - 0.344136 * (cb_channel - 128.0) - 0.714136 * (cr_channel - 128.0)
    blue = y_channel + 1.772 * (cb_channel - 128.0)
    rgb = np.stack([red, green, blue], axis=2)
    return cast(ImageArray, np.clip(rgb, 0, 255).astype(np.uint8))


def _dct2(block: FloatArray) -> FloatArray:
    matrix = _dct_matrix(BLOCK_SIZE)
    return cast(FloatArray, np.matmul(np.matmul(matrix, block), matrix.T).astype(np.float32, copy=False))


def _idct2(coefficients: FloatArray) -> FloatArray:
    matrix = _dct_matrix(BLOCK_SIZE)
    return cast(FloatArray, np.matmul(np.matmul(matrix.T, coefficients), matrix).astype(np.float32, copy=False))


@lru_cache(maxsize=1)
def _dct_matrix(size: int) -> FloatArray:
    rows = np.arange(size, dtype=np.float32)[:, None]
    cols = np.arange(size, dtype=np.float32)[None, :]
    alpha = np.full(size, np.sqrt(2.0 / size), dtype=np.float32)
    alpha[0] = np.sqrt(1.0 / size)
    matrix = alpha[:, None] * np.cos((np.pi * (2.0 * cols + 1.0) * rows) / (2.0 * size))
    matrix = matrix.astype(np.float32, copy=False)
    matrix.setflags(write=False)
    return cast(FloatArray, matrix)
