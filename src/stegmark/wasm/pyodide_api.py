from __future__ import annotations

"""Pyodide 浏览器端薄封装 — Browser-optimised thin wrapper for Pyodide.

在 Pyodide 运行时中无需 Pillow，直接操作原始 RGBA bytes 与 numpy ndarray。
No Pillow required; operates on raw RGBA bytes and numpy ndarrays directly.
"""

import numpy as np

import stegmark
from stegmark.core.codec import encode_payload
from stegmark.core.native import NativeEngine

_ENGINE = NativeEngine()

_MIN_SIDE = 128  # 最小边长（像素） / minimum side length (pixels)
_MAX_IMAGE_PIXELS = 50_000_000


def embed_rgba(
    rgba_bytes: bytes,
    width: int,
    height: int,
    payload: bytes,
    strength: float = 1.0,
) -> bytes:
    """将水印嵌入 RGBA 字节流并返回新的 RGBA 字节流。

    Embed a watermark into an RGBA byte buffer and return the watermarked RGBA bytes.

    Parameters
    ----------
    rgba_bytes:
        原始 RGBA 像素字节，长度必须等于 width * height * 4。
        Raw RGBA pixel bytes; length must equal width * height * 4.
    width:
        图像宽度（像素）。Image width in pixels.
    height:
        图像高度（像素）。Image height in pixels.
    payload:
        要嵌入的原始字节载荷。Raw byte payload to embed.
    strength:
        水印强度因子（默认 1.0）。Watermark strength factor (default 1.0).

    Returns
    -------
    bytes
        含水印的 RGBA 字节流（alpha 通道不变）。
        Watermarked RGBA bytes with alpha channel preserved.

    Raises
    ------
    ValueError
        当 rgba_bytes 长度不符合 width * height * 4 时。
        When rgba_bytes length does not equal width * height * 4.
    ValueError
        当图像尺寸小于 _MIN_SIDE × _MIN_SIDE 时。
        When image dimensions are smaller than _MIN_SIDE × _MIN_SIDE.
    """
    expected = width * height * 4
    if len(rgba_bytes) != expected:
        raise ValueError(
            f"rgba_bytes length {len(rgba_bytes)} does not match width*height*4={expected}"
        )
    if width < _MIN_SIDE or height < _MIN_SIDE:
        raise ValueError(
            f"image too small: {width}×{height}, minimum is {_MIN_SIDE}×{_MIN_SIDE}"
        )

    # 解包 RGBA → 分离 RGB 和 alpha
    rgba_array = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(height, width, 4)
    rgb_array = rgba_array[:, :, :3].copy()
    alpha_channel = rgba_array[:, :, 3].copy()

    # 调用 NativeEngine.encode（payload_bits 路径）
    bits = encode_payload(payload)
    watermarked_rgb = _ENGINE.encode(rgb_array, payload_bits=bits, strength=strength)

    # 重新合并 alpha 通道
    watermarked_rgba = np.dstack([watermarked_rgb, alpha_channel])
    return bytes(watermarked_rgba.tobytes())


def extract_rgba(
    rgba_bytes: bytes,
    width: int,
    height: int,
    num_bytes: int,
) -> bytes | None:
    """从 RGBA 字节流中提取水印载荷。

    Extract the watermark payload from an RGBA byte buffer.

    Parameters
    ----------
    rgba_bytes:
        原始 RGBA 像素字节。Raw RGBA pixel bytes.
    width:
        图像宽度（像素）。Image width in pixels.
    height:
        图像高度（像素）。Image height in pixels.
    num_bytes:
        期望提取的载荷字节数（目前仅作参考，实际由帧头决定）。
        Expected payload byte count (informational; frame header determines actual length).

    Returns
    -------
    bytes | None
        提取到的载荷字节，失败时返回 None。
        Extracted payload bytes, or None on failure.
    """
    expected = width * height * 4
    if len(rgba_bytes) != expected:
        raise ValueError(
            f"rgba_bytes length {len(rgba_bytes)} does not match width*height*4={expected}"
        )

    rgba_array = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(height, width, 4)
    rgb_array = rgba_array[:, :, :3].copy()

    result = _ENGINE.decode(rgb_array)
    if not result.found or result.payload is None:
        return None
    return result.payload


def capabilities() -> dict:
    """返回运行时能力描述字典。

    Return a dict describing the runtime capabilities of this Pyodide API.
    """
    return {
        "version": stegmark.__version__,
        "engines": ["native"],
        "max_image_pixels": _MAX_IMAGE_PIXELS,
    }
