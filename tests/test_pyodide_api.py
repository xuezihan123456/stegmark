"""纯 Python 测试：stegmark.wasm.pyodide_api（无需 Pyodide 运行时）。

Pure-Python tests for stegmark.wasm.pyodide_api — no Pyodide runtime needed.
"""
from __future__ import annotations

import numpy as np
import pytest

import stegmark
from stegmark.wasm.pyodide_api import capabilities, embed_rgba, extract_rgba

# ---------------------------------------------------------------------------
# 辅助函数 / helpers
# ---------------------------------------------------------------------------

def _make_rgba(width: int, height: int, seed: int = 0) -> bytes:
    """生成确定性随机 RGBA 字节流。Generate deterministic random RGBA bytes."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (height, width, 4), dtype=np.uint8)
    # alpha 全不透明
    arr[:, :, 3] = 255
    return bytes(arr.tobytes())


# ---------------------------------------------------------------------------
# 测试 1: capabilities() 结构与版本一致性
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_required_keys_present(self) -> None:
        """capabilities() 必须包含 version、engines、max_image_pixels 三个键。"""
        caps = capabilities()
        assert "version" in caps
        assert "engines" in caps
        assert "max_image_pixels" in caps

    def test_version_matches_package(self) -> None:
        """capabilities()['version'] 必须与 stegmark.__version__ 一致。"""
        caps = capabilities()
        assert caps["version"] == stegmark.__version__

    def test_engines_contains_native(self) -> None:
        """engines 列表必须包含 'native'。"""
        caps = capabilities()
        assert "native" in caps["engines"]

    def test_max_image_pixels_positive(self) -> None:
        """max_image_pixels 必须为正整数。"""
        caps = capabilities()
        assert isinstance(caps["max_image_pixels"], int)
        assert caps["max_image_pixels"] > 0


# ---------------------------------------------------------------------------
# 测试 2: embed_rgba round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_embed_extract_roundtrip(self) -> None:
        """embed_rgba → extract_rgba 应还原原始 payload。"""
        width, height = 256, 256
        payload = b"hello pyodide"
        rgba = _make_rgba(width, height, seed=42)

        watermarked = embed_rgba(rgba, width, height, payload)
        recovered = extract_rgba(watermarked, width, height, len(payload))

        assert recovered == payload

    def test_alpha_channel_preserved(self) -> None:
        """嵌入水印后 alpha 通道必须保持不变。"""
        width, height = 256, 256
        rgba = _make_rgba(width, height, seed=7)
        original_alpha = np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, 3].copy()

        watermarked = embed_rgba(rgba, width, height, b"alpha-test")
        result_alpha = np.frombuffer(bytes(watermarked), dtype=np.uint8).reshape(height, width, 4)[:, :, 3]

        np.testing.assert_array_equal(original_alpha, result_alpha)

    def test_output_length_equals_input(self) -> None:
        """输出 RGBA 字节数必须与输入相同。"""
        width, height = 200, 200
        rgba = _make_rgba(width, height, seed=1)
        watermarked = embed_rgba(rgba, width, height, b"length-check")
        assert len(watermarked) == len(rgba)


# ---------------------------------------------------------------------------
# 测试 3: embed_rgba 尺寸校验 — 字节长度不匹配
# ---------------------------------------------------------------------------

class TestSizeValidation:
    def test_wrong_buffer_length_raises_value_error(self) -> None:
        """当 len(rgba_bytes) ≠ width*height*4 时应抛出 ValueError。"""
        width, height = 256, 256
        bad_bytes = b"\x00" * (width * height * 4 - 1)  # 少一字节
        with pytest.raises(ValueError, match="does not match"):
            embed_rgba(bad_bytes, width, height, b"test")

    def test_extract_wrong_buffer_length_raises_value_error(self) -> None:
        """extract_rgba 对错误长度同样应抛出 ValueError。"""
        width, height = 256, 256
        bad_bytes = b"\x00" * (width * height * 4 + 4)  # 多四字节
        with pytest.raises(ValueError, match="does not match"):
            extract_rgba(bad_bytes, width, height, 5)


# ---------------------------------------------------------------------------
# 测试 4: embed_rgba 过小图像抛错
# ---------------------------------------------------------------------------

class TestSmallImageRejected:
    def test_image_below_min_side_raises(self) -> None:
        """128×128 以下图像嵌入时应抛出 ValueError。"""
        width, height = 64, 64
        rgba = _make_rgba(width, height, seed=99)
        with pytest.raises(ValueError, match="too small"):
            embed_rgba(rgba, width, height, b"tiny")

    def test_exactly_min_side_is_accepted(self) -> None:
        """恰好 128×128 应被接受（不抛出异常）。"""
        width, height = 128, 128
        rgba = _make_rgba(width, height, seed=3)
        result = embed_rgba(rgba, width, height, b"ok")
        assert len(result) == width * height * 4


# ---------------------------------------------------------------------------
# 测试 5: extract_rgba 在无水印图上不崩溃
# ---------------------------------------------------------------------------

class TestExtractUnmarked:
    def test_extract_unmarked_returns_none_or_bytes(self) -> None:
        """无水印图像 extract_rgba 应返回 None 或 bytes，绝不崩溃。"""
        width, height = 256, 256
        rgba = _make_rgba(width, height, seed=123)
        result = extract_rgba(rgba, width, height, 8)
        assert result is None or isinstance(result, bytes)

    def test_extract_uniform_image_does_not_crash(self) -> None:
        """纯色图（全黑 RGBA）不应引发异常。"""
        width, height = 256, 256
        rgba = bytes(width * height * 4)  # 全零
        result = extract_rgba(rgba, width, height, 4)
        assert result is None or isinstance(result, bytes)
