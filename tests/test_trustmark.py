"""Integration tests for the TrustMark engine."""
from __future__ import annotations

import importlib
import importlib.util as ilu
import sys
from unittest.mock import patch

import numpy as np
import pytest

from stegmark.exceptions import EngineUnavailableError, InvalidInputError


def _make_image(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)


def test_trustmark_package_importable() -> None:
    """trustmark 包可以正常导入。"""
    pytest.importorskip("trustmark")
    from trustmark import TrustMark  # type: ignore[import-not-found]  # noqa: F401


def test_trustmark_registered_in_registry() -> None:
    """trustmark 已在引擎注册表中注册。"""
    from stegmark.core.registry import registered_engines

    assert "trustmark" in registered_engines()


def test_available_engines_reports_trustmark_true() -> None:
    """当 trustmark 已安装时 available_engines() 返回 trustmark: True。"""
    pytest.importorskip("trustmark")
    from stegmark.core.weights import available_engines

    info = available_engines()
    assert "trustmark" in info
    assert info["trustmark"] is True


def test_available_engines_reports_trustmark_false_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当 trustmark 未安装时 available_engines() 返回 trustmark: False。"""
    real_find_spec = ilu.find_spec

    def fake_find_spec(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "trustmark":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(ilu, "find_spec", fake_find_spec)
    monkeypatch.delitem(sys.modules, "trustmark", raising=False)

    import stegmark.core.weights as _w

    importlib.reload(_w)
    info = _w.available_engines()
    assert info["trustmark"] is False
    importlib.reload(_w)


def test_trustmark_engine_raises_when_package_missing() -> None:
    """当 trustmark 包不可用时，TrustMarkEngine() 抛 EngineUnavailableError。"""
    with patch.dict(sys.modules, {"trustmark": None}):
        import stegmark.core.trustmark as _tm

        importlib.reload(_tm)
        with pytest.raises(EngineUnavailableError):
            _tm.TrustMarkEngine()
    import stegmark.core.trustmark as _tm

    importlib.reload(_tm)


def test_trustmark_encode_rejects_payload_bits() -> None:
    """TrustMark 不支持 bits 模式，应抛 InvalidInputError。"""
    pytest.importorskip("trustmark")
    from stegmark.core.trustmark import TrustMarkEngine

    engine = TrustMarkEngine()
    image = _make_image()
    with pytest.raises(InvalidInputError, match="bits mode is not supported"):
        engine.encode(image, "ignored", payload_bits=[0, 1, 0])


def test_trustmark_encode_rejects_none_message() -> None:
    """TrustMark encode 要求文本消息，message=None 应抛 InvalidInputError。"""
    pytest.importorskip("trustmark")
    from stegmark.core.trustmark import TrustMarkEngine

    engine = TrustMarkEngine()
    image = _make_image()
    with pytest.raises(InvalidInputError, match="requires a text message"):
        engine.encode(image, None)


def test_trustmark_encode_rejects_message_exceeding_safe_ascii_capacity() -> None:
    """TrustMark 文本模式超过安全 ASCII 容量时应显式拒绝，而不是静默截断。"""
    pytest.importorskip("trustmark")
    from stegmark.core.trustmark import TrustMarkEngine

    engine = TrustMarkEngine()
    image = _make_image()
    with pytest.raises(InvalidInputError, match="supports at most"):
        engine.encode(image, "StegMark!")


def test_trustmark_encode_rejects_non_ascii_message() -> None:
    """TrustMark 文本模式只接受 ASCII，非 ASCII 文本应显式拒绝。"""
    pytest.importorskip("trustmark")
    from stegmark.core.trustmark import TrustMarkEngine

    engine = TrustMarkEngine()
    image = _make_image()
    with pytest.raises(InvalidInputError, match="only supports ASCII"):
        engine.encode(image, "你好")


@pytest.mark.slow
def test_trustmark_encode_decode_round_trip() -> None:
    """嵌入消息后提取，消息一致。"""
    pytest.importorskip("trustmark")
    from stegmark.core.trustmark import TrustMarkEngine

    engine = TrustMarkEngine()
    image = _make_image()
    message = "StegMark"
    watermarked = engine.encode(image, message)

    assert watermarked.shape == image.shape
    assert watermarked.dtype == np.uint8

    result = engine.decode(watermarked)
    assert result.engine == "trustmark"
    assert result.found is True
    assert result.message == message


@pytest.mark.slow
def test_trustmark_decode_plain_image_returns_result() -> None:
    """对未嵌入水印的图片 decode，返回合法 ExtractResult（不崩溃）。"""
    pytest.importorskip("trustmark")
    from stegmark.core.trustmark import TrustMarkEngine

    engine = TrustMarkEngine()
    result = engine.decode(_make_image(seed=99))
    assert result.engine == "trustmark"
    assert isinstance(result.found, bool)
