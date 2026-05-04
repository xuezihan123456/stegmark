"""Regression tests for previously fixed bugs."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from PIL import Image

from stegmark.core.image_io import MAX_FILE_SIZE_BYTES, load_image
from stegmark.exceptions import InvalidInputError
from stegmark.service import _clamp_workers, _normalize_bits_hex
from stegmark.types import ImageMetadata


@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# H1: CLI strength=0.0 被 or 短路吞掉（已修复为 _coalesce_option）
# ---------------------------------------------------------------------------

def test_coalesce_option_preserves_zero_float() -> None:
    """_coalesce_option(0.0, default) 应返回 0.0，不被 or 短路。"""
    from stegmark.cli import _coalesce_option  # type: ignore[attr-defined]

    assert _coalesce_option(0.0, 1.0) == 0.0
    assert _coalesce_option(None, 1.0) == 1.0
    assert _coalesce_option(0, 4) == 0
    assert _coalesce_option(None, 4) == 4


# ---------------------------------------------------------------------------
# H2: bits_hex 带 0x 前缀时 bytes.fromhex 崩溃（已修复为 _normalize_bits_hex）
# ---------------------------------------------------------------------------

def test_normalize_bits_hex_strips_0x_prefix() -> None:
    assert _normalize_bits_hex("0xdeadbeef") == "deadbeef"
    assert _normalize_bits_hex("0XDEADBEEF") == "deadbeef"


def test_normalize_bits_hex_lowercases_plain_hex() -> None:
    assert _normalize_bits_hex("DEADBEEF") == "deadbeef"
    assert _normalize_bits_hex("deadbeef") == "deadbeef"


def test_normalize_bits_hex_none_returns_none() -> None:
    assert _normalize_bits_hex(None) is None


def test_normalize_bits_hex_empty_string() -> None:
    assert _normalize_bits_hex("") == ""


# ---------------------------------------------------------------------------
# SH2: workers 参数无上限（已修复为 _clamp_workers，MAX_WORKERS=32）
# ---------------------------------------------------------------------------

def test_clamp_workers_zero_becomes_one() -> None:
    assert _clamp_workers(0) == 1


def test_clamp_workers_negative_becomes_one() -> None:
    assert _clamp_workers(-5) == 1


def test_clamp_workers_normal_unchanged() -> None:
    assert _clamp_workers(4) == 4


def test_clamp_workers_cap_at_max() -> None:
    assert _clamp_workers(99999) == 32
    assert _clamp_workers(33) == 32


# ---------------------------------------------------------------------------
# SH1: 图像解压炸弹无防护（已修复为文件大小检查）
# ---------------------------------------------------------------------------

def test_load_image_rejects_file_exceeding_size_limit(tmp_dir: Path) -> None:
    """超过 MAX_FILE_SIZE_BYTES 的文件被 load_image 拒绝。"""
    large_file = tmp_dir / "large.bin"
    large_file.write_bytes(b"\x00" * (MAX_FILE_SIZE_BYTES + 1))
    with pytest.raises(InvalidInputError, match="too large"):
        load_image(large_file)


# ---------------------------------------------------------------------------
# H5: ImageMetadata.extras 破坏 frozen 语义（已修复为 MappingProxyType）
# ---------------------------------------------------------------------------

def test_image_metadata_extras_is_mapping_proxy() -> None:
    meta = ImageMetadata(extras={"key": "value"})
    assert isinstance(meta.extras, MappingProxyType)


def test_image_metadata_extras_is_immutable() -> None:
    meta = ImageMetadata(extras={"key": "value"})
    with pytest.raises(TypeError):
        meta.extras["new_key"] = "new_value"  # type: ignore[index]


def test_image_metadata_extras_empty_by_default() -> None:
    meta = ImageMetadata()
    assert isinstance(meta.extras, MappingProxyType)
    assert len(meta.extras) == 0
