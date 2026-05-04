"""Edge case and boundary tests."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stegmark.core.image_io import MIN_IMAGE_SIZE, load_image, save_image
from stegmark.exceptions import ImageTooSmallError, InvalidInputError
from stegmark.service import _collect_image_files


@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# 文件收集边界
# ---------------------------------------------------------------------------

def test_collect_image_files_empty_dir(tmp_dir: Path) -> None:
    """空目录返回空列表。"""
    result = _collect_image_files(tmp_dir, recursive=False)
    assert result == []


def test_collect_image_files_ignores_non_image(tmp_dir: Path) -> None:
    (tmp_dir / "readme.txt").write_text("hello", encoding="utf-8")
    (tmp_dir / "data.json").write_text("{}", encoding="utf-8")
    result = _collect_image_files(tmp_dir, recursive=False)
    assert result == []


def test_collect_image_files_finds_png(tmp_dir: Path) -> None:
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    png_path = tmp_dir / "test.png"
    img.save(str(png_path))
    result = _collect_image_files(tmp_dir, recursive=False)
    assert png_path in result


def test_collect_image_files_skips_symlinks(tmp_dir: Path) -> None:
    """符号链接应被跳过（安全防护）。"""
    real_img = tmp_dir / "real.png"
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    img.save(str(real_img))
    link = tmp_dir / "link.png"
    try:
        link.symlink_to(real_img)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    result = _collect_image_files(tmp_dir, recursive=False)
    assert link not in result
    assert real_img in result


# ---------------------------------------------------------------------------
# 图像尺寸边界
# ---------------------------------------------------------------------------

def test_load_image_rejects_too_small(tmp_dir: Path) -> None:
    """小于 MIN_IMAGE_SIZE 的图片被拒绝。"""
    small = tmp_dir / "small.png"
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    img.save(str(small))
    with pytest.raises(ImageTooSmallError):
        load_image(small)


def test_load_image_accepts_minimum_size(tmp_dir: Path) -> None:
    """恰好等于 MIN_IMAGE_SIZE 的图片可以加载。"""
    ok = tmp_dir / "ok.png"
    img = Image.fromarray(
        np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8), mode="RGB"
    )
    img.save(str(ok))
    loaded = load_image(ok)
    assert loaded.width == MIN_IMAGE_SIZE
    assert loaded.height == MIN_IMAGE_SIZE


# ---------------------------------------------------------------------------
# 路径遍历防护
# ---------------------------------------------------------------------------

def test_save_image_rejects_path_outside_root(tmp_dir: Path) -> None:
    """输出路径逃逸出允许根目录时抛 InvalidInputError。"""
    array = np.zeros((128, 128, 3), dtype=np.uint8)
    allowed = tmp_dir / "output"
    allowed.mkdir()
    escape = tmp_dir / ".." / "escaped.png"
    with pytest.raises(InvalidInputError, match="escapes allowed directory"):
        save_image(escape, array, allowed_root=allowed)


# ---------------------------------------------------------------------------
# Native engine 边界
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="empty-string encode crashes on Windows")
def test_native_engine_encode_empty_bits_no_crash() -> None:
    """嵌入空消息时 native engine 不崩溃，直接返回原图。"""
    from stegmark.core.native import NativeEngine

    engine = NativeEngine()
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = engine.encode(image, "")
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_native_engine_decode_returns_result() -> None:
    """native engine decode 返回合法 ExtractResult。"""
    from stegmark.core.native import NativeEngine

    engine = NativeEngine()
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = engine.decode(image)
    assert result.engine == "native"
    assert isinstance(result.found, bool)


# ---------------------------------------------------------------------------
# StegMark 公共 API 基础
# ---------------------------------------------------------------------------

def test_stegmark_class_instantiates() -> None:
    """StegMark 类可以正常实例化。"""
    import stegmark

    sm = stegmark.StegMark()
    assert sm is not None


def test_stegmark_embed_extract_round_trip(tmp_dir: Path) -> None:
    """公共 API embed → extract 往返（native engine）。"""
    import stegmark

    src = tmp_dir / "src.png"
    out = tmp_dir / "out.png"
    img = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), mode="RGB"
    )
    img.save(str(src))

    stegmark.embed(src, "hello", engine="native", output=out, overwrite=True)
    assert out.exists()

    extract_result = stegmark.extract(out, engine="native")
    assert extract_result.found is True
    assert extract_result.message == "hello"
