from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from stegmark.core.image_io import load_image, save_image
from stegmark.exceptions import InvalidInputError


def test_load_image_rejects_files_larger_than_limit(
    tmp_path: Path,
    sample_image,
    monkeypatch,
) -> None:
    source = tmp_path / "too-large.png"
    save_image(source, sample_image)
    monkeypatch.setattr("stegmark.core.image_io.MAX_FILE_SIZE_BYTES", 1, raising=False)

    with pytest.raises(InvalidInputError, match="image file too large"):
        load_image(source)


def test_load_image_rejects_images_over_pixel_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "too-many-pixels.png"
    Image.new("RGB", (128, 128), (127, 127, 127)).save(source)
    monkeypatch.setattr("stegmark.core.image_io.MAX_IMAGE_PIXELS", 1, raising=False)

    with pytest.raises(InvalidInputError, match="maximum allowed pixel count"):
        load_image(source)


def test_save_image_rejects_output_outside_allowed_root(
    tmp_path: Path,
    sample_image,
) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    target = tmp_path / "outside.png"

    with pytest.raises(InvalidInputError, match="output path escapes allowed directory"):
        save_image(target, sample_image, allowed_root=allowed_root)
