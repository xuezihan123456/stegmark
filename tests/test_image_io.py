from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from stegmark.core.image_io import load_image, save_image


def test_save_then_load_png(tmp_path: Path) -> None:
    sample_image = np.full((128, 128, 3), 127, dtype=np.uint8)
    target = tmp_path / "sample.png"

    save_image(target, sample_image)
    loaded = load_image(target)

    assert loaded.array.shape == sample_image.shape
    assert loaded.array.dtype == np.uint8
    assert loaded.metadata.format == "PNG"


def test_rgba_input_is_composited_to_rgb(tmp_path: Path) -> None:
    rgba = Image.new("RGBA", (128, 128), (255, 0, 0, 0))
    source = tmp_path / "transparent.png"
    rgba.save(source)

    loaded = load_image(source)

    assert loaded.array.shape == (128, 128, 3)
    assert loaded.array[0, 0].tolist() == [255, 255, 255]
