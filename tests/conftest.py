from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def sample_image() -> np.ndarray:
    axis = np.linspace(0, 255, 128, dtype=np.uint8)
    grid_x, grid_y = np.meshgrid(axis, axis)
    blue = np.full_like(grid_x, 96)
    return np.stack([grid_x, grid_y, blue], axis=2)


@pytest.fixture()
def sample_image_path(tmp_path: Path, sample_image: np.ndarray) -> Path:
    target = tmp_path / "sample.png"
    Image.fromarray(sample_image, mode="RGB").save(target)
    return target
