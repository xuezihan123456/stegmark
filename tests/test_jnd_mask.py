"""Tests for the Watson DCT JND perceptual mask."""
from __future__ import annotations

import numpy as np
import pytest

from stegmark.core.jnd_mask import (
    BLOCK_SIZE,
    compute_block_strength,
    compute_strength_map,
    watson_dct_jnd,
)


def test_watson_dct_jnd_returns_basic_table_for_zero_block():
    block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)

    result = watson_dct_jnd(block, luminance=127.5)

    assert result.shape == (BLOCK_SIZE, BLOCK_SIZE)
    # With zero AC coefficients contrast masking is inert; thresholds equal the
    # luminance-adjusted base table.
    assert result[0, 0] == pytest.approx(1.40, rel=1e-3)
    assert result[3, 3] == pytest.approx(3.77, rel=1e-3)


def test_watson_dct_jnd_rejects_wrong_shape():
    bad_block = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        watson_dct_jnd(bad_block, luminance=127.5)


def test_compute_block_strength_flat_region_low_scale():
    flat_block_dct = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    flat_block_dct[0, 0] = 1000.0  # high DC, no AC.

    scale = compute_block_strength(flat_block_dct, base_delta=12.0)

    assert 0.5 <= scale <= 3.0
    assert scale <= 1.0  # flat region is the baseline; textured regions exceed it


def test_compute_block_strength_textured_region_higher_scale():
    flat = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    flat[0, 0] = 1000.0
    textured = flat.copy()
    textured[3:6, 3:6] = 50.0

    flat_scale = compute_block_strength(flat, base_delta=12.0)
    textured_scale = compute_block_strength(textured, base_delta=12.0)

    assert textured_scale > flat_scale


def test_compute_block_strength_respects_bounds():
    extreme = np.full((BLOCK_SIZE, BLOCK_SIZE), 1e6, dtype=np.float32)

    scale = compute_block_strength(extreme, base_delta=12.0, min_scale=0.5, max_scale=3.0)

    assert 0.5 <= scale <= 3.0


def test_compute_strength_map_returns_block_grid():
    rng = np.random.default_rng(0)
    luminance = rng.uniform(0, 255, size=(40, 32)).astype(np.float32)

    strength_map = compute_strength_map(luminance, base_delta=12.0)

    assert strength_map.shape == (5, 4)
    assert strength_map.dtype == np.float32
    assert np.all(strength_map >= 0.5)
    assert np.all(strength_map <= 3.0)


def test_compute_strength_map_empty_for_tiny_image():
    tiny = np.zeros((4, 4), dtype=np.float32)

    strength_map = compute_strength_map(tiny, base_delta=12.0)

    assert strength_map.shape == (0, 0)
