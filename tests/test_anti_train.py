"""Tests for the anti-training perturbation layer (Glaze/Nightshade style)."""
from __future__ import annotations

import numpy as np
import pytest

from stegmark.core.anti_train import (
    CloakConfig,
    apply_cloak,
    estimate_cloak_strength,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rgb_image() -> np.ndarray:
    """128x128 synthetic RGB image with varied content."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test 1: output shape and dtype are preserved
# ---------------------------------------------------------------------------

def test_apply_cloak_preserves_shape_and_dtype(rgb_image: np.ndarray) -> None:
    cloaked = apply_cloak(rgb_image)
    assert cloaked.shape == rgb_image.shape
    assert cloaked.dtype == np.uint8


# ---------------------------------------------------------------------------
# Test 2: default mode PSNR > 30 dB
# ---------------------------------------------------------------------------

def test_apply_cloak_default_psnr_above_30db(rgb_image: np.ndarray) -> None:
    cloaked = apply_cloak(rgb_image, seed=0)
    metrics = estimate_cloak_strength(rgb_image, cloaked)
    assert metrics["psnr"] > 30.0, f"PSNR too low: {metrics['psnr']:.2f} dB"


# ---------------------------------------------------------------------------
# Test 3: estimate_cloak_strength returns required keys
# ---------------------------------------------------------------------------

def test_estimate_cloak_strength_keys(rgb_image: np.ndarray) -> None:
    cloaked = apply_cloak(rgb_image, seed=1)
    metrics = estimate_cloak_strength(rgb_image, cloaked)
    assert "psnr" in metrics
    assert "l_inf" in metrics
    assert "l2_norm" in metrics


# ---------------------------------------------------------------------------
# Test 4: L∞ ≤ config.epsilon * 255
# ---------------------------------------------------------------------------

def test_apply_cloak_linf_within_epsilon(rgb_image: np.ndarray) -> None:
    config = CloakConfig(epsilon=0.05)
    cloaked = apply_cloak(rgb_image, config=config, seed=2)
    metrics = estimate_cloak_strength(rgb_image, cloaked)
    assert metrics["l_inf"] <= config.epsilon * 255.0 + 1e-6, (
        f"L∞ {metrics['l_inf']:.3f} exceeds budget {config.epsilon * 255.0:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 5: same seed → same result
# ---------------------------------------------------------------------------

def test_apply_cloak_deterministic_with_seed(rgb_image: np.ndarray) -> None:
    cloaked_a = apply_cloak(rgb_image, seed=99)
    cloaked_b = apply_cloak(rgb_image, seed=99)
    np.testing.assert_array_equal(cloaked_a, cloaked_b)


# ---------------------------------------------------------------------------
# Test 6: different seeds → different results
# ---------------------------------------------------------------------------

def test_apply_cloak_different_seeds_differ(rgb_image: np.ndarray) -> None:
    cloaked_a = apply_cloak(rgb_image, seed=10)
    cloaked_b = apply_cloak(rgb_image, seed=20)
    assert not np.array_equal(cloaked_a, cloaked_b), (
        "Different seeds should produce different perturbations"
    )


# ---------------------------------------------------------------------------
# Test 7: cloak + native engine watermark round-trip
# ---------------------------------------------------------------------------

def test_cloak_then_native_watermark_roundtrip(rgb_image: np.ndarray) -> None:
    """Cloak the image, embed a watermark, then extract it successfully."""
    from stegmark.core.native import NativeEngine

    engine = NativeEngine()
    message = "hello"

    # Step 1: apply cloak
    cloaked = apply_cloak(rgb_image, seed=7)

    # Step 2: embed watermark into cloaked image
    watermarked = engine.encode(cloaked, message=message)

    # Step 3: extract watermark
    result = engine.decode(watermarked)

    assert result.found, f"Watermark not recovered after cloak; error={result.error}"
    assert result.message == message, (
        f"Recovered message '{result.message}' != expected '{message}'"
    )
