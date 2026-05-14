"""Glaze/Nightshade-style anti-training perturbation for image protection.

Adds structured perturbations in the DCT high-frequency band (radius 16-48)
to interfere with diffusion model feature extraction while preserving the
mid-frequency band used by the native watermark engine.

Pure-numpy implementation; torch mode is a placeholder with graceful fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# DCT radius bands: native engine uses mid-freq (coeffs (2,3)/(3,2) ≈ radius 3-4).
# We perturb high-freq ring [16, 48) to stay clear of watermark region.
_HF_RADIUS_MIN = 16
_HF_RADIUS_MAX = 48


def is_torch_available() -> bool:
    """Return True if torch can be imported."""
    try:
        import torch as _torch  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class CloakConfig:
    """Configuration for the anti-training cloak."""

    epsilon: float = 0.05
    """L-infinity perturbation budget as fraction of 255 (default 0.05 → ~12.75 DN)."""

    steps: int = 5
    """Number of perturbation refinement steps."""

    mode: Literal["frequency", "vgg", "clip", "auto"] = "frequency"
    """Perturbation mode. 'auto' selects 'vgg'/'clip' if torch available, else 'frequency'."""


def _dct2_full(channel: np.ndarray) -> np.ndarray:
    """2-D DCT-II via separable 1-D transforms (scipy-free, numpy-only)."""
    h, w = channel.shape
    # Build DCT-II matrix for each dimension
    def _dct_matrix(n: int) -> np.ndarray:
        k = np.arange(n, dtype=np.float64)
        ns = np.arange(n, dtype=np.float64)
        # D[k, n] = cos(pi * k * (2n+1) / (2N))
        mat = np.cos(np.pi * k[:, None] * (2.0 * ns[None, :] + 1.0) / (2.0 * n))
        mat[0, :] *= 1.0 / np.sqrt(n)
        mat[1:, :] *= np.sqrt(2.0 / n)
        return mat

    dh = _dct_matrix(h)
    dw = _dct_matrix(w)
    return dh @ channel.astype(np.float64) @ dw.T


def _idct2_full(coeffs: np.ndarray) -> np.ndarray:
    """2-D IDCT-II (inverse of _dct2_full)."""
    n_rows, n_cols = coeffs.shape

    def _dct_matrix(n: int) -> np.ndarray:
        k = np.arange(n, dtype=np.float64)
        ns = np.arange(n, dtype=np.float64)
        mat = np.cos(np.pi * k[:, None] * (2.0 * ns[None, :] + 1.0) / (2.0 * n))
        mat[0, :] *= 1.0 / np.sqrt(n)
        mat[1:, :] *= np.sqrt(2.0 / n)
        return mat

    dh = _dct_matrix(n_rows)
    dw = _dct_matrix(n_cols)
    # IDCT = D^T * coeffs * D
    return dh.T @ coeffs.astype(np.float64) @ dw


def _hf_mask(h: int, w: int) -> np.ndarray:
    """Boolean mask of DCT coefficients in high-frequency annulus [r_min, r_max)."""
    ki = np.arange(h, dtype=np.float64)
    kj = np.arange(w, dtype=np.float64)
    ki_grid, kj_grid = np.meshgrid(ki, kj, indexing="ij")
    radius = np.sqrt(ki_grid**2 + kj_grid**2)
    return (radius >= _HF_RADIUS_MIN) & (radius < _HF_RADIUS_MAX)


def _apply_frequency_cloak(
    image: np.ndarray,
    config: CloakConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Frequency-domain cloak: perturb DCT high-frequency band per channel."""
    img_f = image.astype(np.float64)
    epsilon_dn = config.epsilon * 255.0
    result = np.empty_like(img_f)

    for ch in range(3):
        channel = img_f[:, :, ch]
        h, w = channel.shape
        mask = _hf_mask(h, w)
        n_coeffs = int(mask.sum())

        coeffs = _dct2_full(channel)

        # Iterative structured perturbation in high-frequency band.
        # Each step adds a small random signed pattern; total is clipped to epsilon.
        perturbation = np.zeros_like(coeffs)
        step_scale = epsilon_dn / max(config.steps, 1)

        for _ in range(config.steps):
            # Random pattern restricted to high-freq mask
            noise = rng.standard_normal(n_coeffs) * step_scale
            delta = np.zeros_like(coeffs)
            delta[mask] = noise
            perturbation += delta

        # Clip perturbation in spatial domain via reconstruct-clip-reconstruct cycle
        perturbed_coeffs = coeffs + perturbation
        perturbed_spatial = _idct2_full(perturbed_coeffs)
        # Clip spatial perturbation to epsilon_dn
        spatial_delta = np.clip(perturbed_spatial - channel, -epsilon_dn, epsilon_dn)
        cloaked_channel = channel + spatial_delta
        result[:, :, ch] = cloaked_channel

    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def apply_cloak(
    image: np.ndarray,
    config: CloakConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Apply anti-training perturbation to an RGB image.

    Parameters
    ----------
    image:
        Input uint8 RGB image of shape (H, W, 3).
    config:
        Cloak configuration; defaults to ``CloakConfig()``.
    seed:
        Optional integer seed for reproducible perturbations.

    Returns
    -------
    np.ndarray
        Cloaked uint8 RGB image of the same shape as ``image``.
    """
    if config is None:
        config = CloakConfig()

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3) RGB, got shape {image.shape}")

    rng = np.random.default_rng(seed)

    effective_mode = config.mode
    if effective_mode == "auto":
        effective_mode = "vgg" if is_torch_available() else "frequency"

    if effective_mode == "frequency":
        return _apply_frequency_cloak(image, config, rng)

    # torch modes: PGD placeholder — fall back to frequency
    if not is_torch_available():
        return _apply_frequency_cloak(image, config, rng)

    # Placeholder for VGG/CLIP PGD attack (torch available but not yet implemented).
    # Falls back to frequency mode until torch implementation is added.
    return _apply_frequency_cloak(image, config, rng)


def estimate_cloak_strength(
    original: np.ndarray,
    cloaked: np.ndarray,
) -> dict[str, float]:
    """Compute perceptual and distortion metrics between original and cloaked images.

    Parameters
    ----------
    original:
        Original uint8 RGB image.
    cloaked:
        Cloaked uint8 RGB image of the same shape.

    Returns
    -------
    dict with keys:
        ``psnr``   – Peak Signal-to-Noise Ratio in dB (float, inf if identical).
        ``l_inf``  – Maximum absolute pixel difference (L-infinity norm).
        ``l2_norm`` – Root-mean-square pixel difference (L2 norm).
    """
    orig_f = original.astype(np.float64)
    cloak_f = cloaked.astype(np.float64)
    diff = cloak_f - orig_f

    l_inf = float(np.max(np.abs(diff)))
    mse = float(np.mean(diff**2))
    l2_norm = float(np.sqrt(mse))

    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10(255.0**2 / mse))

    return {"psnr": psnr, "l_inf": l_inf, "l2_norm": l2_norm}


__all__ = [
    "CloakConfig",
    "apply_cloak",
    "estimate_cloak_strength",
    "is_torch_available",
]
