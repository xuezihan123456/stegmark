from __future__ import annotations

from collections.abc import Callable
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from stegmark.exceptions import InvalidInputError
from stegmark.types import ImageArray

AttackFn = Callable[..., ImageArray]


def _jpeg_attack(image: ImageArray, *, quality: int) -> ImageArray:
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        return np.asarray(reopened.convert("RGB"), dtype=np.uint8)


def _resize_attack(image: ImageArray, *, scale: float) -> ImageArray:
    source = Image.fromarray(image, mode="RGB")
    width, height = source.size
    resized = source.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
    restored = resized.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(restored, dtype=np.uint8)


def _crop_attack(image: ImageArray, *, keep_ratio: float) -> ImageArray:
    source = Image.fromarray(image, mode="RGB")
    width, height = source.size
    cropped_width = max(1, int(width * keep_ratio))
    cropped_height = max(1, int(height * keep_ratio))
    left = max(0, (width - cropped_width) // 2)
    top = max(0, (height - cropped_height) // 2)
    cropped = source.crop((left, top, left + cropped_width, top + cropped_height))
    restored = cropped.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(restored, dtype=np.uint8)


def _gaussian_blur_attack(image: ImageArray, *, radius: float) -> ImageArray:
    blurred = Image.fromarray(image, mode="RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.uint8)


def _gaussian_noise_attack(image: ImageArray, *, sigma: float, seed: int | None = None) -> ImageArray:
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) / 255.0
    noisy += rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32)
    return np.clip(noisy * 255.0, 0.0, 255.0).astype(np.uint8)


def _brightness_attack(image: ImageArray, *, factor: float) -> ImageArray:
    brightened = ImageEnhance.Brightness(Image.fromarray(image, mode="RGB")).enhance(factor)
    return np.asarray(brightened, dtype=np.uint8)


def _dropout_attack(image: ImageArray, *, probability: float, seed: int | None = 0) -> ImageArray:
    rng = np.random.default_rng(seed)
    mask = rng.random(image.shape[:2]) < probability
    attacked = image.copy()
    attacked[mask] = 0
    return attacked


AVAILABLE_ATTACKS: dict[str, AttackFn] = {
    "jpeg_q90": lambda image, seed=0: _jpeg_attack(image, quality=90),
    "jpeg_q75": lambda image, seed=0: _jpeg_attack(image, quality=75),
    "jpeg_q50": lambda image, seed=0: _jpeg_attack(image, quality=50),
    "resize_0.75": lambda image, seed=0: _resize_attack(image, scale=0.75),
    "resize_0.5": lambda image, seed=0: _resize_attack(image, scale=0.5),
    "crop_0.75": lambda image, seed=0: _crop_attack(image, keep_ratio=0.75),
    "crop_0.5": lambda image, seed=0: _crop_attack(image, keep_ratio=0.5),
    "gaussian_blur_1": lambda image, seed=0: _gaussian_blur_attack(image, radius=1.0),
    "gaussian_blur_2": lambda image, seed=0: _gaussian_blur_attack(image, radius=2.0),
    "gaussian_noise_0.03": lambda image, seed=0: _gaussian_noise_attack(image, sigma=0.03, seed=seed),
    "brightness_1.3": lambda image, seed=0: _brightness_attack(image, factor=1.3),
    "dropout_0.1": lambda image, seed=0: _dropout_attack(image, probability=0.1, seed=seed),
}


def apply_attack(image: ImageArray, attack_name: str, *, seed: int | None = 0) -> ImageArray:
    try:
        attack = AVAILABLE_ATTACKS[attack_name]
    except KeyError as exc:
        supported = ", ".join(sorted(AVAILABLE_ATTACKS))
        raise InvalidInputError(
            f"unknown attack: {attack_name}",
            hint=f"Use one of: {supported}",
        ) from exc
    return attack(image, seed=seed)
