from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from stegmark.exceptions import (
    ImageReadError,
    ImageTooSmallError,
    ImageWriteError,
    InvalidInputError,
    UnsupportedFormatError,
)
from stegmark.types import ImageArray, ImageMetadata, LoadedImage, PathLike

SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
MIN_IMAGE_SIZE = 128
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000


def load_image(path: PathLike) -> LoadedImage:
    source = Path(path)
    try:
        file_size = source.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise InvalidInputError(
                f"image file too large: {file_size} bytes",
                hint=f"Input images must be smaller than {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB.",
            )
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        with Image.open(source) as opened:
            metadata = ImageMetadata(
                format=opened.format,
                mode="RGB",
                exif=opened.info.get("exif"),
                icc_profile=opened.info.get("icc_profile"),
                extras={key: value for key, value in opened.info.items() if key not in {"exif", "icc_profile"}},
            )
            rgb = _to_rgb(opened)
            array = np.array(rgb, dtype=np.uint8, copy=True)
    except FileNotFoundError as exc:
        raise ImageReadError(f"image file does not exist: {source}", hint="Check the input path.") from exc
    except Image.DecompressionBombError as exc:
        raise InvalidInputError(
            f"image exceeds maximum allowed pixel count: {source}",
            hint=f"Input images must be smaller than {MAX_IMAGE_PIXELS} pixels.",
        ) from exc
    except OSError as exc:
        raise ImageReadError(
            f"failed to read image file: {source}",
            hint="Verify that the file is a valid PNG, JPEG, or WebP image.",
        ) from exc

    _ensure_minimum_size(array)
    return LoadedImage(array=array, metadata=metadata, source=source)


def save_image(
    path: PathLike,
    image: np.ndarray[Any, Any],
    *,
    metadata: ImageMetadata | None = None,
    format_name: str | None = None,
    quality: int = 95,
    allowed_root: PathLike | None = None,
) -> Path:
    target = Path(path)
    if allowed_root is not None:
        _ensure_within_root(target, Path(allowed_root))
    array = _normalize_array(image)
    output_format = _resolve_format(target, metadata, format_name=format_name)
    save_kwargs: dict[str, object] = {}
    if metadata and metadata.exif is not None:
        save_kwargs["exif"] = metadata.exif
    if metadata and metadata.icc_profile is not None:
        save_kwargs["icc_profile"] = metadata.icc_profile
    if output_format in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = quality
    if output_format == "JPEG":
        save_kwargs["subsampling"] = 0

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(array, mode="RGB").save(target, format=output_format, **save_kwargs)
    except OSError as exc:
        raise ImageWriteError(
            f"failed to write image file: {target}",
            hint="Check the output path and requested image format.",
        ) from exc
    return target


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if image.mode == "P":
        return _to_rgb(image.convert("RGBA"))
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _ensure_minimum_size(image: ImageArray) -> None:
    if image.shape[0] < MIN_IMAGE_SIZE or image.shape[1] < MIN_IMAGE_SIZE:
        raise ImageTooSmallError(
            f"image must be at least {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE} pixels",
            hint="Use a larger input image.",
        )


def _normalize_array(image: np.ndarray[Any, Any]) -> ImageArray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise InvalidInputError(
            "image array must have shape (height, width, 3)",
            hint="Pass an RGB image array with three channels.",
        )
    clipped = np.clip(array, 0, 255)
    return clipped.astype(np.uint8, copy=False)


def _resolve_format(path: Path, metadata: ImageMetadata | None, *, format_name: str | None = None) -> str:
    if format_name is not None:
        normalized = format_name.strip().upper()
        if normalized == "JPG":
            return "JPEG"
        if normalized in SUPPORTED_FORMATS:
            return normalized
        raise UnsupportedFormatError(
            f"unsupported requested output format: {format_name}",
            hint="Use png, jpeg/jpg, or webp.",
        )
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    if suffix == ".webp":
        return "WEBP"
    if metadata and metadata.format in SUPPORTED_FORMATS:
        return metadata.format
    raise UnsupportedFormatError(
        f"unsupported output format for path: {path}",
        hint="Use a .png, .jpg/.jpeg, or .webp output path.",
    )


def _ensure_within_root(target: Path, root: Path) -> None:
    resolved_target = target.resolve()
    resolved_root = root.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise InvalidInputError(
            f"output path escapes allowed directory: {resolved_target}",
            hint=f"Write outputs under {resolved_root}.",
        ) from exc
