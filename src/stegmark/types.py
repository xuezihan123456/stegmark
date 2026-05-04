from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

PathLike = str | Path
ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class ImageMetadata:
    format: str | None = None
    mode: str | None = None
    exif: bytes | None = None
    icc_profile: bytes | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen_extras = {key: _freeze_extra_value(value) for key, value in dict(self.extras).items()}
        object.__setattr__(self, "extras", MappingProxyType(frozen_extras))

    def __repr__(self) -> str:
        return (
            "ImageMetadata("
            f"format={self.format!r}, "
            f"mode={self.mode!r}, "
            f"exif={_repr_bytes(self.exif)}, "
            f"icc_profile={_repr_bytes(self.icc_profile)}, "
            f"extras={_repr_mapping(self.extras)})"
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            self.__class__,
            (
                self.format,
                self.mode,
                self.exif,
                self.icc_profile,
                _thaw_extra_value(self.extras),
            ),
        )


@dataclass(frozen=True)
class LoadedImage:
    array: ImageArray
    metadata: ImageMetadata
    source: Path | None = None

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])


@dataclass(frozen=True)
class DecodedPayload:
    valid: bool
    bits: tuple[int, ...] = ()
    payload: bytes | None = None
    message: str | None = None
    version: int | None = None
    error: str | None = None

    def __bool__(self) -> bool:
        return self.valid


@dataclass(frozen=True)
class EmbedResult:
    image: ImageArray
    engine: str
    message: str
    bits: tuple[int, ...]
    metadata: ImageMetadata
    payload: bytes | None = None
    elapsed: float = 0.0
    output_path: Path | None = None
    psnr: float | None = None
    compare_report: Path | None = None
    diff_image: Path | None = None

    def __repr__(self) -> str:
        return (
            "EmbedResult("
            f"image=ndarray(shape={tuple(self.image.shape)}, dtype={self.image.dtype}), "
            f"engine={self.engine!r}, message={self.message!r}, bits_len={len(self.bits)}, "
            f"metadata={self.metadata!r}, payload={self.payload!r}, elapsed={self.elapsed:.3f}, "
            f"output_path={self.output_path!r}, psnr={self.psnr!r}, compare_report={self.compare_report!r}, "
            f"diff_image={self.diff_image!r})"
        )

@dataclass(frozen=True)
class ExtractResult:
    found: bool
    engine: str
    bits: tuple[int, ...] = ()
    payload: bytes | None = None
    message: str | None = None
    confidence: float = 0.0
    error: str | None = None

    def __bool__(self) -> bool:
        return self.found

    @property
    def payload_hex(self) -> str | None:
        return self.payload.hex() if self.payload is not None else None


@dataclass(frozen=True)
class VerifyResult:
    matched: bool
    engine: str
    expected: str | None = None
    actual: str | None = None
    confidence: float = 0.0

    def __bool__(self) -> bool:
        return self.matched


@dataclass(frozen=True)
class InfoResult:
    found: bool
    engine: str | None
    width: int
    height: int
    format: str | None
    confidence: float = 0.0


@dataclass(frozen=True)
class BatchItemResult:
    input_path: Path
    output_path: Path | None = None
    success: bool = False
    result: EmbedResult | ExtractResult | VerifyResult | InfoResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    items: tuple[BatchItemResult, ...]

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> int:
        return sum(int(item.success) for item in self.items)

    @property
    def failed(self) -> int:
        return self.total - self.succeeded


def _freeze_extra_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        frozen = np.array(value, copy=True)
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, Mapping):
        frozen_items = {key: _freeze_extra_value(inner) for key, inner in value.items()}
        return MappingProxyType(frozen_items)
    if isinstance(value, list | tuple):
        return tuple(_freeze_extra_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_extra_value(item) for item in value)
    return value


def _thaw_extra_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {key: _thaw_extra_value(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw_extra_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw_extra_value(item) for item in value)
    return value


def _repr_bytes(data: bytes | None) -> str:
    if data is None:
        return "None"
    if len(data) <= 24:
        return repr(data)
    return f"bytes(len={len(data)})"


def _repr_mapping(mapping: Mapping[str, Any]) -> str:
    if not mapping:
        return "{}"

    rendered_items: list[str] = []
    for index, (key, value) in enumerate(mapping.items()):
        if index == 4:
            rendered_items.append("...")
            break
        rendered_items.append(f"{key!r}: {_repr_extra_value(value)}")
    return "{" + ", ".join(rendered_items) + "}"


def _repr_extra_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, Mapping):
        return _repr_mapping(value)

    rendered = repr(value)
    if len(rendered) <= 80:
        return rendered
    return f"{rendered[:77]}..."
