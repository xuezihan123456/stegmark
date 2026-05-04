"""Public package API for StegMark."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from stegmark.core.registry import get_engine, registered_engines
from stegmark.core.weights import available_engines, device_info, is_available
from stegmark.logging_utils import configure_logging, logger


def _install_benchmark_stub() -> None:
    if "stegmark.evaluation.benchmark" in sys.modules:
        return
    if importlib.util.find_spec("stegmark.evaluation.benchmark") is not None:
        return

    def _missing_benchmark(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ModuleNotFoundError("stegmark.evaluation.benchmark is not available in this worktree")

    stub = types.ModuleType("stegmark.evaluation.benchmark")
    for symbol in (
        "BenchmarkAttackResult",
        "BenchmarkComparisonResult",
        "BenchmarkEngineResult",
        "BenchmarkResult",
        "BenchmarkSummary",
    ):
        setattr(stub, symbol, type(symbol, (), {}))
    stub.benchmark_compare_engines = _missing_benchmark
    stub.benchmark_file = _missing_benchmark
    sys.modules[stub.__name__] = stub


_install_benchmark_stub()

from stegmark.service import (
    benchmark_compare_service,
    benchmark_file_service,
    embed_directory as embed_directory_service,
)
from stegmark.service import (
    embed_file,
    extract_file,
    info_file,
    verify_file,
)
from stegmark.service import (
    extract_directory as extract_directory_service,
)
from stegmark.service import (
    info_directory as info_directory_service,
)
from stegmark.service import (
    verify_directory as verify_directory_service,
)
from stegmark.types import (
    BatchResult,
    EmbedResult,
    ExtractResult,
    InfoResult,
    PathLike,
    VerifyResult,
)

if TYPE_CHECKING:
    from stegmark.evaluation.types import BenchmarkComparisonResult, BenchmarkResult
else:
    BenchmarkComparisonResult = Any
    BenchmarkResult = Any

__all__ = [
    "__version__",
    "StegMark",
    "benchmark",
    "benchmark_compare",
    "available_engines",
    "device_info",
    "embed",
    "embed_directory",
    "extract",
    "extract_directory",
    "get_engine",
    "info",
    "info_directory",
    "is_available",
    "logger",
    "configure_logging",
    "registered_engines",
    "verify",
    "verify_directory",
    "BenchmarkResult",
    "BenchmarkComparisonResult",
    "BatchResult",
    "EmbedResult",
    "ExtractResult",
    "InfoResult",
    "VerifyResult",
]

__version__ = "0.3.0a1"


def embed(
    input_path: PathLike,
    message: str | None = None,
    *,
    bits: str | None = None,
    output: PathLike | None = None,
    engine: str = "auto",
    strength: float = 1.0,
    output_format: str | None = None,
    quality: int = 95,
    overwrite: bool = False,
    compare: bool = False,
) -> EmbedResult:
    target = Path(output) if output is not None else _default_output_path(input_path)
    return embed_file(
        input_path,
        target,
        message=message,
        bits_hex=bits,
        engine=engine,
        strength=strength,
        output_format=output_format,
        quality=quality,
        overwrite=overwrite,
        compare=compare,
    )


def embed_directory(
    input_dir: PathLike,
    message: str | None = None,
    *,
    bits: str | None = None,
    engine: str = "auto",
    strength: float = 1.0,
    recursive: bool = False,
    output_dir: PathLike | None = None,
    workers: int = 1,
    overwrite: bool = False,
    output_format: str | None = None,
    quality: int = 95,
    progress: Callable[[int, int, Path], None] | None = None,
    allowed_output_root: PathLike | None = None,
) -> BatchResult:
    return embed_directory_service(
        input_dir,
        message=message,
        bits_hex=bits,
        engine=engine,
        strength=strength,
        recursive=recursive,
        output_dir=output_dir,
        workers=workers,
        overwrite=overwrite,
        output_format=output_format,
        quality=quality,
        progress=progress,
        allowed_output_root=allowed_output_root,
    )


def benchmark(
    input_path: PathLike,
    message: str,
    *,
    engine: str = "native",
    attacks: list[str] | tuple[str, ...] | None = None,
    output_dir: PathLike | None = None,
    report_format: str = "json",
) -> BenchmarkResult:
    from stegmark import service as stegmark_service

    return stegmark_service.benchmark_file_service(
        input_path,
        message=message,
        engine=engine,
        attacks=attacks,
        output_dir=output_dir,
        report_format=report_format,
    )


def benchmark_compare(
    input_path: PathLike,
    message: str,
    *,
    engines: list[str] | tuple[str, ...],
    attacks: list[str] | tuple[str, ...] | None = None,
    output_dir: PathLike | None = None,
    report_format: str = "json",
) -> BenchmarkComparisonResult:
    from stegmark import service as stegmark_service

    return stegmark_service.benchmark_compare_service(
        input_path,
        message=message,
        engines=engines,
        attacks=attacks,
        output_dir=output_dir,
        report_format=report_format,
    )


def extract(input_path: PathLike, *, engine: str = "auto") -> ExtractResult:
    return extract_file(input_path, engine=engine)


def extract_directory(
    input_dir: PathLike,
    *,
    engine: str = "auto",
    recursive: bool = False,
    workers: int = 1,
    progress: Callable[[int, int, Path], None] | None = None,
) -> BatchResult:
    return extract_directory_service(
        input_dir,
        engine=engine,
        recursive=recursive,
        workers=workers,
        progress=progress,
    )


def verify_directory(
    input_dir: PathLike,
    expected: str,
    *,
    engine: str = "auto",
    recursive: bool = False,
    workers: int = 1,
    progress: Callable[[int, int, Path], None] | None = None,
) -> BatchResult:
    return verify_directory_service(
        input_dir,
        expected=expected,
        engine=engine,
        recursive=recursive,
        workers=workers,
        progress=progress,
    )


def verify(input_path: PathLike, expected: str, *, engine: str = "auto") -> VerifyResult:
    return verify_file(input_path, expected, engine=engine)


def info_directory(
    input_dir: PathLike,
    *,
    engine: str = "auto",
    recursive: bool = False,
    workers: int = 1,
    progress: Callable[[int, int, Path], None] | None = None,
) -> BatchResult:
    return info_directory_service(
        input_dir,
        engine=engine,
        recursive=recursive,
        workers=workers,
        progress=progress,
    )


def info(input_path: PathLike, *, engine: str = "auto") -> InfoResult:
    return info_file(input_path, engine=engine)


class StegMark:
    def __init__(self, *, engine: str = "auto", strength: float = 1.0) -> None:
        self.engine = engine
        self.strength = strength

    def __repr__(self) -> str:
        return f"StegMark(engine={self.engine!r}, strength={self.strength!r})"

    def __enter__(self) -> StegMark:
        return self

    def __exit__(self, exc_type: object | None, exc: BaseException | None, tb: object | None) -> None:
        del exc_type, exc, tb

    def embed(
        self,
        input_path: PathLike,
        message: str,
        *,
        output: PathLike | None = None,
        compare: bool = False,
    ) -> EmbedResult:
        return embed(
            input_path,
            message,
            output=output,
            compare=compare,
            engine=self.engine,
            strength=self.strength,
        )

    def embed_bits(
        self,
        input_path: PathLike,
        bits: str,
        *,
        output: PathLike | None = None,
        compare: bool = False,
    ) -> EmbedResult:
        return embed(
            input_path,
            bits=bits,
            output=output,
            compare=compare,
            engine=self.engine,
            strength=self.strength,
        )

    def embed_directory(
        self,
        input_dir: PathLike,
        message: str | None = None,
        *,
        bits: str | None = None,
        recursive: bool = False,
        output_dir: PathLike | None = None,
        workers: int = 1,
        overwrite: bool = False,
        output_format: str | None = None,
        quality: int = 95,
        progress: Callable[[int, int, Path], None] | None = None,
        allowed_output_root: PathLike | None = None,
    ) -> BatchResult:
        return embed_directory(
            input_dir,
            message,
            bits=bits,
            engine=self.engine,
            strength=self.strength,
            recursive=recursive,
            output_dir=output_dir,
            workers=workers,
            overwrite=overwrite,
            output_format=output_format,
            quality=quality,
            progress=progress,
            allowed_output_root=allowed_output_root,
        )

    def benchmark(
        self,
        input_path: PathLike,
        message: str,
        *,
        attacks: list[str] | tuple[str, ...] | None = None,
        output_dir: PathLike | None = None,
        report_format: str = "json",
    ) -> BenchmarkResult:
        return benchmark(
            input_path,
            message,
            engine=self.engine,
            attacks=attacks,
            output_dir=output_dir,
            report_format=report_format,
        )

    def benchmark_compare(
        self,
        input_path: PathLike,
        message: str,
        *,
        engines: list[str] | tuple[str, ...],
        attacks: list[str] | tuple[str, ...] | None = None,
        output_dir: PathLike | None = None,
        report_format: str = "json",
    ) -> BenchmarkComparisonResult:
        return benchmark_compare(
            input_path,
            message,
            engines=engines,
            attacks=attacks,
            output_dir=output_dir,
            report_format=report_format,
        )

    def extract(self, input_path: PathLike) -> ExtractResult:
        return extract(input_path, engine=self.engine)

    def extract_directory(
        self,
        input_dir: PathLike,
        *,
        recursive: bool = False,
        workers: int = 1,
        progress: Callable[[int, int, Path], None] | None = None,
    ) -> BatchResult:
        return extract_directory(
            input_dir,
            engine=self.engine,
            recursive=recursive,
            workers=workers,
            progress=progress,
        )

    def verify(self, input_path: PathLike, expected: str) -> VerifyResult:
        return verify(input_path, expected, engine=self.engine)

    def verify_directory(
        self,
        input_dir: PathLike,
        expected: str,
        *,
        recursive: bool = False,
        workers: int = 1,
        progress: Callable[[int, int, Path], None] | None = None,
    ) -> BatchResult:
        return verify_directory(
            input_dir,
            expected,
            engine=self.engine,
            recursive=recursive,
            workers=workers,
            progress=progress,
        )

    def info(self, input_path: PathLike) -> InfoResult:
        return info(input_path, engine=self.engine)

    def info_directory(
        self,
        input_dir: PathLike,
        *,
        recursive: bool = False,
        workers: int = 1,
        progress: Callable[[int, int, Path], None] | None = None,
    ) -> BatchResult:
        return info_directory(
            input_dir,
            engine=self.engine,
            recursive=recursive,
            workers=workers,
            progress=progress,
        )


def _default_output_path(input_path: PathLike) -> Path:
    source = Path(input_path)
    return source.with_name(f"{source.stem}_wm{source.suffix}")
