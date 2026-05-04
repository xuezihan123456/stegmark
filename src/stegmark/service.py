from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from stegmark.core.codec import encode_bits_hex, encode_text
from stegmark.core.image_io import load_image, save_image
from stegmark.core.registry import get_engine
from stegmark.evaluation.benchmark import benchmark_compare_engines, benchmark_file
from stegmark.evaluation.metrics import compute_psnr, save_compare_report, save_diff_image
from stegmark.exceptions import InvalidInputError
from stegmark.logging_utils import logger
from stegmark.types import (
    BatchItemResult,
    BatchResult,
    EmbedResult,
    ExtractResult,
    InfoResult,
    PathLike,
    VerifyResult,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_WORKERS = 32
ProgressCallback = Callable[[int, int, Path], None]


def embed_file(
    input_path: PathLike,
    output_path: PathLike,
    *,
    message: str | None = None,
    bits_hex: str | None = None,
    engine: str = "auto",
    strength: float = 1.0,
    output_format: str | None = None,
    quality: int = 95,
    overwrite: bool = False,
    compare: bool = False,
    allowed_output_root: PathLike | None = None,
) -> EmbedResult:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {output_path}")
    loaded = load_image(input_path)
    backend = get_engine(engine)
    started = perf_counter()
    normalized_bits_hex = _normalize_bits_hex(bits_hex)
    payload_bits = tuple(encode_bits_hex(normalized_bits_hex)) if normalized_bits_hex is not None else None
    display_message = message if message is not None else normalized_bits_hex or ""
    payload = bytes.fromhex(normalized_bits_hex) if normalized_bits_hex is not None else None
    encoded = backend.encode(
        loaded.array,
        message,
        payload_bits=payload_bits,
        strength=strength,
    )
    elapsed = perf_counter() - started
    result = EmbedResult(
        image=encoded,
        engine=backend.name,
        message=display_message,
        bits=payload_bits or tuple(encode_text(message or "")),
        payload=payload,
        metadata=loaded.metadata,
        elapsed=elapsed,
    )
    saved_path = save_embed_result(
        result,
        output_path,
        format_name=output_format,
        quality=quality,
        allowed_root=allowed_output_root,
    )
    if not compare:
        return replace(result, output_path=saved_path)
    psnr = compute_psnr(loaded.array, encoded)
    diff_image = save_diff_image(
        loaded.array,
        encoded,
        saved_path.with_name(f"{saved_path.stem}_diff.png"),
    )
    compare_report = save_compare_report(
        input_path=input_path,
        output_path=saved_path,
        psnr=psnr,
        diff_image_path=diff_image,
        report_path=saved_path.with_name(f"{saved_path.stem}_compare.json"),
    )
    return replace(
        result,
        output_path=saved_path,
        psnr=psnr,
        compare_report=compare_report,
        diff_image=diff_image,
    )


def save_embed_result(
    result: EmbedResult,
    path: PathLike,
    *,
    format_name: str | None = None,
    quality: int = 95,
    allowed_root: PathLike | None = None,
) -> Path:
    target = Path(path)
    save_image(
        target,
        result.image,
        metadata=result.metadata,
        format_name=format_name,
        quality=quality,
        allowed_root=allowed_root,
    )
    return target


def embed_directory(
    input_dir: PathLike,
    *,
    message: str | None = None,
    bits_hex: str | None = None,
    engine: str = "auto",
    strength: float = 1.0,
    recursive: bool = False,
    output_dir: PathLike | None = None,
    workers: int = 1,
    overwrite: bool = False,
    output_format: str | None = None,
    quality: int = 95,
    progress: ProgressCallback | None = None,
    allowed_output_root: PathLike | None = None,
) -> BatchResult:
    source_dir = Path(input_dir)
    target_dir = Path(output_dir) if output_dir is not None else source_dir
    target_root = Path(allowed_output_root) if allowed_output_root is not None else target_dir
    _ensure_directory_within_root(target_dir, target_root)
    files = _collect_image_files(source_dir, recursive=recursive)
    worker = partial(
        _embed_directory_item,
        source_dir=source_dir,
        target_dir=target_dir,
        target_root=target_root,
        message=message,
        bits_hex=bits_hex,
        engine=engine,
        strength=strength,
        overwrite=overwrite,
        output_format=output_format,
        quality=quality,
    )
    items = _map_paths(
        files,
        worker,
        workers=_clamp_workers(workers),
        use_processes=_should_use_processes(engine),
        progress=progress,
    )
    return BatchResult(items=tuple(items))


def extract_file(input_path: PathLike, *, engine: str = "auto") -> ExtractResult:
    loaded = load_image(input_path)
    backend = get_engine(engine)
    return backend.decode(loaded.array)


def extract_directory(
    input_dir: PathLike,
    *,
    engine: str = "auto",
    recursive: bool = False,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> BatchResult:
    source_dir = Path(input_dir)
    files = _collect_image_files(source_dir, recursive=recursive)
    worker = partial(_extract_directory_item, engine=engine)
    items = _map_paths(
        files,
        worker,
        workers=_clamp_workers(workers),
        use_processes=_should_use_processes(engine),
        progress=progress,
    )
    return BatchResult(items=tuple(items))


def verify_file(input_path: PathLike, expected: str, *, engine: str = "auto") -> VerifyResult:
    extracted = extract_file(input_path, engine=engine)
    return VerifyResult(
        matched=extracted.message == expected,
        engine=extracted.engine,
        expected=expected,
        actual=extracted.message,
        confidence=extracted.confidence,
    )


def verify_directory(
    input_dir: PathLike,
    *,
    expected: str,
    engine: str = "auto",
    recursive: bool = False,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> BatchResult:
    source_dir = Path(input_dir)
    files = _collect_image_files(source_dir, recursive=recursive)
    worker = partial(_verify_directory_item, expected=expected, engine=engine)
    items = _map_paths(
        files,
        worker,
        workers=_clamp_workers(workers),
        use_processes=_should_use_processes(engine),
        progress=progress,
    )
    return BatchResult(items=tuple(items))


def info_file(input_path: PathLike, *, engine: str = "auto") -> InfoResult:
    loaded = load_image(input_path)
    backend = get_engine(engine)
    extracted = backend.decode(loaded.array)
    return InfoResult(
        found=extracted.found,
        engine=extracted.engine,
        width=loaded.width,
        height=loaded.height,
        format=loaded.metadata.format,
        confidence=extracted.confidence,
    )


def info_directory(
    input_dir: PathLike,
    *,
    engine: str = "auto",
    recursive: bool = False,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> BatchResult:
    source_dir = Path(input_dir)
    files = _collect_image_files(source_dir, recursive=recursive)
    worker = partial(_info_directory_item, engine=engine)
    items = _map_paths(
        files,
        worker,
        workers=_clamp_workers(workers),
        use_processes=_should_use_processes(engine),
        progress=progress,
    )
    return BatchResult(items=tuple(items))


def benchmark_file_service(
    input_path: PathLike,
    *,
    message: str,
    engine: str = "native",
    attacks: list[str] | tuple[str, ...] | None = None,
    output_dir: PathLike | None = None,
    report_format: str = "json",
):
    return benchmark_file(
        input_path,
        message=message,
        engine=engine,
        attacks=attacks,
        output_dir=output_dir,
        report_format=report_format,
    )


def benchmark_compare_service(
    input_path: PathLike,
    *,
    message: str,
    engines: list[str] | tuple[str, ...],
    attacks: list[str] | tuple[str, ...] | None = None,
    output_dir: PathLike | None = None,
    report_format: str = "json",
):
    return benchmark_compare_engines(
        input_path,
        message=message,
        engines=engines,
        attacks=attacks,
        output_dir=output_dir,
        report_format=report_format,
    )


def _embed_directory_item(
    path: Path,
    *,
    source_dir: Path,
    target_dir: Path,
    target_root: Path,
    message: str | None,
    bits_hex: str | None,
    engine: str,
    strength: float,
    overwrite: bool,
    output_format: str | None,
    quality: int,
) -> BatchItemResult:
    relative = path.relative_to(source_dir)
    target = _default_output_path(target_dir / relative, output_format=output_format)
    if target.exists() and not overwrite:
        logger.warning("batch embed skipped for %s: output exists", path)
        return BatchItemResult(
            input_path=path,
            output_path=target,
            success=False,
            error="output exists (use overwrite to replace)",
        )
    try:
        result = embed_file(
            path,
            target,
            message=message,
            bits_hex=bits_hex,
            engine=engine,
            strength=strength,
            output_format=output_format,
            quality=quality,
            overwrite=overwrite,
            allowed_output_root=target_root,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("batch embed failed for %s: %s", path, exc)
        return BatchItemResult(
            input_path=path,
            output_path=target,
            success=False,
            error=str(exc),
        )
    result = _strip_embed_image(result)
    return BatchItemResult(
        input_path=path,
        output_path=target,
        success=True,
        result=result,
    )


def _extract_directory_item(path: Path, *, engine: str) -> BatchItemResult:
    try:
        result = extract_file(path, engine=engine)
    except Exception as exc:  # pragma: no cover
        logger.warning("batch extract failed for %s: %s", path, exc)
        return BatchItemResult(input_path=path, success=False, error=str(exc))
    return BatchItemResult(
        input_path=path,
        success=result.found,
        result=result,
        error=None if result.found else result.error,
    )


def _verify_directory_item(path: Path, *, expected: str, engine: str) -> BatchItemResult:
    try:
        result = verify_file(path, expected, engine=engine)
    except Exception as exc:  # pragma: no cover
        logger.warning("batch verify failed for %s: %s", path, exc)
        return BatchItemResult(input_path=path, success=False, error=str(exc))
    return BatchItemResult(
        input_path=path,
        success=result.matched,
        result=result,
        error=None if result.matched else "verification did not match expected message",
    )


def _info_directory_item(path: Path, *, engine: str) -> BatchItemResult:
    try:
        result = info_file(path, engine=engine)
    except Exception as exc:  # pragma: no cover
        logger.warning("batch info failed for %s: %s", path, exc)
        return BatchItemResult(input_path=path, success=False, error=str(exc))
    return BatchItemResult(
        input_path=path,
        success=result.found,
        result=result,
        error=None if result.found else "watermark not found",
    )


def _collect_image_files(root: Path, *, recursive: bool) -> list[Path]:
    return sorted(path for path in _iter_image_files(root, recursive=recursive))


def _iter_image_files(root: Path, *, recursive: bool) -> Iterable[Path]:
    for path in root.iterdir():
        if path.is_symlink():
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
        elif recursive and path.is_dir():
            yield from _iter_image_files(path, recursive=recursive)


def _default_output_path(path: Path, *, output_format: str | None = None) -> Path:
    if output_format is None:
        suffix = path.suffix
    else:
        normalized = output_format.lower()
        suffix = ".jpg" if normalized in {"jpg", "jpeg"} else f".{normalized}"
    return path.with_name(f"{path.stem}_wm{suffix}")


def _map_paths(
    paths: list[Path],
    worker: Callable[[Path], BatchItemResult],
    *,
    workers: int,
    use_processes: bool,
    progress: ProgressCallback | None = None,
) -> list[BatchItemResult]:
    if workers <= 1:
        items = [worker(path) for path in paths]
        _emit_progress(items, progress)
        return items

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    try:
        with executor_cls(max_workers=workers) as executor:
            items = list(executor.map(worker, paths))
    except PermissionError:
        if executor_cls is not ProcessPoolExecutor:
            raise
        with ThreadPoolExecutor(max_workers=workers) as executor:
            items = list(executor.map(worker, paths))
    _emit_progress(items, progress)
    return items


def _emit_progress(items: list[BatchItemResult], progress: ProgressCallback | None) -> None:
    if progress is None:
        return
    total = len(items)
    for index, item in enumerate(items, start=1):
        progress(index, total, item.input_path)


def _clamp_workers(workers: int) -> int:
    return max(1, min(workers, MAX_WORKERS))


def _should_use_processes(engine: str) -> bool:
    return engine.lower() in {"auto", "native"}


def _normalize_bits_hex(bits_hex: str | None) -> str | None:
    if bits_hex is None:
        return None
    normalized = bits_hex.strip().lower()
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    return normalized


def _strip_embed_image(result: EmbedResult) -> EmbedResult:
    if not isinstance(result, EmbedResult):
        return result
    if result.image.size == 0:
        return result
    return replace(result, image=np.empty((0, 0, 3), dtype=result.image.dtype))


def _ensure_directory_within_root(directory: Path, root: Path) -> None:
    resolved_directory = directory.resolve()
    resolved_root = root.resolve()
    try:
        resolved_directory.relative_to(resolved_root)
    except ValueError as exc:
        raise InvalidInputError(
            f"output path escapes allowed directory: {resolved_directory}",
            hint=f"Write outputs under {resolved_root}.",
        ) from exc
