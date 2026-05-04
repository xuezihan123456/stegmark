# Changelog

All notable changes to StegMark will be documented in this file.

## [Unreleased]

## [0.3.0a1] - 2026-04-14

### Added
- TrustMark engine support (`pip install stegmark[trustmark]`)
- HiDDeN ONNX engine with thread-safe lazy loading and checkpoint export
- Engine-level configuration (`[engines.hidden]` in config.toml)
- Batch processing progress callback (`progress` parameter)
- Output path boundary check to prevent path traversal
- Decompression bomb protection (50 MB / 50 MP limit)
- `MAX_WORKERS = 32` cap on thread/process pool size
- `save_embed_result()` as a standalone function (replaces `EmbedResult.save()`)
- `logging_utils.py` — structured logging across the library

### Fixed
- CLI `--strength 0.0` silently ignored due to `or` short-circuit (H1)
- `bits_hex` with `0x` prefix crashed `bytes.fromhex()` (H2)
- `HiddenEngine` lazy ONNX session initialization had a thread race (H6)
- `ImageMetadata.extras` was mutable despite frozen dataclass (H5)
- `trustmark` backend now rejects non-ASCII or over-capacity text instead of silently mangling payloads
- Symlink traversal in batch directory walk (SM5)

### Performance
- DCT watermarking vectorised with `as_strided` + batched `np.matmul` (20–50× faster)
- Batch processing now uses `ProcessPoolExecutor` for CPU-bound native engine

### Security
- Image file size validated before `PIL.Image.open()` (SH1)
- `workers` parameter capped at `MAX_WORKERS = 32` (SH2)
- Output path restricted to allowed root directory (SH3)

## [0.2.0] - 2026-04-07

### Added
- Initial HiDDeN ONNX engine scaffold
- Batch `embed_directory` / `extract_directory` with `ThreadPoolExecutor`
- `benchmark` command with attack simulation and gate evaluation
- `config` command with TOML persistence
- JSON output flag (`--json`) for all CLI commands

## [0.1.0] - 2026-04-07

### Added
- Initial public package scaffold.
- Click-based CLI entrypoint.
- Typed Python API.
- Native invisible watermark round-trip workflow.
- Image I/O, exceptions, config, and codec primitives.
