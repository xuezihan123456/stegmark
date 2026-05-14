# Changelog

All notable changes to StegMark will be documented in this file.

## [Unreleased]

### Added — v0.4 innovation track

- **Watson DCT JND perceptual mask** (`stegmark.core.jnd_mask`)
  Adaptive per-block embed strength with luminance + contrast masking.
- **Forward-error-correction layer** (`stegmark.core.ecc`)
  Hamming(7,4) + repetition codec with `NONE / LIGHT / MEDIUM / HEAVY` levels.
  Tolerates ~20% bit-error rate at HEAVY.
- **Resumable batch manifest** (`stegmark.core.batch_manifest`)
  Thread-safe SQLite store keyed by SHA-256 of inputs; supports skip-if-done semantics.
- **C2PA-compatible Content Credentials** (`stegmark.core.c2pa_manifest`)
  Minimal-compat JSON manifest with HMAC-signed hash chain; emits actions /
  creator / watermark / AI-generated assertions.
- **Multi-modal watermark engines**
  - `stegmark.core.text_engine` — Kirchenbauer-style first-letter case bias for text.
  - `stegmark.core.audio_engine` — Echo-hiding for 16-bit mono PCM WAV.
- **Neural model fingerprint** (`stegmark.core.model_fingerprint`)
  HMAC-derived orthogonal basis embedded in feature space; z-score detection
  + verification API for "model output → owner" attribution.
- **Anti-AI-training cloak** (`stegmark.core.anti_train`)
  Glaze/Nightshade-style frequency-domain perturbation (PSNR > 34 dB) that
  preserves StegMark's mid-band watermark region.
- **Zero-knowledge ownership proof** (`stegmark.core.zk_ownership`)
  Schnorr Sigma-protocol Pedersen commitment on secp256k1 (stdlib only) +
  circom 2.1 circuit template under `circuits/ownership.circom`.
- **Pyodide browser API** (`stegmark.wasm.pyodide_api`) and
  `demo/wasm/index.html` for in-browser embed/extract without uploading images.
- **Differentiable adversarial training pipeline**
  - `stegmark.training.noise_layers` — DifferentiableJPEG, Gaussian, PrintScan,
    ScreenShoot, CropFlipRotate, NoisePool (PyTorch, optional).
  - `stegmark.training.adversarial_pipeline.WatermarkTrainer` — encoder /
    decoder / discriminator joint training loop.

### Tests
- 80+ new tests across 10 modules. Full suite: 321 passed, 5 skipped, 4 xfailed.

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
