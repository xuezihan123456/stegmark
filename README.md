# StegMark

StegMark is a lightweight invisible watermark tool for still images.

The current branch ships:

- a built-in `native` frequency-domain engine that works offline with only NumPy and Pillow
- an optional Adobe `trustmark` adapter if you install the extra dependency yourself
- a `hidden` self-hosted HiDDeN-style training/export scaffold for future ONNX-backed runtime use

Current development version: `0.3.0a1`

## Features

- `stegmark embed`: write an invisible watermark into a PNG, JPEG, or WebP image
- `stegmark extract`: read a watermark back out of an image
- `stegmark verify`: compare an embedded watermark against an expected message
- `stegmark info`: inspect whether a watermark is present and report image metadata
- Python API with top-level helpers and a reusable `StegMark` session object
- Typed exceptions and dataclasses for later engine expansion
- PyTorch training/export scaffolding for the `hidden` engine path

## Requirements

- Python 3.10+

## Installation

Core package:

```bash
pip install .
```

Development setup:

```bash
pip install -e .[dev]
```

Optional TrustMark adapter:

```bash
pip install -e .[trustmark]
```

Optional training stack:

```bash
pip install -e .[train]
```

The `train` extra includes the ONNX export dependency chain used by `scripts/export_onnx.py`.
On Python 3.13, the ONNX export stack and `trustmark` currently have incompatible upstream NumPy constraints, so use separate environments if you need both.

Optional hidden runtime stack:

```bash
pip install -e .[hidden]
```

Optional GPU runtime:

```bash
pip install -e .[gpu]
```

## CLI Quick Start

Embed a message:

```bash
stegmark embed input.png -m "Alice 2026" -o output.png
```

Control output format and overwrite behavior:

```bash
stegmark embed input.png -m "Alice 2026" --format jpeg --quality 90 --overwrite -o output.jpg
```

Generate a compare report:

```bash
stegmark embed input.png -m "Alice 2026" --compare -o output.png
```

Embed a raw hex payload:

```bash
stegmark embed input.png --bits deadbeef -o output.png
```

Extract a message:

```bash
stegmark extract output.png
```

Extract the raw payload as hex:

```bash
stegmark extract output.png --mode bits
```

Verify a message:

```bash
stegmark verify output.png -m "Alice 2026"
```

Inspect watermark metadata:

```bash
stegmark info output.png
```

Run a benchmark report:

```bash
stegmark benchmark input.png -m "Alice 2026" --attacks jpeg_q90,brightness_1.3 --output-dir ./report --report-format html
```

Compare engines in one run:

```bash
stegmark benchmark input.png -m "Alice 2026" --engines native,hidden --attacks jpeg_q90 --json
```

Fail the command when quality gates are not met:

```bash
stegmark benchmark input.png -m "Alice 2026" --attacks jpeg_q90 --min-average-bit-accuracy 0.95
```

Manage persistent defaults:

```bash
stegmark config show
stegmark config set engine native
stegmark config reset --yes
```

Batch embed into a separate output directory:

```bash
stegmark embed ./photos -m "Alice 2026" -o ./protected -r -w 4
```

Use the built-in engine explicitly:

```bash
stegmark embed input.png -m "Alice 2026" -e native
```

The `hidden` backend is recognized, but it requires exported ONNX files at runtime:

```bash
stegmark extract output.png -e hidden
```

## Python API

```python
import stegmark

result = stegmark.embed("input.png", "Alice 2026", output="output.png", engine="native")
print(result.output_path)

extracted = stegmark.extract("output.png", engine="native")
print(extracted.message)

verified = stegmark.verify("output.png", "Alice 2026", engine="native")
print(verified.matched)

benchmark = stegmark.benchmark(
    "input.png",
    "Alice 2026",
    engine="native",
    attacks=["jpeg_q90", "brightness_1.3"],
)
print(benchmark.attack_results["jpeg_q90"].bit_accuracy)

raw = stegmark.embed("input.png", bits="deadbeef", output="bits-output.png", engine="native")
decoded = stegmark.extract("bits-output.png", engine="native")
print(decoded.payload_hex)
```

Reusable session:

```python
from stegmark import StegMark

with StegMark(engine="native", strength=1.0) as client:
    client.embed("input.png", "Studio X", output="session.png")
    extracted = client.extract("session.png")
    print(extracted.message)
```

## Engines

| Engine | Status | Install | Bits mode | Strength |
|--------|--------|---------|-----------|----------|
| `native` (DCT) | ✅ Ready | `pip install stegmark` | ✅ | ✅ |
| `hidden` (HiDDeN ONNX) | ⚠️ Needs model weights | `pip install stegmark[hidden]` | ✅ | ❌ |
| `trustmark` | ✅ Ready | `pip install stegmark[trustmark]` | ❌ text only | ❌ |

## Engine Notes

- `auto` currently resolves to the built-in `native` engine for predictable offline behavior.
- `trustmark` is an explicit opt-in backend. It requires the external `trustmark` package and whatever runtime that package needs.
- `trustmark` text mode currently accepts ASCII only and, under the default BCH-5 profile, safely supports up to 8 ASCII characters.
- `hidden` now includes PyTorch model definitions, a dataset helper, a trainer scaffold, ONNX export scripts, and an ONNX runtime adapter.
- `hidden` runtime lookup is wired, but it still requires exported `encoder.onnx` and `decoder.onnx` files under `~/.stegmark/models/hidden/`.
- Without those files, `hidden` raises a clear `EngineUnavailableError` instead of silently falling back.
- The current native engine is tuned for deterministic round-trip behavior on normal still images; benchmark tooling and stronger robustness targets are planned for later milestones.

## Hidden Workflow

Export starter ONNX graphs:

```bash
python scripts/export_onnx.py --output-dir ./.artifacts/hidden
```

Run a single training step against a local image directory:

```bash
python scripts/train_hidden.py --config configs/hidden_v02.yaml --dataset ./images
```

## Benchmark Gates

Evaluate a saved benchmark report from CI or a local run:

```bash
python scripts/check_benchmark_gate.py --report benchmark-report/benchmark.json --min-average-bit-accuracy 0.9
```

The repository also includes starter GitHub Actions workflows:

- `.github/workflows/ci.yml`
- `.github/workflows/benchmark.yml`
