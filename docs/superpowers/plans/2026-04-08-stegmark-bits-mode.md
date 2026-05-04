# StegMark Bits Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit bits payload support to watermark embedding and extraction, including CLI `--bits` and `extract --mode bits`.

**Architecture:** Extend the codec with reversible hex-to-bit helpers while preserving the existing framed payload format. Service and CLI layers should support either text or bits input without duplicating watermark logic.

**Tech Stack:** Python 3.10+, Click, stdlib, existing codec/service layers

---

### Task 1: Add codec and service support for bits payloads

**Files:**
- Modify: `src/stegmark/core/codec.py`
- Modify: `src/stegmark/service.py`
- Create: `tests/test_bits_mode.py`

- [ ] **Step 1: Write the failing tests**

```python
from stegmark.core.codec import decode_bitstream, encode_bits_hex


def test_encode_bits_hex_round_trip() -> None:
    bits = encode_bits_hex("deadbeef")
    decoded = decode_bitstream(bits)
    assert decoded.payload == bytes.fromhex("deadbeef")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bits_mode.py -q`
Expected: FAIL because the helper does not exist yet

- [ ] **Step 3: Implement bits helpers and service support**

```python
def encode_bits_hex(hex_payload: str) -> list[int]:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bits_mode.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stegmark/core/codec.py src/stegmark/service.py tests/test_bits_mode.py
git commit -m "feat: add bits payload support"
```

### Task 2: Add CLI bits mode

**Files:**
- Modify: `src/stegmark/cli.py`
- Modify: `src/stegmark/__init__.py`
- Modify: `README.md`
- Create: `tests/test_bits_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
from click.testing import CliRunner

from stegmark.cli import main


def test_embed_bits_and_extract_bits(sample_image_path, tmp_path) -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bits_cli.py -q`
Expected: FAIL because CLI does not expose bits mode yet

- [ ] **Step 3: Implement CLI options**

```python
@click.option("--bits", "-b", ...)
@click.option("--mode", type=click.Choice(["text", "bits"]), ...)
```

- [ ] **Step 4: Update README**

```markdown
stegmark embed input.png --bits deadbeef
stegmark extract output.png --mode bits
```

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_bits_cli.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/stegmark/cli.py src/stegmark/__init__.py README.md tests/test_bits_cli.py
git commit -m "feat: add bits mode cli"
```
