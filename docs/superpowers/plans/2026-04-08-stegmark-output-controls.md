# StegMark Output Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--overwrite`, `--format`, and `--quality` controls for embed operations in both single-file and directory modes.

**Architecture:** Keep output-format decisions in `image_io.py` and pass them through service and CLI layers without duplicating save logic. Directory mode should respect overwrite policy per file.

**Tech Stack:** Python 3.10+, Click, Pillow, pathlib

---

### Task 1: Add save format/quality controls in service and image I/O

**Files:**
- Modify: `src/stegmark/core/image_io.py`
- Modify: `src/stegmark/service.py`
- Create: `tests/test_output_controls.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from stegmark.service import embed_file


def test_embed_file_can_force_jpeg_output(sample_image_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "output.jpg"
    result = embed_file(sample_image_path, output, message="Alice 2026", engine="native", output_format="jpeg")
    assert result.output_path == output
    assert output.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_controls.py -q`
Expected: FAIL because service/image I/O do not accept these parameters yet

- [ ] **Step 3: Implement output controls**

```python
def save_image(..., format_name: str | None = None, quality: int = 95) -> Path:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_output_controls.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stegmark/core/image_io.py src/stegmark/service.py tests/test_output_controls.py
git commit -m "feat: add output format controls"
```

### Task 2: Add CLI flags and overwrite handling

**Files:**
- Modify: `src/stegmark/cli.py`
- Create: `tests/test_output_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing CLI tests**

```python
from click.testing import CliRunner

from stegmark.cli import main


def test_embed_command_supports_overwrite(sample_image_path, tmp_path) -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_cli.py -q`
Expected: FAIL because the CLI flags do not exist yet

- [ ] **Step 3: Implement CLI flags**

```python
@click.option("--overwrite", "-y", is_flag=True)
@click.option("--format", "-f", ...)
@click.option("--quality", "-Q", ...)
```

- [ ] **Step 4: Update README examples**

```markdown
stegmark embed input.png -m "Alice 2026" --format jpeg --quality 90 --overwrite
```

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_output_cli.py tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/stegmark/cli.py tests/test_output_cli.py README.md
git commit -m "feat: add embed output flags"
```
