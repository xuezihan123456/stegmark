# StegMark Batch Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add directory batch processing, recursive traversal, and worker-controlled parallel execution for `embed` and `extract`.

**Architecture:** Keep filesystem traversal and parallel execution in the service layer so the CLI remains thin. Batch operations should reuse the existing single-file functions, return structured per-file results, and avoid silent skipping.

**Tech Stack:** Python 3.10+, Click, stdlib `concurrent.futures`, pathlib

---

### Task 1: Add batch service helpers and tests

**Files:**
- Modify: `src/stegmark/types.py`
- Modify: `src/stegmark/service.py`
- Create: `tests/test_batch_service.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from stegmark.service import embed_directory, extract_directory


def test_embed_directory_processes_multiple_files(sample_image_path: Path, tmp_path: Path) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    for name in ("a.png", "b.png"):
        (source_dir / name).write_bytes(sample_image_path.read_bytes())

    result = embed_directory(source_dir, message="Alice 2026", engine="native")

    assert result.total == 2
    assert result.succeeded == 2


def test_extract_directory_supports_recursive(tmp_path: Path, sample_image_path: Path) -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_service.py -q`
Expected: FAIL because batch helpers do not exist yet

- [ ] **Step 3: Implement batch result types and service functions**

```python
@dataclass(frozen=True)
class BatchItemResult:
    ...
```

```python
def embed_directory(...):
    ...


def extract_directory(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_batch_service.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stegmark/types.py src/stegmark/service.py tests/test_batch_service.py
git commit -m "feat: add batch service helpers"
```

### Task 2: Add CLI support for directory inputs

**Files:**
- Modify: `src/stegmark/cli.py`
- Create: `tests/test_batch_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing CLI tests**

```python
from click.testing import CliRunner

from stegmark.cli import main


def test_embed_directory_command(sample_image_path, tmp_path) -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_cli.py -q`
Expected: FAIL because the CLI does not support directory batch mode yet

- [ ] **Step 3: Implement CLI options**

```python
@click.option("--recursive", "-r", is_flag=True)
@click.option("--workers", "-w", type=int, default=None)
```

- [ ] **Step 4: Update README examples**

```markdown
stegmark embed ./photos -m "Alice 2026" -r -w 4
```

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_batch_cli.py tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/stegmark/cli.py tests/test_batch_cli.py README.md
git commit -m "feat: add batch cli processing"
```
