# StegMark Compare Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add single-file compare reporting for `stegmark embed --compare`, including image quality metrics and a saved diff artifact.

**Architecture:** Keep quality metrics and diff rendering under the evaluation layer, and keep embed orchestration in the service layer. The CLI should only trigger compare mode for single-file embeds and report the generated artifact paths.

**Tech Stack:** Python 3.10+, NumPy, Pillow, Click

---

### Task 1: Add metrics and compare report generation

**Files:**
- Create: `src/stegmark/evaluation/metrics.py`
- Modify: `src/stegmark/types.py`
- Modify: `src/stegmark/service.py`
- Create: `tests/test_compare_mode.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from stegmark.service import embed_file


def test_embed_file_compare_generates_report(sample_image_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "output.png"
    result = embed_file(
        sample_image_path,
        output,
        message="Alice 2026",
        engine="native",
        compare=True,
    )
    assert result.psnr is not None
    assert result.compare_report is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare_mode.py -q`
Expected: FAIL because compare mode is not implemented yet

- [ ] **Step 3: Implement minimal compare reporting**

```python
def compute_psnr(reference: ImageArray, candidate: ImageArray) -> float:
    ...
```

```python
def embed_file(..., compare: bool = False) -> EmbedResult:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compare_mode.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stegmark/evaluation/metrics.py src/stegmark/types.py src/stegmark/service.py tests/test_compare_mode.py
git commit -m "feat: add embed compare reporting"
```

### Task 2: Add CLI `--compare`

**Files:**
- Modify: `src/stegmark/cli.py`
- Modify: `README.md`
- Create: `tests/test_compare_cli.py`

- [ ] **Step 1: Write the failing CLI test**

```python
from click.testing import CliRunner

from stegmark.cli import main


def test_embed_compare_outputs_report(sample_image_path, tmp_path) -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare_cli.py -q`
Expected: FAIL because `--compare` is not wired yet

- [ ] **Step 3: Implement CLI support**

```python
@click.option("--compare", is_flag=True)
```

- [ ] **Step 4: Update README**

```markdown
stegmark embed input.png -m "Alice 2026" --compare
```

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_compare_cli.py tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/stegmark/cli.py README.md tests/test_compare_cli.py
git commit -m "feat: add compare cli option"
```
