# StegMark Batch Output Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow directory `embed` runs to write outputs into a separate output directory via `-o/--output`.

**Architecture:** Reuse the existing `embed_directory(..., output_dir=...)` service path and reinterpret the CLI `output` path as an output directory only when the input is a directory. Preserve relative subdirectory structure under that target directory.

**Tech Stack:** Python 3.10+, Click, pathlib

---

### Task 1: Add CLI tests and wiring

**Files:**
- Modify: `src/stegmark/cli.py`
- Modify: `README.md`
- Modify: `tests/test_batch_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
def test_embed_directory_can_target_output_dir(...):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_cli.py -q`
Expected: FAIL because directory mode rejects `--output`

- [ ] **Step 3: Implement output-dir behavior**

```python
if input_path.is_dir():
    target_dir = output or input_path
    ...
```

- [ ] **Step 4: Update README**

```markdown
stegmark embed ./photos -m "Alice 2026" -o ./protected -r
```

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_batch_cli.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/stegmark/cli.py README.md tests/test_batch_cli.py
git commit -m "feat: add batch output directory support"
```
